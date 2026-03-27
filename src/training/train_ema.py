#!/usr/bin/env python3
"""
EMA JEPA training pipeline for longitudinal patient sequences.

Architecture
------------
  encoder        : EncounterEncoder - online context path (with grads)
  target_encoder : EMA copy of encoder (tau -> 1.0), no backprop
  predictor      : MLP(z_context + pos_emb -> z_pred)
  loss           : smooth_l1(z_pred, z_target)  (iJEPA)
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.jepa_ema import JEPA_EMA
from src.training.dataset import MimicDataset, collate_fn
from src.training.optimizers import init_optimizers
from src.training.logging import GradientMonitor, TrainingLogger
from src.training.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.analysis.displacement import save_embedding_vecs
from src.utils.io import load_sequences, build_vocab, EXPERIMENTS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    use_cuda = device.type == "cuda"

    # --- optimization ---------------------------------------------------------
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    tau         = opt_params.get("tau", 0.996)
    ema_start   = opt_params.get("ema_start", 0.996)
    ema_end     = opt_params.get("ema_end", 1.0)

    # --- data -----------------------------------------------------------------
    data_p          = params["data"]
    batch_size      = data_p["batch_size"]
    n_patients      = data_p["n_patients"]
    max_encounters  = data_p.get("max_encounters")
    max_tokens      = data_p.get("max_tokens")

    # --- artifacts ------------------------------------------------------------
    art_p               = params["artifacts"]
    use_bfloat16        = art_p.get("use_bfloat16", False)
    checkpoint_every    = art_p.get("checkpoint_every") or epochs
    log_emb_vecs        = art_p.get("log_emb_vecs", True)
    log_emb_vecs_every  = art_p.get("log_emb_vecs_every") or epochs

    # --- meta -----------------------------------------------------------------
    seed = params.get("meta", {}).get("seed")

    # --- init sequences, vocab, dataset, loader -------------------------------
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx=0, dir=run_dir)
    with open(run_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)

    dataset = MimicDataset(patients, vocab, pad_idx=0,
                          max_encounters=max_encounters, max_tokens=max_tokens)
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False,
        num_workers=2, persistent_workers=True,
        pin_memory=use_cuda)

    ckpt_dir = run_dir / "checkpoints"
    emb_dir  = run_dir / "embeddings"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # --- model ----------------------------------------------------------------
    start_epoch, global_step, loss_history = 1, 1, []
    _ckpt = None

    if params.get("resume_from"):
        ckpt_path = EXPERIMENTS_DIR / params["resume_from"]
        model, _ckpt, start_epoch, global_step, loss_history = \
            load_model_checkpoint(ckpt_path, device, restore_rng=True)
        model_params = _ckpt["model_params"]
    else:
        model_params = {
            "architecture":     params["model"].get("architecture", "ema"),
            "embed_dim":        params["model"]["embed_dim"],
            "num_heads":        params["model"]["num_heads"],
            "num_layers":       params["model"]["num_layers"],
            "ffn_dim":          params["model"]["ffn_dim"],
            "max_seq_len":      params["model"]["max_seq_len"],
            "predictor_hidden": params["model"]["predictor_hidden"],
            "tau":              tau,
            "vocab_size":       len(vocab),
        }
        model = build_model(model_params, device)
    assert type(model) == JEPA_EMA

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {(n_params / 1e6):.2f}M")

    # --- init optimizer / scheduler / scaler ----------------------------------
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params,
        ipe=len(loader),
        num_epochs=epochs,
        use_bfloat16=use_bfloat16)

    if _ckpt is not None:
        if _ckpt.get("optimizer"):
            optimizer.load_state_dict(_ckpt["optimizer"])
        if scheduler is not None and _ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(_ckpt["scheduler"])
        if scaler is not None and _ckpt.get("scaler") is not None:
            scaler.load_state_dict(_ckpt["scaler"])

    logger   = TrainingLogger(run_dir, start_epoch, global_step, loss_history)
    grad_mon = GradientMonitor(model)

    # --- momentum schedule---------------------------------------------
    ipe = len(loader)
    total_steps = ipe * epochs
    momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / total_steps
        for i in range(total_steps + 1)
    )

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {ipe} batches (size: {batch_size}) for {epochs} epochs")
    print(f"Description: {params['meta']['tag']}: {params['meta']['description']}")

    vicreg_keys = ("sim", "var_pred", "var_ctx", "cov_pred", "cov_ctx")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        n_batches = 0

        for batch in loader:
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if use_bfloat16 and scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore
                    z_context, z_pred, z_target = model(batch_dev)
                    loss = F.smooth_l1_loss(z_pred, z_target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_mon.capture()
                scaler.step(optimizer)
                scaler.update()

            else:
                z_context, z_pred, z_target = model(batch_dev)
                loss = F.smooth_l1_loss(z_pred, z_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_mon.capture()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            model.set_momentum(next(momentum_scheduler))
            model.update_target_encoder()

            epoch_losses.append(loss.item())
            logger.log_step(loss.item())
            n_batches += 1

        # -- end of epoch --------------------------------------------------
        model.eval()

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(model, model_params,
                            optimizer, scheduler, scaler,
                            epoch, logger.global_step, logger._loss_history,
                            ckpt_dir, seed=seed)

        stat_log = {}
        if log_emb_vecs and (epoch % log_emb_vecs_every == 0 or epoch == epochs or epoch == 1):
            _, stat_log = save_embedding_vecs(model, loader, device, epoch, emb_dir)

        vicreg_log = {k: 0.0 for k in vicreg_keys}
        logger.log_epoch(
            loss=float(np.mean(epoch_losses)),
            lr=optimizer.param_groups[0]["lr"],
            **vicreg_log, **stat_log, **grad_mon.get_metrics(),
        )
        grad_mon.reset()

    # ------------------------------------------------------------------
    # --- POST-TRAINING ------------------------------------------------
    # ------------------------------------------------------------------

    logger.finalize()
