#!/usr/bin/env python3
"""
Stop-gradient JEPA training pipeline for longitudinal patient sequences.

Architecture
------------
  encoder   : EncounterEncoder (shared by context & target paths)
  target    : stop-gradient (no EMA) - same weights, torch.no_grad() + detach()
  predictor : MLP(z_context + pos_emb -> z_pred)
  loss      : MSE(z_pred, z_target) + VICReg variance/covariance regularization
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.jepa_stopgrad import JEPAStopGrad
from src.training.utils.losses import jepa_stopgrad_loss
from src.training.utils.datasets import MimicDataset, collate_fn, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import GradientMonitor, TrainingLogger
from src.training.utils.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.utils.io import load_sequences, save_embedding_vecs, EXPERIMENTS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:

    # --- most params are sent to respective functions ---
    # --- optimization ---
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    sim_weight  = opt_params.get("sim_weight", 1.0)
    var_weight  = opt_params.get("var_weight", 1.0)
    cov_weight  = opt_params.get("cov_weight", 0.04)

    # --- data ---
    data_params     = params["data"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"

    # --- meta ---
    meta_p          = params["meta"]
    seed            = meta_p["seed"]
    use_bfloat16    = meta_p["use_bfloat16"]
    log_vecs        = meta_p["log_vecs"]
    log_vecs_every  = meta_p["log_vecs_every"] or epochs
    checkpoint_every = meta_p["checkpoint_every"] or epochs

    # --- build sequences, vocab, dataset, loader ---
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx=0, dir=run_dir)
    with open(run_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)

    dataset = MimicDataset(patients, vocab, data_params, pad_idx=0)
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False,
        num_workers=2, persistent_workers=True,
        pin_memory=pin_memory)

    ckpt_dir = run_dir / "checkpoints"
    emb_dir  = run_dir / "embeddings"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # --- model ---
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == JEPAStopGrad
    
    # --- init optimizer / scheduler / scaler ---
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params,
        ipe=len(loader),
        num_epochs=epochs,
        use_bfloat16=use_bfloat16)
    
    
    start_epoch, global_step, loss_history = 1, 1, []
    # --- load checkpoint? ---
    if params.get("resume_from"):
        ckpt_path = EXPERIMENTS_DIR / params["resume_from"]
        model, model_params, optimizer, scheduler, scaler, start_epoch, global_step, loss_history = \
            load_model_checkpoint(
                model,
                optimizer, 
                scheduler,
                scaler,
                ckpt_path, 
                device, 
                restore_rng=True)
    

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {(n_params / 1e6):.2f}M")

    logger   = TrainingLogger(run_dir, start_epoch, global_step, loss_history)
    grad_mon = GradientMonitor(model)
    
    vicreg_keys = ("sim", "var_pred", "var_enc", "cov_pred", "cov_enc")

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {len(loader)} batches (size: {batch_size}) for {epochs} epochs")
    print(f"Description: {params['meta']['tag']}: {params['meta']['description']}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_records: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        vicreg_accum = {k: 0.0 for k in vicreg_keys}
        n_batches = 0

        for batch in loader:
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            def forward_unto_dawn():
                z_enc, z_pred, z_target = model(batch_dev)
                for k, v in zip(
                    ["z_encs", "z_pred", "z_target", "subject_ids", "mask_pos", "labels"], 
                    [z_enc, z_pred, z_target, batch_dev["subject_ids"], batch_dev["mask_pos"], batch_dev["labels"]]
                ):
                    epoch_records[k].append(v.detach().cpu().numpy())
                    
                loss_dict = jepa_stopgrad_loss(
                    z_enc, z_pred, z_target, batch_dev["ctx_pad_mask"],
                    sim_weight, var_weight, cov_weight)
                return loss_dict

            if use_bfloat16 and scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore
                    loss_dict = forward_unto_dawn()
                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_mon.capture()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss_dict = forward_unto_dawn()
                loss = loss_dict["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_mon.capture()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            for k in vicreg_keys:
                vicreg_accum[k] += loss_dict[k].detach().item()
            epoch_losses.append(loss.item())
            logger.log_step(loss.item())
            n_batches += 1

        # --- EVAL --------------------------------------------------------------
        model.eval()

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(model, model_params,
                            optimizer, scheduler, scaler,
                            epoch, logger.global_step, logger._loss_history,
                            ckpt_dir, seed=seed)

        stat_log = {}
        if log_vecs and (epoch % log_vecs_every == 0 or epoch == epochs or epoch == 1):
            save_embedding_vecs(epoch_records, epoch, emb_dir)

        vicreg_log = {k: vicreg_accum[k] / max(n_batches, 1) for k in vicreg_keys}
        logger.log_epoch(
            loss=float(np.mean(epoch_losses)),
            lr=optimizer.param_groups[0]["lr"],
            **vicreg_log, **stat_log, **grad_mon.get_metrics(),
        )
        grad_mon.reset()

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------
    logger.finalize()
