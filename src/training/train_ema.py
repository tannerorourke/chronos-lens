#!/usr/bin/env python3
"""
EMA JEPA training pipeline for longitudinal patient sequences.

Architecture
------------
  encoder        : EncounterEncoder - online context path (with grads)
  target_encoder : EMA copy of encoder (momentum -> 1.0), no backprop
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
from src.training.utils.datasets import MimicDataset, collate_fn
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import GradientMonitor, TrainingLogger
from src.training.utils.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.analysis.displacement import save_embedding_vecs
from src.utils.io import load_sequences, build_vocab, EXPERIMENTS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:

    # --- most params are sent to respective functions ---
    # --- optimization ---
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    ema         = opt_params["ema"]

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
    assert type(model) == JEPA_EMA
    
    # --- init optimizer / scheduler / scaler ---
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params,
        ipe=len(loader),
        num_epochs=epochs,
        use_bfloat16=use_bfloat16)
    
    # --- momentum schedule ---
    ipe = len(loader)
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0]) / (ipe*epochs)
        for i in range(ipe*epochs + 1))


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
        for _ in range((start_epoch - 1) * ipe):
            next(momentum_scheduler)
    

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {(n_params / 1e6):.2f}M")

    logger   = TrainingLogger(run_dir, start_epoch, global_step, loss_history)
    grad_mon = GradientMonitor(model)

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
            
            def forward_unto_dawn():
                z_context, z_pred, z_target = model(batch_dev)
                loss = F.smooth_l1_loss(z_pred, z_target)
                return loss

            if use_bfloat16 and scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore
                    loss = forward_unto_dawn()
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_mon.capture()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss = forward_unto_dawn()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_mon.capture()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            # -- EMA update of target encoder (matching iJEPA reference)
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

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
            _, stat_log = save_embedding_vecs(model, loader, device, epoch, emb_dir)

        vicreg_log = {k: 0.0 for k in vicreg_keys}
        logger.log_epoch(
            loss=float(np.mean(epoch_losses)),
            lr=optimizer.param_groups[0]["lr"],
            **vicreg_log, **stat_log, **grad_mon.get_metrics(),
        )
        grad_mon.reset()

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------

    logger.finalize()
