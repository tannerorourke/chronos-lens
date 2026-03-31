#!/usr/bin/env python3
"""
Supervised Transformer training pipeline for longitudinal patient sequences.

Architecture
------------
  encoder    : TransformerEncoder (same as JEPA context path)
  classifier : Linear(embed_dim, 1)
  loss       : BCEWithLogitsLoss on logits vs binary labels
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.supervised_transformer import SupervisedTransformer
from src.training.utils.datasets import SupervisedDataset, supervised_collate_fn, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import GradientMonitor, TrainingLogger
from src.training.utils.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.utils.io import load_sequences, EXPERIMENTS_DIR

# =============================================================================
# Embedding extraction
# =============================================================================

@torch.no_grad()
def save_supervised_embeddings(
    all_z_c: list[np.ndarray],
    all_sids: list[str],
    epoch: int,
    save_dir: Path,
) -> None:
    """Save supervised encoder outputs in JEPA .npz format.

    z_context is saved as z_encs (N, 1, D) - the pooled context vector unsqueezed
    to match the JEPA per-encounter layout. z_pred and z_target are zeros.
    """
    z_context   = np.concatenate(all_z_c) # (N, D)
    subject_ids = np.array(all_sids, dtype=str)
    N           = z_context.shape[0]

    file = (save_dir / f"embeddings_{epoch}").with_suffix(".npz")
    np.savez(
        file,
        z_encs=z_context[:, np.newaxis, :], # (N, 1, D)
        z_pred=np.zeros_like(z_context),
        z_target=np.zeros_like(z_context),
        subject_ids=subject_ids,
        mask_pos=np.full(N, -1, dtype=np.int64),
    )
    print(f"   Embeddings saved: {save_dir.name}/{file.name} (epoch {epoch})")


# =============================================================================
# Training loop
# =============================================================================

def main(params: Dict, run_dir: Path, device: torch.device) -> None:

    # --- optimization ---------------------------------------------------------
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]

    # --- data -----------------------------------------------------------------
    data_params     = params["data"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"

    # --- meta -----------------------------------------------------------------
    meta_p          = params["meta"]
    seed            = meta_p["seed"]
    use_bfloat16    = meta_p["use_bfloat16"]
    log_vecs        = meta_p["log_vecs"]
    log_vecs_every  = meta_p["log_vecs_every"] or epochs
    checkpoint_every = meta_p["checkpoint_every"] or epochs

    # --- build sequences, vocab, dataset, loader ------------------------------
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx=0, dir=run_dir)
    with open(run_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)

    dataset = SupervisedDataset(patients, vocab, data_params, pad_idx=0)
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=supervised_collate_fn, drop_last=False,
        num_workers=2, persistent_workers=True,
        pin_memory=pin_memory)

    ckpt_dir = run_dir / "checkpoints"
    emb_dir  = run_dir / "embeddings"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # --- model ----------------------------------------------------------------
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == SupervisedTransformer

    # --- init optimizer / scheduler / scaler ----------------------------------
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params,
        ipe=len(loader),
        num_epochs=epochs,
        use_bfloat16=use_bfloat16)

    start_epoch, global_step, loss_history = 1, 1, []
    # --- load checkpoint? ----------------------------------------------
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

    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {len(loader)} batches (size: {batch_size}) for {epochs} epochs")
    print(f"Description: {params['meta']['tag']}: {params['meta']['description']}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        z_c, sids = [], []

        for batch in loader:
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            def forward_unto_dawn():
                z_context, logits = model(batch_dev)
                z_c.append(z_context.detach().cpu().numpy())
                sids.extend(batch["subject_ids"])
                
                loss = criterion(logits, batch_dev["labels"].float())
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

            epoch_losses.append(loss.item())
            logger.log_step(loss.item())

        # -- end of epoch --------------------------------------------------
        model.eval()

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(model, model_params,
                            optimizer, scheduler, scaler,
                            epoch, logger.global_step, logger._loss_history,
                            ckpt_dir, seed=seed)

        if log_vecs and (epoch % log_vecs_every == 0 or epoch == epochs or epoch == 1):
            save_supervised_embeddings(z_c, sids, epoch, emb_dir)

        logger.log_epoch(
            loss=float(np.mean(epoch_losses)),
            lr=optimizer.param_groups[0]["lr"],
            **grad_mon.get_metrics(),
        )
        grad_mon.reset()

    # ------------------------------------------------------------------
    # --- POST-TRAINING ------------------------------------------------
    # ------------------------------------------------------------------

    logger.finalize()
