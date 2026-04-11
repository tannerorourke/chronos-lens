#!/usr/bin/env python3
"""
Stop-gradient JEPA training pipeline for longitudinal patient sequences.

Architecture
------------
  encoder   : EncounterEncoder (shared by context & target paths)
  target    : stop-gradient (no EMA) - same weights, torch.no_grad() + detach()
  predictor : Transformer(z_enc context tokens + mask token -> z_pred)
  loss      : MSE(z_pred, z_target) + VICReg variance/covariance regularization
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
from pathlib import Path
from typing import Dict

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.jepa_stopgrad import JEPAStopGrad
from src.training.utils.losses import jepa_stopgrad_loss
from src.training.utils.datasets import MimicDataset, collate_fn, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger
from src.training.utils.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.utils.io import load_sequences, EXPERIMENTS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    print("Starting...")

    # --- most params are sent to respective functions
    # --- optimization
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    sim_weight  = opt_params.get("sim_weight", 1.0)
    var_weight  = opt_params.get("var_weight", 1.0)
    cov_weight  = opt_params.get("cov_weight", 0.04)
    accum_steps = opt_params.get("accumulation_steps", 1)
    grad_clip   = opt_params.get("grad_clip", 5.0)

    # --- data
    data_params     = params["data"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    max_encounters  = data_params.get("max_encounters", None)
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"
    num_workers     = data_params.get("num_workers", 0)

    # --- meta
    meta_p          = params["meta"]
    seed            = meta_p["seed"]
    use_bfloat16    = meta_p["use_bfloat16"]
    save_ckpt_every = meta_p.get("save_ckpt_every", epochs)
    save_emb_every  = meta_p.get("save_emb_every", epochs)
    m_tag           = meta_p.get("tag", None)
    m_desc          = meta_p.get("description", None)
    ckpt_dir = run_dir / "checkpoints"
    
    print(f"Starting {m_tag if m_tag else 'up'}..")
    if m_desc:
        print(f"  -- {params['meta'].get('description', '')}")

    # --- build sequences, vocab, dataset, loader
    patients = load_sequences(n=n_patients)
    print(f"Patients: {len(patients)}")
    vocab = build_vocab(patients, pad_idx=0, dir=run_dir, save=False)
    print(f"Vocab: {len(vocab)} tokens")

    dataset = MimicDataset(patients, vocab, data_params, pad_idx=0, max_encounters=max_encounters)
    del patients; gc.collect()
    
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=pin_memory)

    

    # --- model ---
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == JEPAStopGrad
    
    # --- init optimizer / scheduler
    optimizer, scheduler = init_optimizers(
        model, opt_params,
        ipe=len(loader),
        num_epochs=epochs)
    
    
    start_epoch, start_step, loss_history = 1, 1, []
    # --- load checkpoint?
    if params.get("resume_from"):
        ckpt_path = EXPERIMENTS_DIR / params["resume_from"]
        model, model_params, optimizer, scheduler, start_epoch, start_step, loss_history = \
            load_model_checkpoint(
                model,
                optimizer, scheduler, 
                ckpt_path, device, restore_rng=True)

    print("Params:",
          f"Total: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M",
          f"Trainable: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.2f}M")

    # --- training utility
    logger = TrainingLogger(
        run_dir,
        epoch=start_epoch - 1,
        global_step=start_step,
        loss_history=loss_history,
        embed_every=save_emb_every,
        total_epochs=epochs,
    )
    
    n_total = len(dataset)
    if max_encounters is not None:
        max_ctx = max_encounters - 1
    else:
        max_ctx = max(len(s["context"]) for s in dataset.samples)
    embed_dim = model_params["embed_dim"]
    
    vicreg_keys = ("sim", "var_pred", "var_enc", "cov_pred", "cov_enc")

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {epochs} epochs ({len(loader)} batches of {batch_size})")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        vicreg_accum = {k: 0.0 for k in vicreg_keys}
        n_batches = 0

        with logger.embedding_writer(epoch, n_total, max_ctx, embed_dim) as ew:
            for i, batch in tqdm(enumerate(loader), leave=False,
                                 total=len(loader), unit="batch", colour="green",
                                 desc=f"[epoch {epoch}/{epochs}]"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                z_enc, z_pred, z_target, z_target_sg = model(batch_dev)
                loss_dict = jepa_stopgrad_loss(
                    z_enc, z_pred, z_target, z_target_sg, 
                    batch_dev["ctx_pad_mask"],
                    sim_weight, var_weight, cov_weight)
                loss = loss_dict["loss"]
                loss.backward()

                # --- grad accumulation
                if (i + 1) % accum_steps == 0:
                    pre_clip = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    logger.log_grad_norm(pre_clip.item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                # --- embedding health
                logger.update_embed_health(
                    z_enc=z_enc,
                    z_pred=z_pred,
                    z_target=z_target,
                    ctx_pad_mask=batch_dev["ctx_pad_mask"],
                )

                # --- embedding disk write (no-op on non-embed epochs)
                ew.write_batch(
                    z_enc=z_enc,
                    z_pred=z_pred,
                    z_target=z_target,
                    mask_pos=batch_dev["mask_pos"],
                    ctx_pad_mask=batch_dev["ctx_pad_mask"],
                    subject_ids=batch["subject_ids"],
                )

                # --- stat logging
                for k in vicreg_keys:
                    vicreg_accum[k] += loss_dict[k].detach().item()
                logger.log_batch(loss.item(), len(batch))
                n_batches += 1

        # --- eval -----------------------------------------------------
        model.eval()

        if epoch % save_ckpt_every == 0 or epoch == epochs:
            save_checkpoint(model.state_dict(), model_params,
                            optimizer, scheduler,
                            epoch, logger.global_step, logger.loss_history,
                            ckpt_dir, seed=seed)

        vicreg_log = {k: vicreg_accum[k] / max(n_batches, 1) for k in vicreg_keys}
        logger.log_epoch(lr=optimizer.param_groups[0]["lr"], **vicreg_log)

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------
    logger.finalize()
