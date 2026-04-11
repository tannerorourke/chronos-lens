#!/usr/bin/env python3
"""
EMA JEPA training pipeline for longitudinal patient sequences.

Architecture
------------
  encoder        : EncounterEncoder - online context path (with grads)
  target_encoder : EMA copy of encoder (momentum -> 1.0), no backprop
  predictor      : Transformer(z_enc context tokens + mask token -> z_pred)
  loss           : smooth_l1(z_pred, z_target)  (iJEPA)
"""
from contextlib import nullcontext
from os import environ

from sklearn.inspection import partial_dependence
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
from pathlib import Path
from typing import Dict

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.jepa_ema import JEPA_EMA
from src.training.utils.datasets import MimicDataset, collate_fn, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger, DriftMonitor
from src.training.utils.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.utils.io import load_sequences, EXPERIMENTS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    
    # --- most params are sent to respective functions
    # --- optimization
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    patience    = opt_params.get("patience", 8)
    ema         = opt_params["ema"]
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

    # --- model
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == JEPA_EMA
    
    # --- init optimizer / scheduler / no scaler necessary
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

    # --- momentum schedule
    total_steps = (len(loader) * epochs) // accum_steps
    def momentum_at(step: int, m0: float, m1: float) -> float:
        return m0 + (m1 - m0) * (step / total_steps)
    
    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    
    # --- for padding saved z_enc
    n_total = len(dataset)
    if max_encounters is not None:
        max_ctx = max_encounters - 1
    else:
        max_ctx = max(len(s["context"]) for s in dataset.samples)
    embed_dim = model_params["embed_dim"]
    
    # -- conditional autocast
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bfloat16 and device.type == "cuda"
        else nullcontext())

    # --- early stopping: using predictor L2 norm mean, pred error should rise to start,
    #     then descend as model learns. only early stop once metric clearly turns over.
    peak_metric = 0.0
    peak_epoch = 0
    is_descending = False
    best_metric, best_epoch, best_state = float("inf"), 0, None
    ep_since_imprvd = 0
    
    # --- training monitors
    logger = TrainingLogger(
        run_dir,
        epoch=start_epoch - 1,
        global_step=start_step,
        loss_history=loss_history,
        embed_every=save_emb_every,
        total_epochs=epochs
    )
    drift_mon = DriftMonitor()
    drift_mon.set_probe(next(iter(loader)))
    
    
    print(f"Training for {epochs} epochs ({len(loader)} batches of {batch_size})")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        n_batches = 0

        with logger.embedding_writer(epoch, n_total, max_ctx, embed_dim) as ew:
            for i, batch in tqdm(enumerate(loader), leave=False,
                                 total=len(loader), colour="green", unit="batch",
                                 desc=f"[epoch {epoch}/{epochs}]"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with autocast_ctx:
                    z_enc, z_pred, z_target = model(batch_dev)
                    loss = nn.functional.smooth_l1_loss(z_pred, z_target)
                loss.backward()

                # --- grad accumulation
                if (i + 1) % accum_steps == 0:
                    pre_clip = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    logger.log_grad_norm(pre_clip.item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                    # --- EMA update of target encoder
                    with torch.no_grad():
                        m = momentum_at(logger.global_step // accum_steps, ema[0], ema[1])
                        for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

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

                # --- step-level collapse early warning
                if logger.global_step % 20 == 0:
                    logger.log_step_scalar("step/z_target_std_min", 
                                           z_target.std(0).min().item())

                # --- stat logging
                logger.log_batch(loss.item(), len(batch))
                n_batches += 1

        # --- eval -----------------------------------------------------
        model.eval()
        drift_metrics = drift_mon.compute(model, device)
        logger.log_epoch(lr=optimizer.param_groups[0]["lr"], **drift_metrics)
        
        # --- save events
        cur_metric = drift_metrics["pred_err_l2_mean"]
        
        # Track climb
        if cur_metric > peak_metric:
            peak_metric = cur_metric
            peak_epoch = epoch
            
        # Descent roughly confirmed: 2% drop below peak
        if not is_descending and (cur_metric < peak_metric * 0.98) and (epoch > peak_epoch + 3):
            is_descending = True
            print(f"Descent confirmed at epoch {epoch} (peak was {peak_metric:.4f} @ ep{peak_epoch})")
            # Reset best tracking - we only care about the descent phase
            best_metric = cur_metric
            best_epoch = epoch
            ep_since_imprvd = 0
            
        # Only track best and patience once descending
        if is_descending:
            if cur_metric < best_metric - 0.005:
                best_metric = cur_metric
                best_epoch = epoch
                ep_since_imprvd = 0
                best_state = { k: v.detach().cpu().clone() for k, v in model.state_dict().items() }
            else:
                ep_since_imprvd += 1
                
            cycle_save = (epoch % save_ckpt_every == 0 or epoch == epochs)
            im_done = epoch > 30 and ep_since_imprvd >= patience
            if cycle_save or im_done:
                print(f"Early stopping at epoch {epoch}, best pred_err_l2_mean={best_metric:.4f} @ ep{best_epoch}")
                if best_state is not None:
                    save_checkpoint(best_state, model_params,
                                    optimizer, scheduler,
                                    best_epoch, logger.global_step, logger.loss_history,
                                    ckpt_dir, seed=seed)
                if im_done:
                    break

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------

    logger.finalize()
