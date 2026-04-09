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
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
from pathlib import Path
from typing import Dict

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.jepa_ema import JEPA_EMA
from src.training.utils.datasets import MimicDataset, collate_fn, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger
from src.training.utils.checkpoint import build_model, save_checkpoint, load_model_checkpoint
from src.utils.io import load_sequences, EXPERIMENTS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    
    # --- most params are sent to respective functions
    # --- optimization
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    ema         = opt_params["ema"]
    accum_steps = opt_params.get("accumulation_steps", 1)

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
    embed_every     = meta_p.get("embed_every", epochs)
    m_tag           = meta_p.get("tag", None)
    m_desc          = meta_p.get("description", None)
    
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

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- model ---
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == JEPA_EMA
    
    # --- init optimizer / scheduler / scaler
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params,
        ipe=len(loader),
        num_epochs=epochs,
        use_bfloat16=use_bfloat16)
    
    # --- momentum schedule
    ipe = len(loader)
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0]) / (ipe*epochs)
        for i in range(ipe*epochs + 1))


    start_epoch, global_step, loss_history = 1, 1, []
    # --- load checkpoint?
    if params.get("resume_from"):
        ckpt_path = EXPERIMENTS_DIR / params["resume_from"]
        model, model_params, optimizer, scheduler, scaler, start_epoch, global_step, loss_history = \
            load_model_checkpoint(
                model,
                optimizer, scheduler, scaler,
                ckpt_path, device, restore_rng=True)
        for _ in range((start_epoch - 1) * ipe):
            next(momentum_scheduler)
    

    print("Params:",
          f"Total: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M",
          f"Trainable: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.2f}M")

    # --- training utility
    logger = TrainingLogger(
        run_dir,
        epoch=start_epoch - 1,
        global_step=global_step,
        loss_history=loss_history,
        embed_every=embed_every,
        total_epochs=epochs,
    )
    
    n_total = len(dataset)
    if max_encounters is not None:
        max_ctx = max_encounters - 1
    else:
        max_ctx = max(len(s["context"]) for s in dataset.samples)
    embed_dim = model_params["embed_dim"]
    
    
    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {epochs} epochs ({ipe} batches of {batch_size})")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        n_batches = 0

        with logger.embedding_writer(epoch, n_total, max_ctx, embed_dim) as ew:
            for i, batch in tqdm(enumerate(loader), leave=False,
                                 total=len(loader), unit="batch",
                                 desc="percolating..", colour="green"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                def forward_unto_dawn():
                    z_enc, z_pred, z_target = model(batch_dev)
                    loss = F.smooth_l1_loss(z_pred, z_target)
                    return loss, z_enc, z_pred, z_target

                if use_bfloat16 and scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore
                        loss, z_enc, z_pred, z_target = forward_unto_dawn()
                    scaler.scale(loss).backward()
                else:
                    loss, z_enc, z_pred, z_target = forward_unto_dawn()
                    loss.backward()

                # --- gradient accumulation
                if (i + 1) % accum_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        logger.grad_mon.capture(model.parameters())
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        logger.grad_mon.capture(model.parameters())
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                # --- EMA update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                # --- embedding health (every epoch, cheap)
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
                logger.log_batch(loss.item(), len(batch))
                n_batches += 1

        # --- after the `with` block, the writer has closed and saved ---
        model.eval()

        if epoch % save_ckpt_every == 0 or epoch == epochs:
            save_checkpoint(model, model_params,
                            optimizer, scheduler, scaler,
                            epoch, logger.global_step, logger.loss_history,
                            ckpt_dir, seed=seed)

        logger.log_epoch(lr=optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------

    logger.finalize()
