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

import gc
from pathlib import Path
from typing import Dict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.supervised_transformer import SupervisedTransformer
from src.training.utils.datasets import MimicDataset, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger
from src.training.utils.checkpoint import build_model, save_checkpoint, sync_model_checkpoint
from src.utils.io import load_sequences, RUNS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    # --- optimization
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    accum_steps = opt_params.get("accumulation_steps", 1)
    grad_clip   = opt_params.get("grad_clip", 5.0)

    # --- data
    data_params     = params["data"]
    label_key       = data_params["label_key"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    max_encounters  = data_params.get("max_encounters", None)
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"
    num_workers     = data_params.get("num_workers", 0)

    # --- meta
    meta_p          = params["meta"]
    seed            = meta_p["seed"]
    use_bfloat16    = meta_p["use_bfloat16"]
    save_cycle      = meta_p.get("save_cycle", epochs)
    m_tag           = meta_p.get("tag", None)
    m_desc          = meta_p.get("description", None)
    ckpt_dir = run_dir / "checkpoints"
    
    print(f"Starting {m_tag if m_tag else 'up'}..")
    if m_desc:
        print(f"  -- {params['meta'].get('description', '')}")

    # --- build sequences, vocab, dataset, loader
    patients = load_sequences(n=n_patients)
    print(f"Patients: {len(patients)}")
    vocab = build_vocab(patients, pad_idx=0, dir=run_dir, save=True)  # freeze vocab in run dir
    print(f"Vocab: {len(vocab)} tokens")

    ds = MimicDataset(patients, vocab, data_params, pad_idx=0, max_enc=max_encounters,
                      is_supervised=True, label_key=label_key)
    del patients; gc.collect()
    
    loader = DataLoader(ds, batch_size, collate_fn=ds.mimic_collate,
        shuffle=True, drop_last=False, pin_memory=pin_memory,
        num_workers=num_workers, persistent_workers=num_workers > 0)

    # --- model
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == SupervisedTransformer

    # --- init optimizer / scheduler
    # ipe is optimizer-steps-per-epoch (one scheduler.step() per accumulation
    # group) so the warmup/cosine schedule is sized in the same units it's
    # stepped in - otherwise accum_steps>1 stretches warmup and the cosine
    # never reaches min_lr.
    optimizer, scheduler = init_optimizers(
        model, opt_params,
        ipe=len(loader) // accum_steps,
        num_epochs=epochs)

    start_epoch, global_step, loss_history = 1, 1, []
    # --- load checkpoint?
    if params.get("resume_from"):
        ckpt_path = RUNS_DIR / params["resume_from"]
        model, model_params, optimizer, scheduler, start_epoch, global_step, loss_history = \
            sync_model_checkpoint(
                model,
                optimizer, scheduler,
                ckpt_path, device, restore_rng=True)

    print("Params:",
          f"Total: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M",
          f"Trainable: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.2f}M")

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    
    criterion = nn.BCEWithLogitsLoss()
    
    cond_autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bfloat16 and device.type == "cuda"
        else nullcontext())
    
    b_state, b_loss, b_epoch, since_imprv = None, float("inf"), 0, 0
    
    logger = TrainingLogger(
        run_dir, arch=model_params["architecture"],
        epoch=start_epoch - 1,
        global_step=global_step,
        loss_history=loss_history,
        total_epochs=epochs,
        log_tb=meta_p.get("log_tb", False),
        log_csv=meta_p.get("log_csv", False),
        log_wandb=meta_p.get("log_wandb", False),
        sync_s3=meta_p.get("sync_s3", False),
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        logger.lap()
        history = []

        for i, batch in tqdm(enumerate(loader), leave=False, total=len(loader),
                             unit="b", desc=f"[epoch {epoch}/{epochs}]"):
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with cond_autocast_ctx:
                z_enc, logits = model(batch_dev)
                loss = criterion(logits, batch_dev["labels"].float())
            loss.backward()
            
            # --- grad accumulation
            if (i + 1) % accum_steps == 0:
                pre_clip = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                logger.log_grad_norm(pre_clip.item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # --- stat logging
            bs = batch_dev["tgt_times"].shape[0]
            history.append((loss.item(), bs))
            logger.log_batch(loss.item(), bs)
            logger.update_embed_health(z_enc, batch_dev["ctx_pad_mask"])

        # --------------------------------------------------------------
        model.eval()
        t_loss = sum([l[0] for l in history])
        metrics = {
            "total_loss": t_loss,
            "loss_p_batch": t_loss / len(history),
            "loss_p_sample": t_loss / sum([l[1] for l in history])
        }
        logger.log_epoch(lr=optimizer.param_groups[0]["lr"], model=model, **metrics)
        
        # --- cherry picking best checkpoints based on logging metrics
        if t_loss < b_loss:
            b_loss, b_epoch = t_loss, epoch
            b_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_imprv = 0
        else:
            since_imprv += 1
        
        if epoch % save_cycle == 0 or epoch == epochs:
            if b_state is not None:
                save_checkpoint(b_state, model_params,
                                optimizer, scheduler,
                                b_epoch, logger.global_step, logger.loss_history,
                                ckpt_dir)
                logger.sync()  # non-blocking S3 sync of the run dir
                b_state, b_epoch = None, 0
            else:
                print(f"  Checked for recent best - none to save.")

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------
    logger.finalize()
    return
