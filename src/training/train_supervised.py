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

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import SupervisedTransformer, build_model

from src.training.utils.datasets import (
    MimicDataset, NoisyBucketedSampler, build_vocab
)
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger
from src.training.utils.checkpoint import sync_model_checkpoint
from src.utils.io import load_sequences, EXPS_DIR


def main(params: Dict, run_dir: Path, device: torch.device):
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
    ckpt_cycle      = meta_p.get("ckpt_cycle", epochs)
    m_tag           = meta_p.get("tag", None)
    m_desc          = meta_p.get("description", None)
    sync_s3         = meta_p.get("sync_s3", True)
    
    print(f"Starting {m_tag if m_tag else 'up'}..")
    if m_desc:
        print(f"-- {params['meta'].get('description', '')}")

    # --- build sequences, vocab, dataset, loader
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx=0, dir=run_dir, save=True) # freeze vocab in run dir

    ds = MimicDataset(
        patients,
        vocab,
        data_params,
        max_enc=max_encounters, 
        is_supervised=True,
        label_key=label_key)
    sampler = NoisyBucketedSampler(
        lengths=ds.sample_lengths,
        batch_size=batch_size,
        shuffle=True, 
        drop_last=False,
        noise=2)
    loader = DataLoader(ds,
        collate_fn=ds.mimic_collate,
        batch_sampler=sampler, 
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    # for post training eval
    emb_shape = ((len(ds), ds.max_enc, params["model"]["embed_dim"]))
    # keep the sampler for re-seeding per epoch
    del patients, ds; gc.collect()

    # --- model
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == SupervisedTransformer

    # --- init optimizer / scheduler
    optimizer, scheduler = init_optimizers(
        model, opt_params,
        ipe=len(loader) // accum_steps,
        num_epochs=epochs)

    start_epoch, start_step, loss_history = 1, 1, []
    # --- load checkpoint
    if params.get("resume_from"):
        ckpt_path = EXPS_DIR / params["resume_from"]
        model, model_params, optimizer, scheduler, start_epoch, start_step, loss_history = \
            sync_model_checkpoint(
                model, optimizer, scheduler,
                ckpt_path, device, restore_rng=True
            )

    print("-- Params:",
          f"Total: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M",
          f"Trainable: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.2f}M")
    
    # --- training monitors
    ckpt = None
    tm = TrainingLogger(
        run_dir, arch=model_params["architecture"],
        global_step=start_step,
        epoch=start_epoch,
        loss_history=loss_history,
        ckpt_cycle=ckpt_cycle,
        total_epochs=epochs,
        log_tb=meta_p.get("log_tb", False),
        log_csv=meta_p.get("log_csv", False))

    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    
    bce_loss = nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        sampler.set_epoch(epoch) # re-seed noisy buckets per epoch
        
        history = []
        for i, batch in tqdm(enumerate(loader), leave=False, total=len(loader),
                             unit="b", desc=f"[epoch {epoch}/{epochs}]"):
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_enc, logits = model(batch_dev)
                loss = bce_loss(logits, batch_dev["labels"].float())
            loss.backward()
            
            # --- grad accumulation
            if (i + 1) % accum_steps == 0:
                gn = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip))
                tm.ep_grad_norms.append(gn)
                tm.batch_gn = gn
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # --- batch logging
            bs = batch_dev["labels"].shape[0]
            tm.record_batch(loss.item(), bs)
            tm.record_embeds(z_enc, batch_dev["ctx_pad_mask"])
            history.append((loss.item(), bs))

        # --- eval ----------------------------------------------------------
        model.eval()
        t_loss = sum([l[0] for l in history])
        metrics = {
            "total_loss": t_loss,
            "loss_p_batch": t_loss / len(history),
            "loss_p_sample": t_loss / sum([l[1] for l in history])
        }
        tm.record_epoch(
          lr=optimizer.param_groups[0]["lr"],
          **metrics
        )
        ckpt = tm.save_checkpoint(
            { k: v.detach().cpu().clone() for k, v in model.state_dict().items() },
            model_params, optimizer, scheduler
        )
        tm.lap()

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------
    if sync_s3:
        tm.save_embeds(model, loader, device, emb_shape, write_local=True)
    tm.finalize()
