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

from src.models import JEPAStopGrad, build_model
from src.training.utils.vicreg import VicRegLoss

from src.training.utils.datasets import (
    MimicDataset, NoisyBucketedSampler, build_vocab
)
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger, DriftMonitor, extract_time_scale
from src.training.utils.checkpoint import sync_model_checkpoint
from src.infra.s3 import S3Client
from src.utils.io import load_sequences, EXPS_DIR


def _set_requires_grad(modules: list[nn.Module], flag: bool) -> None:
    """Toggle grad for whole submodules - used to freeze the encoder + projector
       during predictor warmup."""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


def main(params: Dict, run_dir: Path, device: torch.device):
    # --- optimization
    opt_params          = params["optimization"]
    epochs              = opt_params["epochs"]
    accum_steps         = opt_params.get("accumulation_steps", 1)
    grad_clip           = opt_params.get("grad_clip", 5.0)
    base_lr             = float(opt_params.get("base_lr", 0.0))
    pred_warmup_epochs  = opt_params.get("predictor_warmup_epochs", 0)
    pred_warmup_lr      = float(opt_params.get("predictor_warmup_lr", base_lr))
    assert params.get("vicreg"), "VICReg loss params missing"

    # --- data
    data_params     = params["data"]
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
        max_enc=max_encounters)
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
    assert type(model) == JEPAStopGrad
    
    
    # --- init optimizer / scheduler
    # -- cosine is sized to the JOINT phase only; the warmup phase runs flat at
    #    pred_warmup_lr (stepped separately below), so the encoder gets its full
    #    min_lr->base_lr ramp re-anchored to unlock.
    optimizer, scheduler = init_optimizers(
        model, opt_params,
        ipe=len(loader) // accum_steps,
        num_epochs=epochs - pred_warmup_epochs)

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
    
    drift_mon = DriftMonitor()
    probe_batch = next(iter(loader))
    drift_mon.set_probe(probe_batch)
    
    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------

    vic_reg_loss = VicRegLoss(**params["vicreg"])

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        sampler.set_epoch(epoch)

        # -- predictor warmup: for the first `pred_warmup_epochs`, freeze encoder +
        #    projector and train the predictor alone (at a static pred_warmup_lr),
        #    so it is not cold when it first pulls on the encoder. A cold predictor's
        #    gradient drags the encoder toward a constant (representation collapse);
        #    warming it first removes that pull. While frozen only the invariance
        #    (MSE) term carries a gradient - var/cov sit on the constant encoder
        #    output. The cosine scheduler does not step during this phase.
        in_warmup = epoch <= pred_warmup_epochs
        if pred_warmup_epochs:
            _set_requires_grad([model.encoder, model.projector], not in_warmup)
            if in_warmup:
                for g in optimizer.param_groups:
                    g["lr"] = pred_warmup_lr
            if epoch == 1:
                print(f"  -- predictor warmup: encoder+projector frozen for "
                      f"{pred_warmup_epochs} epochs @ lr={pred_warmup_lr:g} (predictor-only)")
            elif epoch == pred_warmup_epochs + 1:
                print("  -- warmup done: encoder+projector unlocked")
        
        n_batches = 0
        for i, batch in tqdm(enumerate(loader), leave=False, total=len(loader),
                             unit="b", desc=f"[epoch {epoch}/{epochs}]"):
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_enc, z_pred, z_target, z_target_sg = model(batch_dev)
                loss, _ = vic_reg_loss(
                    z_enc, z_pred, z_target, batch_dev["ctx_pad_mask"], 
                    z_target_sg, projector=model.projector)
            loss.backward()

            # --- grad accumulation
            if (i + 1) % accum_steps == 0:
                gn = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip))
                tm.ep_grad_norms.append(gn)
                tm.batch_gn = gn
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if not in_warmup:  # warmup runs flat at pred_warmup_lr; cosine starts at unlock
                    scheduler.step()

            # --- batch logging
            tm.record_batch(loss.item(), batch_dev["tgt_times"].shape[0])
            tm.record_embeds(z_enc, batch_dev["ctx_pad_mask"], z_pred, z_target)
            n_batches += 1

        # --- eval ----------------------------------------------------------
        model.eval()
        tm.record_epoch(
            lr=optimizer.param_groups[0]["lr"],
            ts=extract_time_scale(model),
            **drift_mon.compute(model, device),
            **vic_reg_loss.compute_epoch(n_batches)
        )
        ckpt = tm.save_checkpoint(
            { k: v.detach().cpu().clone() for k, v in model.state_dict().items() },
            model_params, optimizer, scheduler
        )
        tm.lap()

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------
    # --- extract embeddings from the final checkpoint and send to S3
    if sync_s3:
        try:
            from src.infra.inference import save_embeds_for_analysis
            stem = "last" if not ckpt else ckpt.stem
            save_embeds_for_analysis(
              model, loader, device,
              run_dir, stem,
              emb_shape=emb_shape,
            )
        except Exception as e:
            print(f"  WARNING: end-of-run embedding extraction failed: {e}")

    tm.finalize()
    return
