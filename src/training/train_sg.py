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
from contextlib import nullcontext

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.jepa_stopgrad import JEPAStopGrad
from src.training.utils.vicreg import VicRegLoss

from src.training.utils.datasets import MimicDataset, NoisyBucketedSampler, build_vocab
from src.training.utils.optimizers import init_optimizers
from src.training.utils.logging import TrainingLogger, DriftMonitor
from src.training.utils.checkpoint import (
    build_model,
    save_checkpoint, sync_model_checkpoint,
    count_improvement)
from src.utils.io import load_sequences, RUNS_DIR


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    # --- optimization
    opt_params  = params["optimization"]
    epochs      = opt_params["epochs"]
    accum_steps = opt_params.get("accumulation_steps", 1)
    grad_clip   = opt_params.get("grad_clip", 5.0)
    assert params.get("vicreg"), "VICReg loss params not specified"

    # --- data
    data_params     = params["data"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    max_encounters  = data_params.get("max_encounters", None)
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"
    num_workers     = data_params.get("num_workers", 0)

    # --- meta
    meta_p          = params["meta"]
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

    ds = MimicDataset(
        patients, 
        vocab, 
        data_params,  
        max_enc=max_encounters)
    sampler = NoisyBucketedSampler(
        lengths=ds.sample_lengths,
        batch_size=data_params["batch_size"],
        shuffle=True, 
        drop_last=False,
        noise=2)
    loader = DataLoader(ds, 
        collate_fn=ds.mimic_collate,
        batch_sampler=sampler, 
        pin_memory=pin_memory,
        num_workers=num_workers, 
        persistent_workers=num_workers > 0)
    # keep the sampler for re-seeding per epoch
    del patients, ds; gc.collect()

    # --- model
    model_params = params["model"]
    model_params["vocab_size"] = len(vocab)
    model = build_model(model_params, device)
    assert type(model) == JEPAStopGrad
    
    # --- init optimizer / scheduler
    optimizer, scheduler = init_optimizers(
        model, opt_params,
        ipe=len(loader) // accum_steps,
        num_epochs=epochs)

    start_epoch, start_step, loss_history = 1, 1, []
    # --- load checkpoint
    if params.get("resume_from"):
        ckpt_path = RUNS_DIR / params["resume_from"]
        model, model_params, optimizer, scheduler, start_epoch, start_step, loss_history = \
            sync_model_checkpoint(
                model,
                optimizer, scheduler, 
                ckpt_path, device, restore_rng=True)

    print("Params:",
          f"Total: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M",
          f"Trainable: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.2f}M")

    # --- training monitors
    logger = TrainingLogger(
        run_dir, arch=model_params["architecture"],
        epoch=start_epoch - 1,
        global_step=start_step,
        loss_history=loss_history,
        total_epochs=epochs,
        log_tb=meta_p.get("log_tb", False),
        log_csv=meta_p.get("log_csv", False),
        log_wandb=meta_p.get("log_wandb", False),
        sync_s3=meta_p.get("sync_s3", False))
    drift_mon = DriftMonitor()
    probe_batch = next(iter(loader))
    drift_mon.set_probe(probe_batch)
    
    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    
    # --- JEPA training is a game of "cat and mouse"
    #     "good" representations ~= encoder stalling and the pred_err stalling after peaking
    pred_err_peak, pred_err_peak_epoch = 0.0, 0
    is_descending = False
    zem_high, since_zem_imprv = float("-inf"), 0
    pem_low, since_pem_imprv = float("inf"), 0
    b_epoch, b_state = 0, None

    vic_reg_loss = VicRegLoss(**params["vicreg"])
    
    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch) # re-seed noisy buckets per epoch
        model.train()
        logger.lap()
        n_batches = 0

        for i, batch in tqdm(enumerate(loader), leave=False, total=len(loader),
                             unit="b", desc=f"[epoch {epoch}/{epochs}]"):
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_enc, z_pred, z_target, z_target_sg = model(batch_dev)
                loss, _ = vic_reg_loss(z_enc, z_pred, z_target, batch_dev["ctx_pad_mask"], 
                                       z_target_sg, projector=model.projector)
            loss.backward()

            # --- grad accumulation
            if (i + 1) % accum_steps == 0:
                pre_clip = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                logger.log_grad_norm(pre_clip.item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # --- stat logging
            logger.log_batch(loss.item(), batch_dev["tgt_times"].shape[0])
            logger.update_embed_health(z_enc, batch_dev["ctx_pad_mask"], z_pred, z_target)
            n_batches += 1

        # --------------------------------------------------------------
        model.eval()
        drift_log = drift_mon.compute(model, device)
        vr_log = vic_reg_loss.compute_accum(n_batches)
        ep_metrics = logger.log_epoch(lr=optimizer.param_groups[0]["lr"], model=model, **drift_log, **vr_log)

        # --- cherry picking best checkpoints based on logging metrics
        pem = ep_metrics["pred_err_l2_mean"]
        zem = ep_metrics["embed_z_enc_std_mean"]
        zes_min = ep_metrics["embed_z_enc_std_min"]
        dop = ep_metrics["drift_over_pred"]
        if zes_min < 0.25:
            print(f"  WARNING: z_enc_std_min={zes_min:.4f} - possible dim collapse")
        if dop < 0.1 and zes_min < 0.4:
            print(f"  WARNING: drift_over_pred={dop:.4f} - possible partial collapse")
        
        # Only look for save points if prediction error is falling
        if pem > pred_err_peak:
            pred_err_peak, pred_err_peak_epoch = pem, epoch
        
        pred_down = (pem < pred_err_peak * 0.97)
        pred_lag = (epoch > pred_err_peak_epoch + 3)
        if not is_descending and pred_down and pred_lag:
            is_descending = True
            pem_low, b_epoch = pem, epoch
            since_pem_imprv = 0
            print(f"Predictor descending at epoch {epoch}.")
        
        # Track best or cycle save once prediction error peaks
        if is_descending:
            zem_high, since_zem_imprv, _ = \
                count_improvement(zem, zem_high, since_zem_imprv, delta=0.003)
            pem_low, since_pem_imprv, pem_imprvd = \
                count_improvement(pem, pem_low, since_pem_imprv, delta=0.005)
            if pem_imprvd:
                b_epoch = epoch
                b_state = { k:v.detach().cpu().clone() for k,v in model.state_dict().items() }
            
            # -- check for recent best, save, reset best
            if (epoch % save_cycle == 0 or epoch == epochs):
                if b_state is not None:
                    save_checkpoint(
                        b_state, model_params,
                        optimizer, scheduler,
                        b_epoch, logger.global_step,
                        logger.loss_history,
                        ckpt_dir)
                    logger.sync()  # non-blocking S3 sync of the run dir
                    b_state, b_epoch = None, 0
                else:
                    print(f"  Checked for recent best - none to save.")
        

    # ------------------------------------------------------------------
    # --- DONE ---------------------------------------------------------
    logger.finalize()
