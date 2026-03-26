#!/usr/bin/env python3
"""
Minimal JEPA training pipeline for longitudinal patient sequences.
Architecture
------------
  token_embedding  : nn.Embedding(vocab_size, 64), mean-pooled per encounter
  context_encoder  : hand-rolled Transformer (2 layers, 2 heads, dim=64) → z_context
  target_encoder   : EMA copy of context_encoder (τ=0.996), no backprop → z_target
  predictor        : MLP(z_context ⊕ pos_emb → z_pred, hidden=128)
  loss             : MSE(z_pred, z_target)
"""
from os import environ



environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ^^ Odd non-breaking Windows issue where PyTorch's MKL/Intel OpenMP gets initialized during the save operation

from typing import Dict
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.sequential_jepa import JEPA
from src.training.dataset import JEPADataset, collate_fn
from src.training.optimizers import init_optimizers
from src.training.logging import GradientMonitor, TrainingLogger
from src.training.checkpoint import build_model, save_embedding_vecs, save_checkpoint, load_model_checkpoint
from src.utils.io import load_sequences, build_vocab, EXPERIMENTS_DIR


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    # --- optimization
    opt_params =    params['optimization']
    epochs =        opt_params['epochs']
    tau =           opt_params.get('tau', 0.996)
    
    # --- data settings
    data_p =        params['data']
    batch_size =    data_p['batch_size']
    n_patients =    data_p.get('n_patients', 0)
    pad_idx =       data_p.get('pad_idx', 0)
    max_encounters = data_p.get('max_encounters', None)
    max_tokens =    data_p.get('max_tokens', None)
    
    # --- artifact settings
    artifact_p =    params['artifacts']
    checkpoint_every = artifact_p['checkpoint_every'] or epochs
    log_emb_vecs =  artifact_p.get('log_emb_vecs', True)
    log_emb_vecs_every = artifact_p['log_emb_vecs_every'] or epochs
    
    # --- init sequences, vocab, dataset, loader ---
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx, dir=run_dir)
    with open(run_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
        
    dataset = JEPADataset(patients, vocab, pad_idx,
                          max_encounters=max_encounters, max_tokens=max_tokens)
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False)
    
    ckpt_dir = run_dir / "checkpoints"
    emb_dir = run_dir / "embeddings"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    
    # --- model ---
    start_epoch, global_step, loss_history = 1, 1, []
    _ckpt = None
    
    if params.get("resume_from"):
        ckpt_path = EXPERIMENTS_DIR / params["resume_from"]
        model, _ckpt, start_epoch, global_step, loss_history = \
            load_model_checkpoint(ckpt_path, device, restore_rng=True)
        
        # Back-fill params from checkpoint in case they differ from config
        model_params = _ckpt["model_params"]
    else:
        model_params = {
            "architecture":     params['model'].get("architecture", "stopgrad"),
            "use_bfloat16":     params['model'].get("use_bfloat16", False),
            "embed_dim":        params['model']['embed_dim'],
            "num_heads":        params['model']['num_heads'],
            "num_layers":       params['model']['num_layers'],
            "ffn_dim":          params['model']['ffn_dim'],
            "max_seq_len":      params['model']['max_seq_len'],
            "predictor_hidden": params['model']['predictor_hidden'],
            "tau":              tau,
            "vocab_size":       len(vocab),
            "pad_idx":          pad_idx,
        }
        model = build_model(model_params, device)
        assert type(model) == JEPA
    
    use_bfloat16 =  model_params.get("use_bfloat16", False)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {(n_params / 1e6):2f}M")
    
    
    # --- init optimizer/scheduler/scaler ---
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params, 
        ipe=len(loader), 
        num_epochs=epochs, 
        use_bfloat16=use_bfloat16)
    
    if _ckpt is not None and params.get("resume_optimizer", False):
        if "optimizer" in _ckpt:
            optimizer.load_state_dict(_ckpt["optimizer"])
        if scheduler is not None and _ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(_ckpt["scheduler"])
        if scaler is not None and _ckpt.get("scaler") is not None:
            scaler.load_state_dict(_ckpt["scaler"])
        
    logger = TrainingLogger(run_dir, start_epoch, global_step, loss_history)
    grad_mon = GradientMonitor(model)
    
    # ------------------------------------------------------------------
    # --- TRAINING LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    print(f"Training for {len(loader)} batches (size: {batch_size}) for {epochs} epochs")
    print(f"Description: {params['meta']['tag']}: {params['meta']['description']}")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            
            if use_bfloat16 and scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore # (pylance)
                    z_context, z_pred, z_target = model(batch_dev)
                    loss = F.mse_loss(z_pred, z_target)
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                grad_mon.capture()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                z_context, z_pred, z_target = model(batch_dev)
                loss = F.mse_loss(z_pred, z_target)
                loss.backward()
                
                grad_mon.capture()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                optimizer.step()
                
            if scheduler is not None:
                scheduler.step()

            model.update_target_encoder()

        # ----------------------------------------------------------------------------
        model.eval()
        
        if checkpoint_every is not None and (epoch % checkpoint_every == 0 or epoch == epochs):
            save_checkpoint(model, model_params,
                            optimizer, scheduler, scaler,
                            epoch, logger.global_step, logger._loss_history, 
                            ckpt_dir)
            
        stat_log = {}
        if log_emb_vecs and (epoch % log_emb_vecs_every == 0 or epoch == epochs or epoch == 1):
            _, stat_log = save_embedding_vecs(model, loader, device, epoch, emb_dir)
        
        logger.log_epoch(loss=float(np.mean(epoch_losses)), **stat_log)
        grad_mon.reset()
    
    # ------------------------------------------------------------------
    # --- POST-TRAINING ------------------------------------------------
    # ------------------------------------------------------------------

    logger.finalize()