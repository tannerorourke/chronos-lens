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
from src.training.checkpoint import save_embedding_vecs, save_checkpoint, load_checkpoint
from src.utils.io import load_sequences, build_vocab, EXPERIMENTS_DIR



SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main(params: Dict, run_dir: Path, device: torch.device) -> None:
    # --- Params passed from config file ---
    # --- model hypers
    model_p =       params['model']
    embed_dim =     model_p['embed_dim']
    num_heads =     model_p['num_heads']
    num_layers =    model_p['num_layers']
    ffn_dim =       model_p['ffn_dim']
    max_seq_len =   model_p['max_seq_len']
    predictor_hidden = model_p['predictor_hidden']
    use_bfloat16 =  model_p.get('use_bfloat16', False)
    
    # --- optimization
    opt_params =    params['optimization']
    epochs =        opt_params['epochs']
    tau =           opt_params.get('tau', 0.996)
    
    # --- data settings
    data_p =        params['data']
    batch_size =    data_p['batch_size']
    n_patients =    data_p.get('n_patients', 0)
    pad_idx =       data_p.get('pad_idx', 0)
    
    # --- artifact settings
    artifact_p =    params['artifacts']
    checkpoint_every = artifact_p['checkpoint_every'] or epochs
    log_emb_vecs =  artifact_p.get('log_emb_vecs', True)
    log_emb_vecs_every = artifact_p['log_emb_vecs_every'] or epochs
    
    # Disabling tf32 matmul when using bfloat16 avoids stacking two levels of reduced precision
    if device.type == "cuda" and use_bfloat16:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    
    ckpt_dir = run_dir / "checkpoints"
    emb_dir = run_dir / "embeddings"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    
    # --- init sequences, vocab, dataset, loader ---
    patients = load_sequences(n=n_patients)
    vocab = build_vocab(patients, pad_idx, dir=run_dir)
    with open(run_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
        
    dataset = JEPADataset(patients, vocab)
    loader = DataLoader(
        dataset, batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False)

    
    # --- init model ---
    model = JEPA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        ffn_dim=ffn_dim,
        predictor_hidden=predictor_hidden,
        vocab_size=len(vocab), tau=tau, pad_idx=pad_idx,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {(n_params / 1e6):2f}M")
    
    
    # --- init optimizer/scheduler/scaler ---
    optimizer, scheduler, scaler = init_optimizers(
        model, opt_params, 
        ipe=len(loader), 
        num_epochs=epochs, 
        use_bfloat16=use_bfloat16)

    
    # --- (Optionally) load from checkpoint ---
    start_epoch, global_step, loss_history = 1, 1, []
    if params.get("resume_from"):
        ckpt_path = EXPERIMENTS_DIR / params["resume_from"]
        start_epoch, global_step, loss_history = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device,
            resume_optimizer=params.get("resume_optimizer", False))
        
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
            
            epoch_losses.append(loss.item())
            logger.log_step(loss.item())

        # ----------------------------------------------------------------------------
        model.eval()
        
        if checkpoint_every is not None and (epoch % checkpoint_every == 0 or epoch == epochs):
            save_checkpoint(epoch, logger.global_step,model, optimizer, scheduler, scaler, logger._loss_history, {
                "embed_dim":    embed_dim,
                "num_heads":    num_heads,
                "num_layers":   num_layers,
                "ffn_dim":      ffn_dim,
                "max_seq_len":  max_seq_len,
                "predictor_hidden": predictor_hidden,
                "vocab_size":   len(vocab),
                "tau":          tau,
                "pad_idx":      pad_idx,
            }, ckpt_dir)
            
        stat_log = {}
        if log_emb_vecs and (epoch % log_emb_vecs_every == 0 or epoch == epochs or epoch == 1):
            _, stat_log = save_embedding_vecs(model, loader, device, epoch, emb_dir)
        
        logger.log_epoch(loss=float(np.mean(epoch_losses)), **stat_log)
        grad_mon.reset()
    
    # ------------------------------------------------------------------
    # --- POST-TRAINING ------------------------------------------------
    # ------------------------------------------------------------------

    logger.finalize()