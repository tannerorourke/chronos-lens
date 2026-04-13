
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from src.models.jepa_ema import JEPA_EMA
from src.models.supervised_transformer import SupervisedTransformer
from src.analysis.eval_infra import load_scaffolding
from src.utils.tensors import EmbeddingWriter, EmbeddingWriterSupv


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, type=str)
parser.add_argument("--ckpt", required=True, type=str)
parser.add_argument("--output-subdir", default="embeddings", required=True)
    

def extract_embeddings(
    model: JEPA_EMA | SupervisedTransformer,
    loader: DataLoader,
    epoch: int,
    config: dict,
    is_supervised: bool,
    device: torch.device,
    output_dir: Path
):
    cond_autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config["meta"]["use_bfloat16"] and device.type == "cuda"
        else nullcontext())
    
    cond_writer = (
        EmbeddingWriter(output_dir, n_total, max_ctx, model.embed_dim, epoch)
        if not is_supervised else
        EmbeddingWriterSupv(output_dir, n_total, max_ctx, model.embed_dim, epoch)
    )
    
    # one epoch
    with cond_writer as ew:
        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                data = None
                with cond_autocast_ctx:
                    # (z_enc, z_pred, z_target) if not supervised, else (z_enc, _)
                    data = model(batch_dev) 
                    
                ew.write_batch(
                    data,
                    mask_pos=batch_dev["mask_pos"],
                    ctx_pad_mask=batch_dev["ctx_pad_mask"],
                    subject_ids=batch["subject_ids"],
                )

if __name__ == "__main__":
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loader, exp_dir, (checkpoint, config), (ds, is_supervised, label_key, _) = \
        load_scaffolding(args.ckpt, args.config, args.exp)
        
    n_total = len(ds)
    
    
    if config["data"].get("max_encounters"):
        max_ctx = config["data"].get("max_encounters") - 1
    else:
        max_ctx = max(len(s["context"]) for s in ds.samples)
    
    epoch = checkpoint["epoch"]
    embed_dim = checkpoint["model_params"]["embed_dim"]
    use_bf16 = config["meta"]["use_bfloat16"]
    output_dir = exp_dir / args.output_subdir
    
    extract_embeddings(model, loader, epoch, use_bf16, is_supervised, device, output_dir)