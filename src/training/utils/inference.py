"""Model reconstruction + post-hoc embedding extraction (training-side infra).

Both helpers operate on an existing run's *frozen artifacts* (config, vocab,
checkpoint) and run **inference only** - no gradients, no training. They live on
the training side so the embedding-extraction bridge (``scripts/embeddings.py``)
and the analysis scripts can share them without training importing ``src.analysis``.
"""
from pathlib import Path
from contextlib import nullcontext

import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.models.jepa_ema import JEPA_EMA
from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.supervised_transformer import SupervisedTransformer
from src.training.utils.datasets import MimicDataset
from src.training.utils.checkpoint import load_model_checkpoint
from src.utils.io import RUNS_DIR, EXPERIMENTS_DIR, load_json, load_sequences
from src.utils.s3 import ensure_local
from src.utils.seed import set_global_seed
from src.utils.tensors import set_cuda_precision, EmbeddingWriter, EmbeddingWriterSupv


def load_scaffolding(
    ckpt_name: str,
    exp_name: str,
    device: torch.device,
) -> tuple[JEPA_EMA | JEPAStopGrad | SupervisedTransformer, DataLoader, Path, tuple[dict, dict], tuple]:
    """Rebuild model + loader for an existing run from its self-contained run dir.

    ``exp_name`` is a *run-id* under :data:`RUNS_DIR`; config + vocab are read from
    that run's frozen artifacts, and the (heavy) checkpoint is pulled from S3 on
    demand via :func:`ensure_local` if it isn't already local. Returns the run dir
    as the third element.
    """
    # --- resolve the run dir + frozen config (flat input config as fallback)
    run_dir = RUNS_DIR / exp_name
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        legacy = EXPERIMENTS_DIR / f"{exp_name}.yaml"
        if legacy.exists():
            cfg_path = legacy
        else:
            raise FileNotFoundError(
                f"No config.yaml for run '{exp_name}' (looked in {run_dir} and {legacy})")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    run_id = run_dir.name

    # --- data
    data_params     = config["data"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    max_encounters  = data_params.get("max_encounters", None)
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"
    num_workers     = data_params.get("num_workers", 0)

    # --- meta
    meta_p          = config["meta"]
    seed            = meta_p["seed"]
    use_bfloat16    = meta_p["use_bfloat16"]
    is_supervised   = config["model"]["architecture"] == "supervised"
    label_key       = (meta_p.get("label_key", "label_escalation_per_enc")
                       if is_supervised else None)

    set_global_seed(seed)
    if device.type == "cuda": set_cuda_precision(use_bfloat16)

    # --- checkpoint (local first, else fetch the single heavy object from S3)
    local_ckpt = run_dir / "checkpoints" / ckpt_name
    ckpt_path = local_ckpt if local_ckpt.exists() else ensure_local(
        f"checkpoints/{ckpt_name}", run_id)
    model, ckpt = load_model_checkpoint(ckpt_path, device, restore_rng=True)
    model.eval()

    patients = load_sequences(n=n_patients)
    vocab = load_json(run_dir / "vocab.json")
    if vocab is None:  # vocab may also live only in S3
        try:
            vocab = load_json(ensure_local("vocab.json", run_id))
        except FileNotFoundError:
            pass
    assert vocab is not None, f"vocab.json not found for run {exp_name}"

    ds = MimicDataset(patients, vocab, data_params, pad_idx=0,
                      max_enc=max_encounters, is_supervised=is_supervised,
                      label_key=label_key)
    loader = DataLoader(
        ds, batch_size,
        shuffle=True, collate_fn=ds.mimic_collate, drop_last=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=pin_memory)

    return model, loader, run_dir, (ckpt, config), (ds, is_supervised, label_key, vocab)


def extract_embeddings(
    model: JEPA_EMA | JEPAStopGrad | SupervisedTransformer,
    loader: DataLoader,
    epoch: int,
    n_total: int,
    max_ctx: int,
    embed_dim: int,
    use_bf16: bool,
    is_supervised: bool,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Run one inference epoch and stream embeddings to ``output_dir`` in batches.

    The z_enc vector alone can be tens of GB, so this writes out incrementally via
    :class:`EmbeddingWriter` / :class:`EmbeddingWriterSupv` (S3-syncable on demand).
    """
    cond_autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext())

    cond_writer = (
        EmbeddingWriter(output_dir, n_total, max_ctx, embed_dim, epoch)
        if not is_supervised else
        EmbeddingWriterSupv(output_dir, n_total, max_ctx, embed_dim, epoch)
    )

    with cond_writer as ew:
        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with cond_autocast_ctx:
                    if is_supervised:
                        assert type(model) is SupervisedTransformer, "expected Supervised Transformer model"
                        # Per-encounter z_enc (B, C, D) - matches the JEPA z_enc
                        # the EmbeddingWriterSupv expects; the model's own forward
                        # pools for its classifier, which we bypass here.
                        data = (model.encode(batch_dev, pool=False), None)
                    else:
                        assert type(model) is not SupervisedTransformer, "expected Unsupervised JEPA model"
                        # (z_enc, z_pred, z_target)
                        data = model(batch_dev)

                ew.write_batch(
                    data,
                    mask_pos=batch_dev["mask_pos"],
                    ctx_pad_mask=batch_dev["ctx_pad_mask"],
                    subject_ids=batch["subject_ids"],
                )
