#!/usr/bin/env python3
"""
Representation-health gate for a finished run.

Resolves a run's embeddings (local .npz, else S3, else re-extraction from the
stem-matched checkpoint), runs the latent-geometry health panel in
`src.analysis.health`, and prints a GO / GO (marginal) / NO-GO verdict plus a
structured JSON. Label-free and fast: the go/no-go check on whether a run
produced usable `z_enc` geometry before the analysis lenses are run.

Panel
-----
  variance            : per-dim std (collapse floor 0.25) and scale inflation
  effective_rank      : participation ratio + Marchenko-Pastur signal count
  covariance          : off-diagonal correlation mass (redundancy)
  predictor_alignment : cos_dist + P-T radius-gap fraction (JEPA only)
  time_scale          : learned temporal-encoding scale sanity band

Usage
-----
  python -m scripts.diagnostic --exp sg-vr_256_v02 --ckpt last
  python -m scripts.diagnostic --exp sg-vr_256_v02 --ckpt last --save-emb-local
  python -m scripts.diagnostic --exp sg-vr_256_v02 --no-s3

Resolution is local-first: a local ``embeddings/<stem>.npz`` is used directly and S3 is touched
only when the local copy is absent. S3 egress is billed, so prefer staging artifacts on disk;
``--no-s3`` hard-confines resolution to local disk.
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import torch

from src.infra.inference import load_embeddings_for_analysis
from src.infra.vector_computation import compute_derived_vectors
from src.analysis import health
from src.utils.io import EXPS_DIR
from src.utils.system import set_global_seed, load_exp_seed

_DERIVED_INPUTS = ("z_encs", "mask_pos", "z_pred", "z_target")
_MARK = {health.PASS: "PASS", health.WARN: "WARN", health.FAIL: "FAIL"}


# =============================================================================
# time_scale resolution
# =============================================================================

def time_scale_from_model(model) -> float | None:
    """Read the temporal-encoding scale from a live model, if extraction ran."""
    if model is None:
        return None
    try:
        for name, p in model.named_parameters():
            if name.rsplit(".", 1)[-1] == "time_scale":
                return float(p.detach().cpu().reshape(-1)[0])
    except Exception:
        return None
    return None


def time_scale_from_metrics(run_dir: Path) -> float | None:
    """Fall back to the last epoch record's time_scale in metrics.jsonl."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    ts = None
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "epoch" and "time_scale" in rec:
                ts = rec["time_scale"]
    return ts


# =============================================================================
# Reporting
# =============================================================================

def print_report(report: dict) -> None:
    arch = report["architecture"]
    print(f"\n  Representation Health <{report['run_id']} :: {report['checkpoint']}>")
    print(f"  arch={arch}  n_samples={report['n_samples']}  D={report['n_dims']}")
    print("=" * 70)

    for name, check in report["checks"].items():
        status = check.get("status", health.PASS)
        metrics = {k: v for k, v in check.items() if k != "status"}
        fields = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"  [{_MARK[status]}] {name:<20s} {fields}")

    print("=" * 70)
    flagged = f"  ({', '.join(report['flagged'])})" if report["flagged"] else ""
    print(f"  OVERALL: {report['verdict']}{flagged}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Representation-health gate for a run")
    parser.add_argument("--exp", type=str, required=True,
                        help="run-id under the runs/ directory")
    parser.add_argument("--ckpt", type=str, default="last",
                        help="embeddings/checkpoint stem to resolve (default: last)")
    parser.add_argument("--output", type=str, default=None,
                        help="results JSON path (default: <run_dir>/results/repr_health.json)")
    parser.add_argument("--no-s3", action="store_true",
                        help="confine embedding resolution to local disk")
    parser.add_argument("--save-emb-local", action="store_true",
                        help="persist a freshly extracted .npz to embeddings/")
    args = parser.parse_args()

    run_dir = EXPS_DIR / args.exp
    set_global_seed(load_exp_seed(run_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Resolving embeddings for {args.exp} :: {args.ckpt} ...")
    embeddings, (model, config) = load_embeddings_for_analysis(
        run_id=args.exp,
        name=args.ckpt,
        device=device,
        sync_ckpts=not args.no_s3,
        write_emb_local=args.save_emb_local,
        no_s3=args.no_s3,
    )

    with embeddings:
        raw = {k: embeddings[k] for k in _DERIVED_INPUTS if k in embeddings}
        vecs = compute_derived_vectors(raw)
        time_scale = time_scale_from_model(model)
        if time_scale is None:
            time_scale = time_scale_from_metrics(run_dir)
        result = health.assess(vecs, time_scale=time_scale)

    z_enc_recency = vecs["z_enc_recency"]
    report = {
        "run_id": args.exp,
        "checkpoint": args.ckpt,
        "architecture": config["model"]["architecture"],
        "n_samples": int(z_enc_recency.shape[0]),
        "n_dims": int(z_enc_recency.shape[1]),
        "verdict": result["verdict"],
        "flagged": result["flagged"],
        "checks": result["checks"],
    }

    print_report(report)

    output_path = Path(args.output) if args.output else run_dir / "results" / "repr_health.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"  Results saved -> {output_path}")


if __name__ == "__main__":
    main()
