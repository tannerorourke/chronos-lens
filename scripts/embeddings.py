#!/usr/bin/env python3
"""Embeddings tooling for a run: list / fetch / extract / get.

A run's embeddings live at ``RUNS_DIR/<run-id>/embeddings/embeddings_<epoch>.npz``.
They are too big to keep locally, so the canonical read path is "check local, else
pull just that one object from S3" (a single ``aws s3 cp``, never a bulk sync). When
no ``.npz`` exists yet, they are *extracted* from a checkpoint by a forward pass.

This is a training-side tool (runs on the GPU box right after training). It imports
``src.infra.{loaders,s3}`` + ``src.utils.io`` - never ``src.analysis``.

Subcommands
-----------
  list     python -m scripts.embeddings list    --exp <run-id>
  fetch    python -m scripts.embeddings fetch   --exp <run-id> [--emb embeddings_40.npz] [--all]
  extract  python -m scripts.embeddings extract --exp <run-id> --ckpt checkpoint_100.pt
  get      python -m scripts.embeddings get     --exp <run-id> [--emb embeddings_40.npz] [--ckpt checkpoint_100.pt]
           # cascade: local -> S3 fetch -> extract from --ckpt

Env: CHRONOS_S3_BUCKET (default chronos-ml), CHRONOS_NO_FETCH=1 to fail fast.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.infra.loaders import extract_and_write_embeddings
from src.utils.io import RUNS_DIR, _embedding_epoch, _normalize_npz, _latest_local
from src.infra.s3 import (
    ensure_local, s3_list, s3_uri, get_bucket, aws_available,
    push_file, download_object)


# =============================================================================
# Helpers
# =============================================================================

def _extract(run_id: str, ckpt_name: str, output_subdir: str = "embeddings") -> None:
    """Extract embeddings from a checkpoint into RUNS_DIR/<run-id>/<output_subdir>/.

    Thin CLI wrapper over :func:`src.infra.loaders.extract_and_write_embeddings`.
    Seed is applied inside ``load_scaffolding`` (from the run's frozen meta.seed)
    before the loader is built, so extraction order is deterministic.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_and_write_embeddings(run_id, ckpt_name, device, output_subdir)


def _resolve_name(run_id: str, name: str | None) -> str:
    """Resolve an embeddings filename: normalize an explicit one, else the latest
    available locally, else the latest in S3."""
    if name:
        return _normalize_npz(name)
    latest_local = _latest_local(RUNS_DIR / run_id / "embeddings")
    if latest_local:
        return latest_local
    remote = [n for n in s3_list(run_id, "embeddings") if n.endswith(".npz")]
    if not remote:
        raise SystemExit(
            f"No embeddings found for run '{run_id}' (local or S3). Pass --emb, or "
            f"`extract --ckpt <checkpoint.pt>`.")
    remote.sort(key=lambda n: (_embedding_epoch(n), n))
    return remote[-1]


def fetch_embeddings(run_id: str, name: str | None = None, *, save: str = "none"):
    """Resolve an embeddings .npz for a run under one of three persistence modes.

    - ``none`` (default): return the loaded ``dict[str, np.ndarray]`` to the caller
      without persisting a copy under the run's ``embeddings/`` dir. If a local copy
      already exists it is read in place; otherwise the object is pulled to a temp
      file, loaded, and the temp file is discarded.
    - ``local``: ensure the object is in the canonical local cache
      (``RUNS_DIR/<run-id>/embeddings/``) and return its ``Path``.
    - ``s3``: ensure the object exists in S3 (pushing the local copy up if missing)
      and return the local ``Path``.

    The default is deliberately no-persist: embeddings can be tens of GB, so the
    in-code analysis path that just needs the arrays should not litter local disk.
    """
    name = _resolve_name(run_id, name)
    local = RUNS_DIR / run_id / "embeddings" / name

    if save == "local":
        return ensure_local(f"embeddings/{name}", run_id)

    if save == "s3":
        src = local if local.exists() else ensure_local(f"embeddings/{name}", run_id)
        push_file(src, run_id, f"embeddings/{name}", verify=True)
        return src

    if save != "none":
        raise ValueError(f"save must be one of none|local|s3, got '{save}'")

    # none: prefer an existing local copy, else transient temp download
    if local.exists():
        return dict(np.load(local, allow_pickle=True))
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / name
        if not download_object(run_id, f"embeddings/{name}", dest):
            raise FileNotFoundError(
                f"Could not fetch embeddings/{name} for run '{run_id}' from S3 "
                f"(and no local copy). Check the bucket / aws CLI, or `list`.")
        return dict(np.load(dest, allow_pickle=True))


# =============================================================================
# Subcommands
# =============================================================================

def cmd_list(args) -> None:
    run_id = args.exp
    names = [n for n in s3_list(run_id, "embeddings") if n.endswith(".npz")]
    if not names:
        print(f"No embeddings listed at {s3_uri(run_id, 'embeddings/')} "
              f"(bucket={get_bucket()}, aws={'yes' if aws_available() else 'no'}).")
        return
    print(f"Embeddings available for run '{run_id}':")
    for n in sorted(names, key=lambda n: (_embedding_epoch(n), n)):
        print(f"  {n}")


def cmd_fetch(args) -> None:
    run_id = args.exp

    # bulk pull (opt-in)
    if args.all:
        src = s3_uri(run_id, "embeddings/")
        dst = RUNS_DIR / run_id / "embeddings"
        dst.mkdir(parents=True, exist_ok=True)
        print(f"Bulk pull {src} -> {dst}")
        subprocess.run(["aws", "s3", "sync", src, str(dst), "--only-show-errors"], check=False)
        return

    # single object: resolve under the requested persistence mode (default: none)
    result = fetch_embeddings(run_id, args.emb, save=args.save)
    if args.save == "none":
        print(f"Fetched (not persisted) embeddings for '{run_id}': "
              f"keys={list(result.keys())}. Use --save local to keep a copy on disk.")
    else:
        print(f"Ready ({args.save}): {result}")


def cmd_extract(args) -> None:
    _extract(args.exp, args.ckpt, args.output_subdir)


def cmd_get(args) -> None:
    """Cascade: local -> S3 fetch -> extract from --ckpt."""
    run_id = args.exp
    emb_dir = RUNS_DIR / run_id / "embeddings"
    name = _normalize_npz(args.emb) if args.emb else None

    # 1. local
    if name is not None:
        local = emb_dir / name
        if local.exists():
            print(f"Ready (local): {local}")
            return
    else:
        latest_local = _latest_local(emb_dir)
        if latest_local is not None:
            print(f"Ready (local): {emb_dir / latest_local}")
            return

    # 2. S3 (single object cp)
    remote = [n for n in s3_list(run_id, "embeddings") if n.endswith(".npz")]
    if name is not None and name in remote:
        print(f"Ready (S3): {ensure_local(f'embeddings/{name}', run_id)}")
        return
    if name is None and remote:
        remote.sort(key=lambda n: (_embedding_epoch(n), n))
        print(f"Ready (S3): {ensure_local(f'embeddings/{remote[-1]}', run_id)}")
        return

    # 3. extract from checkpoint
    if args.ckpt:
        print("No embeddings found locally or in S3; extracting from checkpoint...")
        _extract(run_id, args.ckpt, args.output_subdir)
        return

    raise SystemExit(
        f"No embeddings found locally or in S3 for run '{run_id}', and no --ckpt "
        f"given to extract. Pass --ckpt <checkpoint.pt> to compute them.")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list embeddings available in S3 for a run")
    p_list.add_argument("--exp", "--run-id", dest="exp", required=True,
                        help="run-id directory under RUNS_DIR")
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="pull an existing .npz from S3")
    p_fetch.add_argument("--exp", "--run-id", dest="exp", required=True,
                         help="run-id directory under RUNS_DIR")
    p_fetch.add_argument("--emb", type=str, default=None,
                         help="embeddings_<N>.npz to fetch (default: latest available)")
    p_fetch.add_argument("--save", choices=["none", "local", "s3"], default="none",
                         help="persistence: none=load+return (no disk copy, default); "
                              "local=keep under embeddings/; s3=ensure object is in S3")
    p_fetch.add_argument("--all", action="store_true",
                         help="bulk-pull the entire embeddings/ folder (opt-in)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_extract = sub.add_parser("extract", help="compute embeddings from a checkpoint")
    p_extract.add_argument("--exp", "--run-id", dest="exp", required=True,
                           help="run-id directory under RUNS_DIR")
    p_extract.add_argument("--ckpt", required=True, type=str,
                           help="checkpoint .pt filename in the run's checkpoints/ dir")
    p_extract.add_argument("--output-subdir", default="embeddings",
                           help="subdir of the run dir to write into (default: embeddings)")
    p_extract.set_defaults(func=cmd_extract)

    p_get = sub.add_parser("get", help="resolve embeddings: local -> S3 -> extract")
    p_get.add_argument("--exp", "--run-id", dest="exp", required=True,
                       help="run-id directory under RUNS_DIR")
    p_get.add_argument("--emb", type=str, default=None,
                       help="embeddings_<N>.npz to resolve (default: latest available)")
    p_get.add_argument("--ckpt", type=str, default=None,
                       help="checkpoint to extract from if no .npz exists locally or in S3")
    p_get.add_argument("--output-subdir", default="embeddings",
                       help="subdir of the run dir to write into (default: embeddings)")
    p_get.set_defaults(func=cmd_get)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
