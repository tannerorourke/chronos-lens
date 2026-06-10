#!/usr/bin/env python3
"""Embeddings tooling for a run: list / fetch / extract / get.

A run's embeddings live at ``EXPS_DIR/<run-id>/embeddings/embeddings_<epoch>.npz``.
They are too big to keep locally, so the canonical read path is "check local, else
pull just that one object from S3" (a single ``aws s3 cp``, never a bulk sync). When
no ``.npz`` exists yet, they are *extracted* from a checkpoint by a forward pass.

This is a training-side tool (runs on the GPU box right after training). It imports
``src.infra.{loaders,s3}`` + ``src.utils.io`` - never ``src.analysis``.

Subcommands
-----------
  list     python -m scripts.embeddings list    --exp <run-id>
  fetch    python -m scripts.embeddings fetch   --exp <run-id> [--emb embeddings_40.npz] [--all]
  extract  python -m scripts.embeddings extract --exp <run-id> [--ckpt checkpoint_100.pt] [--save s3]
  get      python -m scripts.embeddings get     --exp <run-id> [--emb embeddings_40.npz] [--ckpt ...] [--save local|s3]
           # unified entry. cascade: local -> S3 fetch -> extract. The checkpoint
           # is auto-resolved from --exp (local then S3) when --ckpt is omitted;
           # --save controls persistence of a freshly extracted .npz.

The checkpoint resolver prefers the highest-epoch ``checkpoint_<N>.pt`` over the
rolling ``last.pt``. Extraction always writes the .npz locally (streaming writer),
so on the extract path ``--save none`` and ``--save local`` are equivalent on disk
and ``--save s3`` additionally pushes it to ``runs/<run-id>/embeddings/``.

Env: AWS_S3_BUCKET (default chronos-ml), CHRONOS_NO_FETCH=1 to fail fast.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)

from src.infra.inference import load_embeddings_for_analysis
from src.utils.io import EXPS_DIR
from src.infra.s3 import S3Client

# =============================================================================
# Subcommands
# =============================================================================

def cmd_fetch(args) -> None:
    run_id = args.exp
    run_dir = EXPS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run {run_id} not found in ../{run_dir.parts[-2]}")
    
    ow = args.overwrite
    with S3Client(run_dir, s3_subdir="runs", strict=True) as s3:
        if args.file is not None:
            if not ow and (run_dir / args.file).exists():
                logger.info(f"Skipping existing {args.file}")
                return
            s3.fetch(args.file)
        elif args.folder is not None:
            folder = Path(args.folder).as_posix().strip("/")
            s3.fetch_folder(folder, _overwrite=ow, _async=False)
        else:
            return

def cmd_sync(args) -> None:
    run_id = args.exp
    run_dir = EXPS_DIR / run_id
    if not run_dir.is_dir():
        run_dir.mkdir(parents=True)
    
    ow = args.overwrite
    with S3Client(run_dir, s3_subdir="runs", strict=True) as s3:
        if args.file is not None:
            if args.source == "local":
                if not ow and s3.exists(args.file):
                    logger.info(f"Skipping existing {args.file}")
                    return
                s3.upload(args.file)
            elif args.source == "s3":
                if not ow and (run_dir / args.file).exists():
                    logger.info(f"Skipping existing {args.file}")
                    return
                s3.fetch(args.file)
        
        elif args.folder is not None:
            folder = args.folder.strip("/")
            ldir = run_dir / f"{folder}"
            
            if args.source == "local":
                s3.upload_folder(ldir, _overwrite=ow, _validate=True)
                    
            elif args.source == "s3":
                s3.fetch_folder(folder, _overwrite=ow, _async=False)
                
                


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--exp', required=True,
                    help="Run-id naming the input config 'experiments/<exp>.yaml'")

sub = parser.add_subparsers(dest="command", required=True)
p_fetch = sub.add_parser("fetch", help="download `S3/runs/<exp>/<file>` to local `artifacts/<exp>/<file>`")
p_fetch.add_argument("--overwrite", type=bool, default=False, 
    help="if existing file should be overwritten")
p_fetch.add_argument("--file", type=str, default=None, 
    help="file path to fetch from S3")
p_fetch.add_argument("--folder", type=str, default=None, 
    help="folder path to fetch from S3")
p_fetch.set_defaults(func=cmd_fetch)

p_sync = sub.add_parser("sync", 
    help="Sync a file between local `artifacts/<exp>/<file>` and `S3/runs/<exp>/<file>`")
p_sync.add_argument("--source", type=str, choices=["local", "s3"], required=True, 
    help="which source to push from")
p_sync.add_argument("--overwrite", type=bool, default=False, 
    help="if destination should be overwritten")
p_sync.add_argument("--file", type=str, default=None, 
    help="file path to sync")
p_fetch.add_argument("--folder", type=str, default=None, 
    help="folder path to sync")
p_sync.set_defaults(func=cmd_sync)


def main() -> None:
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
