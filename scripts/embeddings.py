#!/usr/bin/env python3
"""Artifact transfer tooling for a run: fetch / sync against S3.

A run's artifacts live at ``EXPS_DIR/<run-id>/`` locally and under
``s3://<bucket>/runs/<run-id>/`` remotely; paths inside the run dir mirror 1:1.
Embeddings (``embeddings/<stem>.npz``) move as single-object copies, never bulk
syncs. Extraction from a checkpoint is not done here: the analysis loader
(``src.infra.inference.load_embeddings_for_analysis``) resolves
local .npz -> S3 -> extract on demand, with the stem naming both files
(``embeddings/<stem>.npz`` pairs with ``checkpoints/<stem>.pt``).

Subcommands
-----------
  fetch  python -m scripts.embeddings --exp <run-id> fetch --file embeddings/<stem>.npz
         python -m scripts.embeddings --exp <run-id> fetch --folder sae_<target>
  sync   python -m scripts.embeddings --exp <run-id> sync --source local --file results/composition.json
         (one-way copy: --source local pushes to S3, --source s3 pulls to disk)

Env: AWS_S3_BUCKET (default chronos-ml), AWS_REGION.
"""
import argparse
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from src.infra.s3 import S3Client
from src.utils.io import EXPS_DIR

# =============================================================================
# Subcommands
# =============================================================================

def cmd_fetch(args) -> None:
    run_dir = EXPS_DIR / args.exp
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run {args.exp} not found in ../{run_dir.parts[-2]}")

    ow = args.overwrite
    with S3Client(run_dir, s3_subdir="runs", strict=True) as s3:
        if args.file is not None:
            if not ow and (run_dir / args.file).exists():
                logger.info(f"Skipping existing {args.file}")
                return
            s3.fetch(args.file, _overwrite=ow)
        elif args.folder is not None:
            folder = Path(args.folder).as_posix().strip("/")
            s3.fetch_folder(folder, _overwrite=ow, _async=False)


def cmd_sync(args) -> None:
    run_dir = EXPS_DIR / args.exp
    if not run_dir.is_dir():
        run_dir.mkdir(parents=True)

    ow = args.overwrite
    with S3Client(run_dir, s3_subdir="runs", strict=True) as s3:
        if args.file is not None:
            if args.source == "local":
                if not ow and s3.exists(args.file):
                    logger.info(f"Skipping existing {args.file}")
                    return
                s3.upload(args.file, _overwrite=ow)
            elif args.source == "s3":
                if not ow and (run_dir / args.file).exists():
                    logger.info(f"Skipping existing {args.file}")
                    return
                s3.fetch(args.file, _overwrite=ow)

        elif args.folder is not None:
            folder = Path(args.folder).as_posix().strip("/")
            if args.source == "local":
                s3.upload_folder(run_dir / folder, _overwrite=ow, _validate=True)
            elif args.source == "s3":
                s3.fetch_folder(folder, _overwrite=ow, _async=False)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--exp', required=True,
                    help="Run-id under artifacts/training-runs/ (and s3://<bucket>/runs/)")

sub = parser.add_subparsers(dest="command", required=True)

p_fetch = sub.add_parser("fetch",
    help="download s3://<bucket>/runs/<exp>/<path> into the local run dir")
p_fetch.add_argument("--overwrite", default=False, action="store_true",
    help="overwrite an existing local copy")
p_fetch.add_argument("--file", type=str, default=None,
    help="run-relative file path to fetch (e.g. embeddings/checkpoint_40.npz)")
p_fetch.add_argument("--folder", type=str, default=None,
    help="run-relative folder to fetch recursively (e.g. sae_pred_error)")
p_fetch.set_defaults(func=cmd_fetch)

p_sync = sub.add_parser("sync",
    help="one-way copy of a file/folder between the local run dir and S3")
p_sync.add_argument("--source", type=str, choices=["local", "s3"], required=True,
    help="side holding the authoritative copy")
p_sync.add_argument("--overwrite", default=False, action="store_true",
    help="overwrite the destination if it exists")
p_sync.add_argument("--file", type=str, default=None,
    help="run-relative file path to copy")
p_sync.add_argument("--folder", type=str, default=None,
    help="run-relative folder to copy recursively")
p_sync.set_defaults(func=cmd_sync)


def main() -> None:
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
