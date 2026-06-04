"""S3 archive helpers for run artifacts (the ``chronos-ml`` bucket).

Thin wrappers over the **AWS CLI** (``aws s3 ...``) - deliberately NO ``boto3``
dependency, matching the Lambda Cloud workflow. Everything degrades gracefully:

* If the AWS CLI is missing or no bucket is configured, sync is a quiet no-op
  (warn once) and ``ensure_local`` raises a clear, actionable error instead of
  hanging - telemetry/archival must never crash a multi-hour training run.

Bucket layout mirrors the local run tree::

    s3://<bucket>/runs/<run-id>/<rel-path>

so a run directory under ``RUNS_DIR/<run-id>/`` is a single clean sync source /
fetch target. ``RUNS_DIR`` is imported lazily (inside functions) to avoid a
circular import with :mod:`src.utils.io`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv() 
except:
    pass


# =============================================================================
# Configuration / availability
# =============================================================================

def get_bucket() -> str | None:
    """Configured bucket name (env ``CHRONOS_S3_BUCKET``), or None when unset."""
    return os.environ.get("CHRONOS_S3_BUCKET") or None


def fetch_disabled() -> bool:
    """True when ``CHRONOS_NO_FETCH=1`` (offline / air-gapped: fail fast)."""
    return os.environ.get("CHRONOS_NO_FETCH") == "1"


def aws_available() -> bool:
    return shutil.which("aws") is not None


def s3_uri(run_id: str, rel_path: str = "", bucket: str | None = None) -> str:
    bucket = bucket or get_bucket()
    base = f"s3://{bucket}/runs/{run_id}"
    return f"{base}/{rel_path}" if rel_path else base


# =============================================================================
# Non-blocking sync (training -> S3)
# =============================================================================

class S3Syncer:
    """Fire-and-forget ``aws s3 sync`` of a run directory to S3.

    Each :meth:`sync` spawns a detached ``aws s3 sync`` subprocess and returns
    immediately so the training loop never stalls on network I/O. A final
    *blocking* sync in :meth:`close` guarantees nothing is lost on teardown.

    Disabled (no-op) unless ``enabled`` is True AND the AWS CLI + a bucket are
    both available; warns exactly once when it can't run.
    """

    def __init__(self, run_dir, run_id: str | None = None,
                 enabled: bool = False, bucket: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.bucket = bucket or get_bucket()
        self.enabled = bool(enabled)
        self._procs: list[subprocess.Popen] = []
        self._warned = False

        if self.enabled and not aws_available():
            self._warn("aws CLI not found on PATH")
        if self.enabled and not self.bucket:
            self._warn("no S3 bucket configured (set S3_BUCKET)")

    def _warn(self, msg: str) -> None:
        if not self._warned:
            print(f"[S3Syncer] disabled: {msg}; continuing on local artifacts only")
            self._warned = True
        self.enabled = False

    def sync(self, *, blocking: bool = False, subpath: str = "") -> None:
        """Sync ``run_dir/<subpath>`` -> ``s3://<bucket>/runs/<run-id>/<subpath>``."""
        if not self.enabled:
            return
        src = self.run_dir / subpath if subpath else self.run_dir
        dst = s3_uri(self.run_id, subpath, self.bucket)
        cmd = ["aws", "s3", "sync", str(src), dst, "--only-show-errors"]
        try:
            if blocking:
                subprocess.run(cmd, check=False)
            else:
                # Reap any finished background syncs, then spawn a new detached one.
                self._procs = [p for p in self._procs if p.poll() is None]
                self._procs.append(subprocess.Popen(cmd))
        except Exception as e:  # never crash a run over telemetry
            self._warn(f"sync failed ({e})")

    def push(self, rel_path: str, *, verify: bool = True) -> bool:
        """Blocking, **verified** single-object upload of ``run_dir/<rel_path>``.

        Unlike :meth:`sync` (fire-and-forget bulk dir sync), this guarantees the
        one irreplaceable object (a checkpoint or the final embeddings npz) lands
        in S3 before returning, and confirms it. No-op (returns False) unless the
        syncer is enabled.
        """
        if not self.enabled:
            return False
        return push_file(self.run_dir / rel_path, self.run_id, rel_path,
                         verify=verify, bucket=self.bucket)

    def close(self) -> None:
        """Wait for outstanding async syncs, then do one final blocking full sync."""
        for p in self._procs:
            try:
                p.wait(timeout=300)
            except Exception:
                pass
        self.sync(blocking=True)


# =============================================================================
# Blocking, verified single-object push (training -> S3)
# =============================================================================

def _verify_object(uri: str, expected_size: int) -> bool:
    """True iff the S3 object at ``uri`` exists with ``ContentLength`` == expected."""
    no_scheme = uri[len("s3://"):]
    bucket, _, key = no_scheme.partition("/")
    out = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return False
    try:
        return json.loads(out.stdout).get("ContentLength") == expected_size
    except Exception:
        return False


def push_file(local_path, run_id: str, rel_path: str, *,
              verify: bool = True, retries: int = 1, bucket: str | None = None) -> bool:
    """Blocking upload of one file to ``s3://<bucket>/runs/<run-id>/<rel_path>``.

    Checkpoints and the final embeddings npz are the only artifacts a run cannot
    regenerate, so they get a verified push (size-checked via ``head-object``)
    rather than the fire-and-forget bulk sync. Retries once on failure. Degrades
    to a warn-and-continue no-op when the CLI or bucket is unavailable - archival
    must never crash a finished run. Returns True on confirmed upload.
    """
    local_path = Path(local_path)
    bucket = bucket or get_bucket()
    if not bucket or not aws_available():
        print(f"[push_file] skipped: aws/bucket unavailable; "
              f"{local_path.name} stays local-only")
        return False
    if not local_path.exists():
        print(f"[push_file] WARNING: {local_path} does not exist; nothing to upload")
        return False

    uri = s3_uri(run_id, rel_path, bucket)
    size = local_path.stat().st_size
    for attempt in range(retries + 1):
        cp = subprocess.run(
            ["aws", "s3", "cp", str(local_path), uri, "--only-show-errors"], check=False)
        if cp.returncode == 0 and (not verify or _verify_object(uri, size)):
            print(f"[push_file] uploaded {local_path.name} -> {uri}")
            return True
        print(f"[push_file] upload/verify failed for {uri} "
              f"(attempt {attempt + 1}/{retries + 1})")
    print(f"[push_file] ERROR: could not upload {local_path.name} -> {uri} "
          f"after {retries + 1} attempts; file remains local at {local_path}")
    return False


def download_object(run_id: str, rel_path: str, dest) -> bool:
    """Blocking ``aws s3 cp`` of one object to an explicit ``dest`` path (not the
    canonical local cache). Used for transient (no-persist) fetches. Returns True
    on success; quiet False when CLI/bucket/fetch are unavailable.
    """
    bucket = get_bucket()
    if not bucket or not aws_available() or fetch_disabled():
        return False
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    uri = s3_uri(run_id, rel_path, bucket)
    cp = subprocess.run(
        ["aws", "s3", "cp", uri, str(dest), "--only-show-errors"], check=False)
    return cp.returncode == 0 and dest.exists()


# =============================================================================
# On-demand fetch (S3 -> local).  Embeddings / SAE ckpts are too big to keep.
# =============================================================================

def ensure_local(rel_path: str, run_id: str, *, allow_fetch: bool = True) -> Path:
    """Return the local path for ``runs/<run-id>/<rel_path>``, pulling the single
    object from S3 if it isn't already on disk.

    Logic: build ``RUNS_DIR/<run-id>/<rel_path>``; if present, return it. Else, if
    fetching is allowed and a bucket is configured, ``aws s3 cp`` exactly that one
    object (NOT a bulk sync of the run) and return it. Otherwise raise a clear
    error naming both the missing local path and the S3 URI it tried.

    Fetching is disabled (fail fast) when ``CHRONOS_NO_FETCH=1`` or
    ``allow_fetch=False``.
    """
    from src.utils.io import RUNS_DIR  # lazy import to avoid circular dependency

    local = RUNS_DIR / run_id / rel_path
    if local.exists():
        return local

    if not allow_fetch or fetch_disabled():
        raise FileNotFoundError(
            f"[ensure_local] '{local}' not found locally and fetching is disabled "
            f"(CHRONOS_NO_FETCH=1 or allow_fetch=False).")

    bucket = get_bucket()
    if not bucket:
        raise FileNotFoundError(
            f"[ensure_local] '{local}' not found locally and no S3 bucket is "
            f"configured (set CHRONOS_S3_BUCKET).")
    if not aws_available():
        raise FileNotFoundError(
            f"[ensure_local] '{local}' not found locally and the aws CLI is "
            f"unavailable to fetch '{s3_uri(run_id, rel_path, bucket)}'.")

    uri = s3_uri(run_id, rel_path, bucket)
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ensure_local] fetching {uri} -> {local}")
    result = subprocess.run(
        ["aws", "s3", "cp", uri, str(local), "--only-show-errors"], check=False)
    if result.returncode != 0 or not local.exists():
        raise FileNotFoundError(
            f"[ensure_local] failed to fetch '{uri}' (aws exit "
            f"{result.returncode}). Intended local target: {local}")
    return local


def s3_list(run_id: str, subdir: str, bucket: str | None = None) -> list[str]:
    """List object basenames under ``s3://<bucket>/runs/<run-id>/<subdir>/``.

    Returns an empty list (never raises) when the CLI/bucket are unavailable, so
    callers can fall back to local-only behaviour.
    """
    bucket = bucket or get_bucket()
    if not bucket or not aws_available() or fetch_disabled():
        return []
    uri = s3_uri(run_id, subdir.rstrip("/") + "/", bucket)
    try:
        out = subprocess.run(["aws", "s3", "ls", uri],
                             capture_output=True, text=True, check=False)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    names: list[str] = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts and not line.rstrip().endswith("/"):  # skip PRE (sub-prefix) rows
            names.append(parts[-1])
    return names
