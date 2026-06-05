# chronos-lens - repository guide

Mechanistic-interpretability study of a JEPA (stop-gradient / EMA, VICReg) trained on
longitudinal MIMIC-IV patient-encounter sequences. See `README.md` for research motivation;
this file is the **structural map** - the purpose of every folder and the rules that keep the
codebase legible. It is self-contained: a fork of this repo alone is enough to navigate it.

---

## Layered architecture

Code is split into three layers with a **strict one-way import direction**:

$
  \text{utils }\leftarrow\text{ infra }\leftarrow\text{ training, analysis}
$

- **`src/utils/`** - low-level, dependency-light helpers. No imports from other `src/` layers.
- **`src/infra/`** - shared foundation that *sets up* analysis: model/loader scaffolding, embedding extraction, label prep, metrics, S3, etc. Imports `utils` and `models`; **never** imports `analysis` or the training loops.
- **`src/training/`** - only trains. Imports `utils`, `models`, and `infra` where infra is a leaf (e.g. `infra.s3`); **never** imports `analysis`.
- **`src/analysis/`** - operates on already-loaded data. Imports `infra` and `models`; **never** imported by training or infra.

If a helper is needed by both training and analysis, it belongs in `infra` (or `utils`), never duplicated across layers. Before adding an import, confirm it respects this direction.

---

## Folder guide

### `src/`

| Folder | Purpose |
| --- | --- |
| `models/` | Network definitions only: `encoder.py` (encounter encoder), `predictor.py`, `jepa_ema.py`, `jepa_stopgrad.py`, `supervised_transformer.py`, `sae.py`. Every model `forward` returns `z_enc (B, C, D)` as its first output (see `src/utils/types.md` for the full shape contract). |
| `mimic/` | MIMIC-IV extraction and causal label computation: `mimic.py`, `labels.py` (escalation state machine, 30-day readmission), `helper.py`, `metadata.py`, `baselines.py`. Upstream of training; produces `data/processed/sequences.jsonl`. |
| `training/` | Training loops (`train_ema.py`, `train_sg.py`, `train_supervised.py`) and supporting utils (`datasets.py`, `checkpoint.py`, `optimizers.py`, `logging.py`, `vicreg.py`). |
| `infra/` | `loaders.py` - model/loader scaffolding, in-memory `extract_embeddings`, streaming `stream_embeddings` + `EmbeddingWriter`s. `labels.py` - per-sample labels, ICD-block targets, temporal split, subset masks. `metrics.py` - AUROC / AUPRC / F1 / Brier / ECE. `vector_computation.py` - recency / pred-error derivation, patient/encounter reshaping. `s3.py` - run-dir sync and on-demand fetch. |
| `analysis/` | `geometry.py` (PCA, CKA, label subspaces), `composition.py` (SAE decomposition), `sae.py` (feature enrichment), `probing.py` (linear probes, layer sweep), `trajectories.py` (velocity / curvature / drift), `clustering.py`, `plotting.py`. |
| `utils/` | `io.py` - paths, config / npz / sequences I/O, and the single source of truth for all output path constants. `seed.py`, `constants.py` (`set_cuda_precision`), `types.md` (shape contract). |

### Top-level

| Folder / file | Purpose |
| --- | --- |
| `scripts/` | Runnable entry points: `train.py`, `embeddings.py` (extract / fetch / get), `diagnostic.py`, `analyze_{trajectories,features,composition,comparison}.py`, `probe.py`, `extract_mimic.py`. Thin CLIs over `src/`. |
| `experiments/` | Flat `<run-id>.yaml` input configs only - one file per run, no subdirectories. All outputs go out-of-repo to `artifacts/training-runs/<run-id>/`. |
| `notebooks/` | Exploratory / figure notebooks. Not part of the runnable pipeline. |

---

## Conventions

### Imports

**All imports are hard imports of declared dependencies.** Never hide a missing package behind `try: import X / except ImportError: ...` and a degraded fallback. If a feature needs a package, add it to `pyproject.toml` `[project].dependencies`, install it, and import it normally at module top.

Intentional exceptions to keep:

- **External-CLI availability checks**, e.g. `aws_available()` in `src/infra/s3.py`. S3 sync is a deliberate AWS-CLI workflow; absence of the CLI degrades to local-only, by design.
- **Numerical guards** in analysis, e.g. `try/except numpy.linalg.LinAlgError` around SVD / degenerate fits in `src/analysis/{geometry,probing,composition,clustering}.py`.

To add a new dependency: add it to `chronos-lens/pyproject.toml` and `chronos-lens/requirements.txt`, then `pip install -e .` (or `uv pip install <pkg>`).

### Paths and run identity

All output path constants are defined in `src/utils/io.py`. Never redefined or changed elsewhere.

| Constant | Default path | Purpose |
| --- | --- | --- |
| `EXPERIMENTS_DIR` | `chronos-lens/experiments/` | git-tracked flat input configs (`<run-id>.yaml`) |
| `RUNS_DIR` | `artifacts/training-runs/` | all training outputs (checkpoints, embeddings, logs) |
| `ANALYSIS_DIR` | `artifacts/analysis/` | generated analysis artifacts |

A run is identified by `<run-id>`: `experiments/<run-id>.yaml` (input config) ↔ `artifacts/training-runs/<run-id>/` (all outputs) ↔ the `--exp` argument to any script. The `experiments/*/` rule in `.gitignore` enforces no per-run subdirectories in `experiments/` - do not remove it.

Override `RUNS_DIR` with the `CHRONOS_ARTIFACTS_ROOT` env var when the artifacts root changes (e.g. on a remote GPU node).

### Code style

- Each file opens with a multi-line comment describing its purpose. Length proportional to importance.
- Single-line inline comments follow `-- [short comment]` style.
- Comments are instructive and future-facing - no notes about transient changes or review flags.
