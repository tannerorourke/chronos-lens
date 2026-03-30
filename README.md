# chronos-lens

Interpretability analysis of Joint Embedding Predictive Architectures (JEPA) applied to longitudinal clinical encounter sequences from MIMIC-IV.

Repository for "Interpreting JEPA Representations for Clinical Encounter Sequences" (O'Rourke, 2026).

## Motivation

Standard mechanistic interpretability techniques, operating on autoregressive transformers, assume feature representations in token space. Furthermore, clinical ML models operating on temporal patient data demand interpretability, yet the representations they learn remain opaque. These techniques assume a lossy translation layer - that is: features depend on reconstructions of residual streams, losing valuable signal towards features in the process.

This project treats the JEPA's latent representations as first-class analytical objects, performing geometric analysis of the raw encoder, target, and predictor vectors, and cross-referencing with labels and clinical metadata. Specifically, the JEPA encodes patient encounter sequences (ICD codes, active medications) and predicts the embedding of a masked encounter given the remaining context. Three architectures are trained:
- **EMA** variant: exponential moving average target encoder, smooth L1 loss (Assran, 2023)
- **Stop-Gradient** variant: Shared encoder, blocked gradients on the target path, VICReg regularization
- **Supervised transformer** baseline

The forward pass returns three objects:
- **`z_enc`** `(B, C, D)`: per-encounter encoder representations - what the encoder learns about each clinical encounter
- **`z_pred`** `(B, D)`: the predictor's output for the masked encounter - what the model expects to see
- **`z_target`** `(B, D)`: the target encoder's representation of the masked encounter - what's actually there

**pred_error (pred $-$ target)** - what the model gets wrong - is also computed for analysis.

Interpretability is probe-based and SAE-focused: linear probes on `z_enc` and `z_pred` test what clinical information is preserved and predicted, while sparse autoencoders on encoder representations and prediction errors decompose the geometry into sparse features.

Core questions:
- What clinical information does the encoder preserve per encounter? (probe `z_enc`)
- What does the predictor expect the masked encounter to contain? (probe `z_pred`)
- Where do predictions diverge from reality, and do those errors have clinical structure? (SAE on `P−T`)
- Do sparse autoencoder features on encoder representations correspond to clinically meaningful phenotypes? (SAE on `z_enc`)
- Can we map latent space embedding predictions to untampered features?


## Architecture

```
Patient encounters: [enc_0, enc_1, ..., enc_N]
        │
        │  mask encounter at position k
        ▼
┌─────────────────┐
│ Context Encoder  │  enc_{≠k} → token embed → mean-pool per encounter → transformer
│ (EncounterEncoder)│  returns z_enc (B, C, D) per-encounter representations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Predictor     │  transformer over z_enc + learnable mask token at position k
│ (Transformer)    │  returns z_pred (B, D) — predicted representation of enc_k
└─────────────────┘

┌─────────────────┐
│ Target Encoder   │  enc_k only → same architecture as context encoder
│ (stop-grad or   │  returns z_target (B, D) — ground truth representation
│  EMA copy)       │
└─────────────────┘

Loss: MSE(z_pred, z_target) + VICReg on z_enc, z_pred  (stop-grad)
      smooth_L1(z_pred, z_target)                       (EMA)
```

## Usage

### Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

Requires Python $\geq$ 3.12.

### Data extraction

Requires MIMIC-IV access via BigQuery:
1. PhysioNet credentials: https://physionet.org/content/mimiciv/
2. Link PhysioNet account to a GCP project: https://physionet.org/settings/cloud/
3. Authenticate locally:
```bash
gcloud auth application-default login
gcloud config set project <your-project-id>
```
4. Update `BQ_PROJECT_ID` in `scripts/extract_mimic.py`
5. Run extraction:
```bash
python -m scripts.extract_mimic
```

### Training
```bash
# Stop-gradient JEPA
python -m scripts.train_jepa --model stopg_42_v01

# EMA JEPA
python -m scripts.train_jepa --model ema_42_v01

# Supervised baseline
python -m scripts.train_jepa --model supervised_v01
```

Runs are configured via `experiments/<model>/config.yaml`. If a run directory already has checkpoints/logs, a new versioned directory is created automatically (e.g., `stopg_42_v01` → `stopg_42_v01_v01-1`).

### Evaluation
```bash
python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt
```

### SAE training
```bash
python -m scripts.train_sae --model stopg_42_v01
```
