# Types.md

Reference contract for variables used throughout application.

- N = number of samples (one per patient x mask_pos combination)
- P = number of unique patients
- C = max padded context encounters per sample (varies per batch, padded to max)
- D = embedding dim
- N_valid = total valid (non-padding) encounters across all samples
- F = number of metadata features

## Paths and config

- model_dir: Path
- output_dir: Path
- sequences_path: Path
- config: yaml $\rightarrow$ dict
- seed: int
- architecture: str
- is_supervised: bool
- sae_checkpoints: dict[str, Path]  - { vector_name: checkpoint.pt }

## Patient data

- patients: list[dict]              - patients loaded for training
- patients_dict: dict[str, dict]    - patients loaded for analysis
- subject_ids: (N,) str             - sample-level subject IDs (may repeat)
- patient_subject_ids: (P,) str     - unique patient IDs (from np.unique or pooling)

## Embeddings - all architectures

- z_enc_recency: (N, D)             - recency encounter z_enc[k-1] per sample (the
                                      most-recent context slot; replaces the old
                                      context mean-pool). Stacked per patient by
                                      ascending mask_pos = the encounter trajectory.

## Embeddings - JEPA only    

- z_encs: (N, C, D)                 - encoder output (per-encounter, padded)
- z_pred: (N, D)                    - predictor output (per encounter, masked)
- z_target: (N, D)                  - target encoder output (per encounter, masked)
- pred_error: (N, D)                - z_pred - z_target
- ctx_pad_mask: (N, C) bool        - True = padding encounter slot
- mask_pos: (N,) int64              - which encounter was masked per sample

## Flattened valid (padding removed) encounters from z_encs 

- z_enc_flat: (N_valid, D)         - all valid encounter representations
- enc_subject_ids: (N_valid,) str  - patient ID for each flattened encounter
- enc_indices: (N_valid,) int      - context position index (0..C-1)
- enc_original_indices: (N_valid,) int - original encounter index in patient sequence 
                                         (accounts for mask_pos gap: positions before mask_pos
                                         are unchanged, positions >= mask_pos are shifted +1)

## Patient-level vectors (terminal sample per patient - largest mask_pos, no mean)

- z_pred: (P, D)
- z_target: (P, D)
- pred_error: (P, D)
- z_enc_recency: (P, D)

## Labels - patient-level (length P, indexed by patient_subject_ids)

- label_escalation: (P,) int
- label_30d: (P,) int

## Labels - encounter-level (length N, n/a for supervised)

- label_escalation_enc: (N,) int    - escalation label @ masked encounter position.
                                      indexed by subject_ids + mask_pos

## Metadata - patient-level (length P, aligned to patient_subject_ids)

- has_metadata: bool
- metadata: (P, F)                  - aligned to patient_subject_ids order
- metadata_raw: (P_meta, F)         - original, file order
- metadata_patient_ids: (P_meta,) str - patient IDs in metadata file order
- feature_names: list[str]          - metadata column names, length F
