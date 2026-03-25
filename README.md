# chronos-lens
Analysis of displacement and geometric latent space analysis of Joint Embedding Predictive Architectures (JEPA) for temporal mental health prediction to aid feature trajectory interpretability

Displacement Representations In Feature Trajectories

Repository for preprint "Interpreting the Geometric Latent Space Analysis of JEPA for time-series feature extraction (UT Austin, 2026).

### Motivation
Clinical ML models operating on temporal patient data demand interpretability, yet the representations they learn remain opaque. Standard mechanistic interpretability techniques, done on autoregressive transformers, assume prediction in token space - JEPAs break this by predicting in a learned latent space with no discrete vocabulary. Furthermore, these techniques assume a lossy translation layer to human words - that is: residual stream $\rightarrow$ logits $\rightarrow$ vocabulary $\rightarrow$ human vocabulary.

This project aims to utilize JEPA's prediction of it's latent embedding space as a medium of mechanistic interpretability (MI) analysis directly, rather than force human labels and discretize them out of the model. Instead of realigning labels to basis directions in an SAE, the core contribution of this work is to identify geometric clusters, basis directions, and bands, and then iterate the **entire** clinical concept space to extract features. In doing this, the hypothesis is the model may show a learned geometry that doesn't exist directly in our vocabulary.

The JEPA encodes patient encounter sequences (ICD codes, active medications) and predicts the embedding of a masked future encounter. A context encoder processes the sequence an EMA-updated target encoder providing training signal. The predictor maps context representations to target representations in latent space. Because $z_{pred}$, $z_{context}$, and $z_{target}$ live in the same embedding space by construction, three displacement vectors decompose the prediction geometry:
- **$\Delta$ (P$-$C)**: the predicted displacement — what the model expects to change
- **pred_error (P$-$T)**: the prediction error — what the model gets wrong
- **observed_traj (T$-$C)**: the observed trajectory — what actually changed

These are geometrically interpretable in ways that softmax'ed models are not. In doing this, we can question:
- what of clinical importance in JEPA's latent space's geometric structure can be explained by clinical metadata via LASSO regression?
- How does the predictor learn the *structure* of the embedding space?
- Can we map latent space embedding predictions to untampered human labels?
- where in the encoder do meaningful representations emerge?
- do principal component axes correspond to stable patients traits vs. dynamic patient state changes?
- how much of each axis of real patient change does the predictor capture vs. miss?

Simply, *can the latent space of a [JEPA] model explain its predictions?*

### Analysis Pipeline
1. **Displacement field construction**: Compute $\Delta_i$ = $z_{pred} - z_{context}$ per patient, decompose into magnitude (uncertainty/volatility) and direction.
2. **PCA & UMAP decomposition**: Eigenvalue spectrum for intrinsic dimensionality, Tracy-Widom test to separate structure from noise, UMAP comparison to validate linearity.
3. **Divergence analysis**: Identify patient pairs with similar histories but divergent predictions; test whether divergence vectors cluster along principal axes.
4. **Stability analysis**: For each top-*k* PC axis of the divergence $\|\Delta\|_2 = \|\z_{pred} - z_{context}\|_2$, compute the ICC (Intraclass correlation coefficients) of per-patient PC projections across encounter windows to classify axes such as trait-like (high ICC) vs. state-like (low ICC).
5. **Labeling bridge**: Connect the model’s geometric structure back to clinical concepts by running three progressively less constrained identification methods analyses on the PC axes:
   - **LASSO regression** of clinical metadata against PC scores for various sets of relevant clinical concepts. Residual variance quantifies structure the model learned beyond the clinical concept space.
   - **UMAP + HBDSCAN cluster enrichment**:
   - **Comparitive test to Sparse AutoEncoder on $\Delta$**: Train a TopK sparse autoencoder on the displacement vectors to learn a dictionary of sparse basis directions that reconstruct $\Delta$. Each learned direction is a "feature" discovered from the geometry itself - not from a human-chosen label set.

Supporting MI techniques (linear probing, activation patching, CKA, and SAE encoding) provide additional context but are secondary to the aim of the project.

### Reproduce
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

This project uses MIMIC-IV via BigQuery. Reproducing requires:
1. Physionet credentials, from https://physionet.org/content/mimiciv/
2. Link your PhysioNet account to a GCP project: https://physionet.org/settings/cloud/
3. Authenticate locally:
```bash
   gcloud auth application-default login
   gcloud config set project 
```
1. Update `BQ_PROJECT_ID` in the config
