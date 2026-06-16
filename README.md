# chronos-lens

Interpretability analysis of Joint Embedding Predictive Architectures (JEPA) applied to longitudinal clinical encounter sequences from MIMIC-IV.

![analysis overview of JEPA clinical representations](analysis.png)

## Motivation

Standard mechanistic interpretability techniques, operating on autoregressive transformers, assume feature representations in softmax'ed token space. Furthermore, the representations clinical models learn remain opaque, demanding interpretabile results in order to remain remain credible and legible by end users. These techniques assume a lossy translation layer - that is: features depend on reconstructions of residual streams, losing valuable signal towards features in the process. This work aims to utilize known autoregressive interpretability techniques and novel manifold-driven analyses to explore the latent representations of JEPA-class models. By treating JEPA's encoder, target, and predictor vectors as first-class objects, this allows the unique oppurtunity to analyze raw encoded embeddings directly.

## Method

This codebase trains (3) models on temporal sequences of MIMIC-IV patient encounters (ICD codes, active medications). All models predict the *embedding* of a masked k$^{th}$ encounter from the $k-1$ subsequent encounters.

- **EMA** variant: exponential moving average target encoder, smooth L1 loss (Assran, 2023)
- **Stop-Gradient** variant: Shared encoder, blocked gradients on the target path, VICReg regularization (Bardes et al, 2022)
- **Supervised transformer** baseline. Uses the same encoder as above JEPA variants, using the masked k$^{th}$ encounter as a label target.

JEPA variant forward passes return three vectors for the encoder, predictor, and target encoded embeddings ($z_{enc}$, $z_{pred}$, $z_{target}$). The (supervised) transformer returns only $z_{enc}$:

- **`z_enc`** `(B, C, D)`: per-encounter encoder representations. "what the encoder learns about each clinical encounter"
- **`z_pred`** `(B, D)`: the predictor's output for the masked encounter - "what the model expects to see"
- **`z_target`** `(B, D)`: the target encoder's representation of the masked encounter - "what actually occured"

## Interpretability

A patient encounter sequence indirectly describes their *trajectory* through the latent embedding space. This provides a wide range of oppurtunities for feature extraction.

- *Can the shape (velocity, curvature, position, etc) of a patient's encounters be described?*
- *Do the activation patterns of encoder representations correspond to clinically meaningful phenotypes? (SAE on `z_enc`)*
- *What does the encoder's manifold encode?* Clinical states and trajectories, comorbidities, temporal information, or all the above?*. 
- *How do encoded representations differ between JEPA-class and an auto-regressive model?*
- *Do these models encode linearly seperable information, and if so where?*

And so many more!

This work runs a combination of classical interpretability techniques, as well as novel approaches for interpreting the "3D" (vectors x time) nature of a sequence of encoded vectors. As this work is still in progress, if you'd like more information, feel free to contact me.

