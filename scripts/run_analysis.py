#!/usr/bin/env python3
"""
Unified analysis pipeline for trained JEPA and supervised models.

Four interpretability stages on saved embeddings:
  Stage 1 - Representation characterization
  Stage 2 - Predictive alignment
  Stage 3 - Error decomposition
  Stage 4 - Evaluation and cross-layer synthesis

Each stage is independently callable via CLI flags.  Standardised inputs
(embeddings npz + sequences.jsonl) and outputs (JSON results + numpy
artifacts per stage, saved to the experiment's analysis/ subdirectory).

Usage
-----
  python -m scripts.run_analysis --model stopg_42_v01 --stages 1,2,3,4
  python -m scripts.run_analysis --model stopg_42_v01 --stages 1
  python -m scripts.run_analysis --model stopg_42_v01 --stages 3 --sae-checkpoints pred_error:sae_checkpoint.pt
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats

from src.analysis.geometry import fit_pca, get_pca_stats
from src.analysis.sae import (
    load_sae, 
    extract_sae_activations, 
    inspect_sae_features,
    sae_cluster_crossref, 
    decompose_patient)
from src.analysis.probing import (
    evaluate_binary_probe,
    probe_icd_blocks,
    probe_vectors, 
    probe_encounter_level)
from src.analysis.eval_infra import (
    broadcast_to_samples,
    compute_escalation_criterions,
    compute_subset_mask,
    extract_icd_block_targets,
    load_label, 
    load_escalation_labels,
    select_interesting_patients)
from src.analysis.plotting import build_patient_profile_figure
from src.utils.seed import load_exp_seed, set_global_seed
from src.utils.io import load_json, save_npz, save_json, load_embeddings, EXPERIMENTS_DIR, DATA_DIR
from src.utils.constants import PSYCH_MEDS_FLAT


# ============================================================================
# Settings
# ============================================================================

PCA_TOP_K = 10

# ============================================================================
# Helpers - SAE utilities
# ============================================================================

def _classify_sae_feature(card):
    """Classify SAE feature: clinical_match (OR>3), mixed, or no_match."""
    max_or = 0.0
    for e in card.get("top_enriched_icd", []):
        max_or = max(max_or, e.get("odds_ratio", 0))
    for e in card.get("top_enriched_meds", []):
        max_or = max(max_or, e.get("odds_ratio", 0))
    if max_or > 3.0:
        return "clinical_match"
    if max_or > 1.5:
        return "mixed"
    return "no_match"


def _decoder_cosine_sim(sae_a, sae_b):
    """Cosine similarity matrix between decoder weight columns of two SAEs.

    Returns (n_features_a, n_features_b) array.
    """
    W_a = sae_a.decoder.weight.data.cpu().numpy()   # (D, F_a)
    W_b = sae_b.decoder.weight.data.cpu().numpy()   # (D, F_b)
    W_a = W_a / (np.linalg.norm(W_a, axis=0, keepdims=True) + 1e-8)
    W_b = W_b / (np.linalg.norm(W_b, axis=0, keepdims=True) + 1e-8)
    return W_a.T @ W_b  # (F_a, F_b)


def _sae_feature_comparison(sae_a, sae_b, threshold=0.8):
    """Compare dictionary directions between two SAEs.

    Returns dict with fraction shared, unique-to-a, unique-to-b.
    """
    sim = _decoder_cosine_sim(sae_a, sae_b)
    max_sim_a = sim.max(axis=1)   # best match for each A feature in B
    max_sim_b = sim.max(axis=0)   # best match for each B feature in A
    return {
        "frac_shared_a": float((max_sim_a > threshold).mean()),
        "frac_shared_b": float((max_sim_b > threshold).mean()),
        "n_unique_a": int((max_sim_a <= threshold).sum()),
        "n_unique_b": int((max_sim_b <= threshold).sum()),
        "mean_best_match_a": float(max_sim_a.mean()),
        "mean_best_match_b": float(max_sim_b.mean()),
        "n_features_a": int(sim.shape[0]),
        "n_features_b": int(sim.shape[1]),
    }


# ============================================================================
# Stage 1 - Representation Characterization
# ============================================================================

def _build_encounter_content_labels(patients_dict, subject_ids, mask_pos,
                                    ctx_pad_mask):
    """Build per-encounter binary labels for content attributes.

    Returns dict mapping attribute_name -> (N, C) int8 array.
    Context position j maps to original encounter index j if j < mask_pos,
    else j + 1 (the masked encounter is excluded from context).
    """
    N, C = ctx_pad_mask.shape
    has_fcode = np.zeros((N, C), dtype=np.int8)
    has_psych_med = np.zeros((N, C), dtype=np.int8)

    for i in range(N):
        sid = str(subject_ids[i])
        encs = patients_dict.get(sid, {}).get("encounters", [])
        mp = int(mask_pos[i])
        for j in range(C):
            if ctx_pad_mask[i, j]:
                continue
            orig_idx = j if j < mp else j + 1
            if orig_idx >= len(encs):
                continue
            enc = encs[orig_idx]
            if any(c.upper().startswith("F")
                   for c in enc.get("icd_codes", [])):
                has_fcode[i, j] = 1
            enc_meds = {m.lower() for m in enc.get("meds", [])}
            if enc_meds & PSYCH_MEDS_FLAT:
                has_psych_med[i, j] = 1

    return {"has_fcode": has_fcode, "has_psych_med": has_psych_med}


def _try_layer_probing(ctx: dict, model_dir: Path):
    """Attempt layer-wise probing.  Returns results dict or None."""
    try:
        from src.training.utils.checkpoint import load_model_notrain
        from src.training.utils.datasets import MimicDataset, collate_fn, build_vocab
        from src.utils.io import load_sequences
        from src.analysis.probing import extract_layer_representations
        from torch.utils.data import DataLoader

        ckpts = sorted(model_dir.glob("checkpoints/checkpoint_*.pt"))
        if not ckpts:
            print("    [layer probing] No checkpoint found, skipping")
            return None
        ckpt_path = ckpts[-1]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_params = ctx["config"]["data"]
        patients = load_sequences(n=data_params.get("n_patients", None))

        vocab = load_json(model_dir / "vocab.json")
        if vocab is None:
            vocab = build_vocab(patients, pad_idx=0, dir=model_dir)
        model, _ = load_model_notrain(ckpt_path, device, restore_rng=False)
        dataset = MimicDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(dataset, batch_size=data_params.get("batch_size", 64),
                            shuffle=False, collate_fn=collate_fn, drop_last=False,
                            num_workers=0)

        print("    Extracting layer representations...")
        layer_reps = extract_layer_representations(model, loader, device)

        # Probe each layer with encounter-level escalation labels
        labels = ctx["label_escalation_enc"]
        subject_ids = ctx["subject_ids"]

        layer_results = {}
        for layer_name in sorted(
            k for k in layer_reps
            if k.startswith("layer_") or k == "final"
        ):
            vec = layer_reps[layer_name]
            # layer reps may have different sample count if subset was applied
            if len(vec) != len(labels):
                print(f"    [layer probing] Sample count mismatch ({len(vec)} vs {len(labels)}), skipping")
                return None
            res = probe_vectors(
                { layer_name: vec }, labels, subject_ids, pool_to_patient=False
            )
            layer_results[layer_name] = res[layer_name]
            print(f"      {layer_name}: AUROC={res[layer_name]['mean_auroc']:.4f}")

        return layer_results
    except Exception as e:
        print(f"    [layer probing] Failed: {e}")
        return None


def run_representation(ctx, stage_dir):
    from src.analysis.geometry import compute_icc, fit_tform_phate_2d, fit_tform_umap_2d
    from src.analysis.clustering import run_lasso_enrichment, run_cluster_enrichment
    
    result = {}
    is_sup = ctx["is_supervised"]
    z_enc_pooled = ctx["z_enc_pooled"]

    # --- 1 Dimensionality and structure ---
    print("\n  1.1 Dimensionality and structure")
    
    pca_pat, proj_pat_all, proj_pat_topk = fit_pca(z_enc_pooled, PCA_TOP_K)
    stats_pat = get_pca_stats(pca_pat, PCA_TOP_K, z_enc_pooled.shape[0])
    result["pca_patient"] = stats_pat
    
    save_npz(stage_dir / "pca_patient.npz",
              eigenvalues=np.array(stats_pat["eigenvalues_all"]),
              explained_variance_ratio=pca_pat.explained_variance_ratio_,
              projections_topk=proj_pat_topk)
    print(f"    Patient (z_enc_pooled) PCA:   d_eff={stats_pat['effective_dimensionality']:.1f}, "
          f"signal={stats_pat['n_signal_components']}/{pca_pat.n_components_}")

    vec_for_embed = z_enc_pooled
    if not is_sup:
        z_enc_flat = ctx["z_enc_flat"]
        vec_for_embed = z_enc_flat
        pca_enc, proj_enc_all, proj_enc_topk = fit_pca(z_enc_flat, PCA_TOP_K)
        stats_enc = get_pca_stats(pca_enc, PCA_TOP_K, z_enc_flat.shape[0])
        result["pca_encounter"] = stats_enc
        
        save_npz(stage_dir / "pca_encounter.npz",
                  eigenvalues=np.array(stats_enc["eigenvalues_all"]),
                  explained_variance_ratio=pca_enc.explained_variance_ratio_,
                  projections_topk=proj_enc_topk)
        print(f"    Encounter (z_enc_flat) PCA: d_eff={stats_enc['effective_dimensionality']:.1f}, "
              f"signal={stats_enc['n_signal_components']}/{pca_enc.n_components_}")
        
        print("\n  1.2 Stability analysis (ICC)")
        icc_result = compute_icc(proj_enc_topk, ctx["enc_subject_ids"], PCA_TOP_K)
        result["icc"] = icc_result
        print(f"    Trait PCs: {len(icc_result['trait_pcs'])}, "
              f"State PCs: {len(icc_result['state_pcs'])}, "
              f"Eligible patients: {icc_result['eligible_patients']}")
        

    # UMAP
    print("    Fitting UMAP...")
    umap_emb = fit_tform_umap_2d(vec_for_embed)
    np.save(stage_dir / "umap.npy", umap_emb)

    # PHATE
    try:
        print("    Fitting PHATE...")
        phate_emb = fit_tform_phate_2d(vec_for_embed)
        np.save(stage_dir / "phate.npy", phate_emb)
    except Exception as e:
        print(f"    PHATE skipped: {e}")
        phate_emb = None

    # -- 3 Linear probing --
    print("\n  1.3 Linear probing")
    probing_results = {}

    # -- 4 Encounter-level probing for content attributes --
    if not is_sup:
        enc_labels = _build_encounter_content_labels(
            ctx["patients_dict"], ctx["subject_ids"],
            ctx["mask_pos"], ctx["ctx_pad_mask"]
        )

        for attr_name, attr_labels in enc_labels.items():
            n_pos = int(attr_labels[~ctx["ctx_pad_mask"]].sum())
            n_valid = int((~ctx["ctx_pad_mask"]).sum())
            if n_pos < 5 or (n_valid - n_pos) < 5:
                print(f"    Encounter {attr_name}: insufficient samples, skipping")
                continue
            
            try:
                res = probe_encounter_level(
                    ctx["z_encs"], ctx["ctx_pad_mask"],
                    attr_labels, ctx["subject_ids"])
                probing_results[f"encounter_{attr_name}"] = res
                print(f"    Encounter {attr_name}: AUROC={res['mean_auroc']:.4f}")
            except Exception as e:
                print(f"    Encounter {attr_name}: probe failed: {e}")

    # -- 5 Patient-level probing on z_enc_pooled --
    patient_ids = ctx["patient_subject_ids"]
    for label_name, label_arr in [
        ("escalation", ctx["label_escalation"]),
        ("30d", ctx["label_30d"])
    ]:
        n_pos = int(label_arr.sum())
        if n_pos < 5 or (len(label_arr) - n_pos) < 5:
            print(f"    Patient {label_name}: insufficient samples, skipping")
            continue
        
        res = probe_vectors(
            {"z_enc_pooled": z_enc_pooled},
            label_arr, patient_ids, pool_to_patient=False
        )
        probing_results[f"patient_{label_name}"] = res["z_enc_pooled"]
        print(f"    Patient {label_name}: AUROC={res['z_enc_pooled']['mean_auroc']:.4f}")

    # -- 6 Layer-wise probing --
    layer_probe = _try_layer_probing(ctx, ctx["model_dir"])
    if layer_probe is not None:
        probing_results["layer_wise"] = layer_probe

    result["probing"] = probing_results

    # ---- 7 Tier A: LASSO -------------------------------------------------
    print("\n  7. Tier A: LASSO enrichment")
    patient_meta = ctx["metadata"]
    feat_names = ctx["feature_names"]

    lasso_pat = run_lasso_enrichment(
        proj_pat_topk, patient_meta, feat_names, top_k=PCA_TOP_K)
    result["lasso_patient"] = {
        k: v for k, v in lasso_pat.items()
        if not isinstance(v, np.ndarray)
    }
    result["lasso_patient"]["mean_r2"] = lasso_pat["mean_r2"]
    result["lasso_patient"]["unexplained_variance_fraction"] = (
        lasso_pat["unexplained_variance_fraction"])
    print(f"    Patient LASSO: mean R²={lasso_pat['mean_r2']:.4f}, "
          f"unexplained={lasso_pat['unexplained_variance_fraction']:.4f}")

    if not is_sup:
        enc_meta = broadcast_to_samples(
            ctx["metadata_raw"], ctx["metadata_patient_ids"],
            ctx["enc_subject_ids"])
        lasso_enc = run_lasso_enrichment(proj_enc_topk, enc_meta, feat_names, 
                                         top_k=PCA_TOP_K, encounter_level=True)
        result["lasso_encounter"] = {
            k: v for k, v in lasso_enc.items()
            if not isinstance(v, np.ndarray)
        }
        result["lasso_encounter"]["mean_r2"] = lasso_enc["mean_r2"]
        result["lasso_encounter"]["unexplained_variance_fraction"] = (
            lasso_enc["unexplained_variance_fraction"])
        print(f"    Encounter LASSO: mean R²={lasso_enc['mean_r2']:.4f}, unexplained="
              f"{lasso_enc['unexplained_variance_fraction']:.4f}")

    # ---- 1.5 Tier B: Clustering ---------------------------------------------
    print("\n  1.5 Tier B: Clustering")
    cluster_results = {}

    embed_meta = (
        broadcast_to_samples(ctx["metadata_raw"], 
                             ctx["metadata_patient_ids"],
                             ctx["enc_subject_ids"])
        if not is_sup else ctx["metadata"]
    )

    # -- PHATE clustering on low-dimensional z_enc_flat --
    if phate_emb is not None:
        try:
            cluster_phate = run_cluster_enrichment(phate_emb, embed_meta, ctx["feature_names"])
            n_unlabeled = len(cluster_phate.get("unlabeled_clusters", []))
            cluster_results["phate"] = {
                "n_clusters": cluster_phate["n_clusters"],
                "n_noise": cluster_phate["n_noise"],
                "n_unlabeled": n_unlabeled,
                "cluster_sizes": cluster_phate.get("cluster_sizes", {}),
                "cluster_labels": cluster_phate.get("cluster_labels"),
            }
            print(f"    PHATE --: {cluster_phate['n_clusters']} clusters, {cluster_phate['n_noise']} noise, {n_unlabeled} unlabeled")
        except Exception as e:
            print(f"    PHATE clustering failed: {e}")

    # -- UMAP clustering on low-dimensional z_enc_flat --
    try:
        cluster_umap = run_cluster_enrichment(umap_emb, embed_meta, ctx["feature_names"])
        n_unlabeled = len(cluster_umap.get("unlabeled_clusters", []))
        cluster_results["umap"] = {
            "n_clusters": cluster_umap["n_clusters"],
            "n_noise": cluster_umap["n_noise"],
            "n_unlabeled": n_unlabeled,
            "cluster_sizes": cluster_umap.get("cluster_sizes", {}),
            "cluster_labels": cluster_umap.get("cluster_labels"),
        }
        print(f"    UMAP --: {cluster_umap['n_clusters']} clusters, {cluster_umap['n_noise']} noise, {n_unlabeled} unlabeled")
    except Exception as e:
        print(f"    UMAP clustering failed: {e}")

    # -- HDBSCAN on high-dimensional z_enc_flat --
    if not is_sup:
        try:
            from hdbscan import HDBSCAN as HDBSCAN_HD
            clusterer = HDBSCAN_HD(min_cluster_size=10, metric="cosine")
            hd_labels = clusterer.fit_predict(z_enc_flat.astype(np.float64))
            n_hd = len(set(hd_labels)) - (1 if -1 in hd_labels else 0)
            n_hd_noise = int((hd_labels == -1).sum())
            cluster_results["high_dim"] = {
                "cluster_labels": hd_labels,
                "n_clusters": n_hd,
                "n_noise": n_hd_noise,
            }
            print(f"    HD clusters:   {n_hd} clusters, {n_hd_noise} noise")
        except Exception as e:
            print(f"    HD clustering failed: {e}")

    result["clustering"] = cluster_results

    # Save cluster labels for cross-stage use
    if "high_dim" in cluster_results:
        np.save(stage_dir / "cluster_labels.npy",
                cluster_results["high_dim"]["cluster_labels"])
    save_json({
        k: {kk: vv for kk, vv in v.items() if kk != "cluster_labels"}
        if isinstance(v, dict) else v
        for k, v in cluster_results.items()
    }, stage_dir / "clusters.json")

    # ---- 1.6 Tier C: SAE on z_enc ------------------------------------------
    sae_results = {}
    z_enc_ckpt = ctx["sae_eval_dict"].get("z_enc")
    if z_enc_ckpt and not is_sup:
        print("\n  1.6 Tier C: SAE on z_enc")
        try:
            device = torch.device("cpu")
            sae_model = load_sae(z_enc_ckpt, device)
            sae_acts = extract_sae_activations(sae_model, z_enc_flat)

            hd_labels_for_sae = cluster_results.get("high_dim", {}).get("cluster_labels")
            feature_cards = inspect_sae_features(
                sae_acts, ctx["enc_subject_ids"], ctx["sequences_path"],
                cluster_labels=hd_labels_for_sae,
                encounter_indices=ctx["enc_original_indices"],
                encounter_level=True)

            classifications = [_classify_sae_feature(c) for c in feature_cards]
            n_clinical = classifications.count("clinical_match")
            n_mixed = classifications.count("mixed")
            n_nomatch = classifications.count("no_match")

            sae_results = {
                "n_features_active": len(feature_cards),
                "n_clinical_match": n_clinical,
                "n_mixed": n_mixed,
                "n_no_match": n_nomatch,
                "feature_cards": feature_cards,
            }
            print(f"    SAE features: {len(feature_cards)} active, "
                  f"{n_clinical} clinical, {n_mixed} mixed, "
                  f"{n_nomatch} no match")

            if hd_labels_for_sae is not None:
                crossref = sae_cluster_crossref(sae_acts, hd_labels_for_sae)
                sae_results["cluster_crossref_summary"] = crossref.get(
                    "summary", "")
                print(f"    SAE-cluster crossref done")

            save_json(sae_results, stage_dir / "sae_features.json")
        except Exception as e:
            print(f"    SAE analysis failed: {e}")

    result["sae"] = sae_results

    # Save main results (strip large arrays)
    save_result = {}
    for k, v in result.items():
        if k in ("clustering",):
            save_result[k] = {
                kk: {kkk: vvv for kkk, vvv in vv.items()
                     if kkk != "cluster_labels"}
                if isinstance(vv, dict) else vv
                for kk, vv in v.items()
            }
        elif k == "sae":
            save_result[k] = {
                kk: vv for kk, vv in v.items()
                if kk != "feature_cards"
            }
        else:
            save_result[k] = v
    save_json(save_result, stage_dir / "representation.json")

    return result


# ============================================================================
# Stage 2 - Predictive Alignment
# ============================================================================

def run_pred_alignment(ctx, stage_dir):
    from src.analysis.geometry import linear_cka, subspace_alignment
    
    result = {}

    z_pred = ctx["z_pred"]
    z_target = ctx["z_target"]
    z_pred_pooled = ctx["z_pred_pooled"]
    z_target_pooled = ctx["z_target_pooled"]
    subject_ids = ctx["subject_ids"]
    patient_ids = ctx["patient_subject_ids"]

    # ---- 2.1 Linear probing -------------------------------------------------
    print("\n  2.1 Linear probing")
    probe_results = {}

    # -- Encounter-level probing (z_pred and z_target vs escalation at mask_pos) --
    label_enc = ctx["label_escalation_enc"]
    if label_enc is not None and label_enc.sum() >= 5:
        enc_res = probe_vectors({"z_pred": z_pred, "z_target": z_target},
            label_enc, subject_ids, pool_to_patient=False
        )
        probe_results["encounter_escalation"] = {
            name: metrics for name, metrics in enc_res.items()
        }
        for name, metrics in enc_res.items():
            print(f"    {name} (escalation): AUROC={metrics['mean_auroc']:.4f}")

    # -- Patient-level probing (pooled vectors vs patient labels) --
    for label_name, label_arr in [
        ("escalation", ctx["label_escalation"]),
        ("30d", ctx["label_30d"])
    ]:
        n_pos = int(label_arr.sum())
        if n_pos < 5 or (len(label_arr) - n_pos) < 5:
            print(f"    Patient {label_name}: insufficient samples, skipping")
            continue
        
        res = probe_vectors({"z_pred_pooled": z_pred_pooled, "z_target_pooled": z_target_pooled},
            label_arr, patient_ids, pool_to_patient=False)
        probe_results[f"patient_{label_name}"] = {
            name: metrics for name, metrics in res.items()
        }
        for name, metrics in res.items():
            print(f"    {name} ({label_name}): AUROC={metrics['mean_auroc']:.4f}")

    result["probing"] = probe_results

    # ---- 2.2 Representational alignment (CKA) -------------------------------
    print("\n  2.2 Representational alignment (CKA)")
    cka_score = linear_cka(z_pred, z_target)
    result["cka"] = cka_score
    print(f"    CKA(z_pred, z_target) = {cka_score:.4f}")

    # ---- 2.3 PCA subspace alignment -----------------------------------------
    print("\n  2.3 PCA subspace alignment")
    pca_pred, _, _ = fit_pca(z_pred, PCA_TOP_K)
    pca_tgt, _, _ = fit_pca(z_target, PCA_TOP_K)
    stats_pred = get_pca_stats(pca_pred, PCA_TOP_K, z_pred.shape[0])
    stats_tgt = get_pca_stats(pca_tgt, PCA_TOP_K, z_target.shape[0])

    sa = subspace_alignment(pca_pred, pca_tgt, PCA_TOP_K)
    result["pca_pred"] = stats_pred
    result["pca_target"] = stats_tgt
    result["subspace_alignment"] = sa

    print(f"    z_pred d_eff={stats_pred['effective_dimensionality']:.1f}, "
          f"z_target d_eff={stats_tgt['effective_dimensionality']:.1f}")
    print(f"    Subspace alignment: mean={sa['mean_alignment']:.4f}, "
          f"min={sa['min_alignment']:.4f}")

    if stats_pred["effective_dimensionality"] < \
            stats_tgt["effective_dimensionality"]:
        print("    -> Predictor is compressing the target representation")

    # ---- 2.4 SAE feature comparison -----------------------------------------
    sae_pred_ckpt = ctx["sae_eval_dict"].get("z_pred")
    sae_tgt_ckpt = ctx["sae_eval_dict"].get("z_target")
    if sae_pred_ckpt and sae_tgt_ckpt:
        print("\n  2.4 SAE feature comparison")
        try:
            device = torch.device("cpu")
            sae_pred = load_sae(sae_pred_ckpt, device)
            sae_tgt = load_sae(sae_tgt_ckpt, device)

            comparison = _sae_feature_comparison(sae_pred, sae_tgt)
            result["sae_comparison"] = comparison
            print(f"    Shared vocabulary (cosine > 0.8): "
                  f"{comparison['frac_shared_a']:.1%} of z_pred features, "
                  f"{comparison['frac_shared_b']:.1%} of z_target features")
            print(f"    Unique to z_pred: {comparison['n_unique_a']}, "
                  f"unique to z_target: {comparison['n_unique_b']}")
        except Exception as e:
            print(f"    SAE comparison failed: {e}")
    else:
        print("\n  2.4 SAE comparison: skipped (need both z_pred and "
              "z_target SAE checkpoints)")

    save_json(result, stage_dir / "alignment.json")
    return result


# ============================================================================
# Stage 3 - Error Decomposition
# ============================================================================

def run_error_decomp(ctx, output_dir):
    from src.analysis.geometry import fit_tform_umap_2d
    from src.analysis.clustering import run_cluster_enrichment
    
    stage_dir = output_dir / "error_decomp"
    stage_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    pred_error = ctx["pred_error"]
    pred_error_pooled = ctx["pred_error_pooled"]
    subject_ids = ctx["subject_ids"]
    patient_ids = ctx["patient_subject_ids"]

    # ---- 3.1 Structure vs noise ---------------------------------------------
    print("\n  3.1 Error structure (PCA)")
    pca_err, proj_err_all, proj_err_topk = fit_pca(pred_error, PCA_TOP_K)
    stats_err = get_pca_stats(pca_err, PCA_TOP_K, pred_error.shape[0])
    result["pca_error"] = stats_err
    save_npz(stage_dir / "error_pca.npz",
              eigenvalues=np.array(stats_err["eigenvalues_all"]),
              explained_variance_ratio=pca_err.explained_variance_ratio_,
              projections_topk=proj_err_topk)

    print(f"    Error PCA: d_eff={stats_err['effective_dimensionality']:.1f}, "
          f"signal={stats_err['n_signal_components']}/{pca_err.n_components_}")

    # ---- 3.2 Magnitude analysis ---------------------------------------------
    print("\n  3.2 Magnitude analysis")
    norms = np.linalg.norm(pred_error, axis=1)
    label_enc = ctx["label_escalation_enc"]

    magnitude_result = {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
    }

    if label_enc is not None and label_enc.sum() >= 5:
        norms_esc1 = norms[label_enc == 1]
        norms_esc0 = norms[label_enc == 0]
        if len(norms_esc1) > 0 and len(norms_esc0) > 0:
            stat_mw, p_mw = stats.mannwhitneyu(norms_esc1, norms_esc0, alternative="two-sided")
            magnitude_result["mann_whitney_U"] = float(stat_mw)
            magnitude_result["mann_whitney_p"] = float(p_mw)
            magnitude_result["mean_norm_esc1"] = float(norms_esc1.mean())
            magnitude_result["mean_norm_esc0"] = float(norms_esc0.mean())
            print(f"    ||error|| esc=1: {norms_esc1.mean():.4f}, "
                  f"esc=0: {norms_esc0.mean():.4f}, "
                  f"p={p_mw:.2e}")

        # Per-criterion magnitude comparison
        criteria_labels = compute_escalation_criterions(ctx["patients_dict"], subject_ids, ctx["mask_pos"])
        criterion_mw = {}
        for crit, crit_labels in criteria_labels.items():
            n_pos = int(crit_labels.sum())
            if n_pos < 5 or (len(crit_labels) - n_pos) < 5:
                continue
            norms_1 = norms[crit_labels == 1]
            norms_0 = norms[crit_labels == 0]
            _, p_val = stats.mannwhitneyu(
                norms_1, norms_0, alternative="two-sided")
            criterion_mw[crit] = {
                "mean_norm_pos": float(norms_1.mean()),
                "mean_norm_neg": float(norms_0.mean()),
                "p_value": float(p_val),
                "n_positive": n_pos,
            }
        magnitude_result["per_criterion"] = criterion_mw

    # Probing on pred_error
    probe_results = {}
    if label_enc is not None and label_enc.sum() >= 5:
        enc_res = probe_vectors(
            {"pred_error": pred_error},
            label_enc, subject_ids, pool_to_patient=False
        )
        probe_results["encounter_escalation"] = enc_res["pred_error"]
        print(f"    pred_error probe (escalation): "
              f"AUROC={enc_res['pred_error']['mean_auroc']:.4f}")

    for label_name, label_arr in [("escalation", ctx["label_escalation"]),
                                  ("30d", ctx["label_30d"])]:
        n_pos = int(label_arr.sum())
        if n_pos < 5 or (len(label_arr) - n_pos) < 5:
            continue
        res = probe_vectors(
            {"pred_error_pooled": pred_error_pooled},
            label_arr, patient_ids, pool_to_patient=False
        )
        probe_results[f"patient_{label_name}"] = res["pred_error_pooled"]
        print(f"    pred_error_pooled ({label_name}): "
              f"AUROC={res['pred_error_pooled']['mean_auroc']:.4f}")

    result["magnitude"] = magnitude_result
    result["probing"] = probe_results

    # ---- 3.3 SAE on pred_error ----------------------------------------------
    sae_pe_ckpt = ctx["sae_eval_dict"].get("pred_error")
    sae_results = {}
    if sae_pe_ckpt:
        print("\n  3.3 SAE on pred_error")
        try:
            device = torch.device("cpu")
            sae_pe = load_sae(sae_pe_ckpt, device)
            sae_acts = extract_sae_activations(sae_pe, pred_error)

            # Load cluster labels from Stage 1 if available
            cluster_path = output_dir / "representation" / "cluster_labels.npy"
            cluster_labels = (np.load(cluster_path) if cluster_path.exists() else None)

            feature_cards = inspect_sae_features(
                sae_acts, subject_ids, ctx["sequences_path"],
                cluster_labels=cluster_labels)

            classifications = [_classify_sae_feature(c)
                               for c in feature_cards]
            sae_results = {
                "n_features_active": len(feature_cards),
                "n_clinical_match": classifications.count("clinical_match"),
                "n_mixed": classifications.count("mixed"),
                "n_no_match": classifications.count("no_match"),
                "feature_cards": feature_cards,
            }
            print(f"    SAE features: {len(feature_cards)} active, "
                  f"{sae_results['n_clinical_match']} clinical, "
                  f"{sae_results['n_mixed']} mixed, "
                  f"{sae_results['n_no_match']} no match")

            # Correlate SAE activations with per-criterion labels
            if label_enc is not None:
                criteria_labels = compute_escalation_criterions(ctx["patients_dict"], subject_ids, ctx["mask_pos"])
                criterion_correlations = {}
                for crit, crit_labels in criteria_labels.items():
                    if crit_labels.sum() < 5:
                        continue
                    # Pearson correlation per SAE feature
                    corrs = []
                    for fi in range(sae_acts.shape[1]):
                        col = sae_acts[:, fi]
                        if col.std() < 1e-10:
                            continue
                        r, _ = stats.pearsonr(col, crit_labels)
                        if abs(r) > 0.1:
                            corrs.append({"feature": fi, "r": float(r)})
                    corrs.sort(key=lambda x: abs(x["r"]), reverse=True)
                    criterion_correlations[crit] = corrs[:10]
                sae_results["criterion_correlations"] = criterion_correlations

            # Cross-reference with Stage 1 clusters
            if cluster_labels is not None:
                crossref = sae_cluster_crossref(sae_acts, cluster_labels)
                sae_results["cluster_crossref_summary"] = crossref.get(
                    "summary", "")

            save_json(sae_results, stage_dir / "sae_features.json")
        except Exception as e:
            print(f"    SAE analysis failed: {e}")

    result["sae"] = sae_results

    # ---- 3.4 Cross-reference with z_enc SAE ---------------------------------
    sae_enc_ckpt = ctx["sae_eval_dict"].get("z_enc")
    crossref_result = {}
    if sae_pe_ckpt and sae_enc_ckpt:
        print("\n  3.4 Cross-reference pred_error SAE vs z_enc SAE")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
            sae_pe = load_sae(sae_pe_ckpt, device)
            sae_enc = load_sae(sae_enc_ckpt, device)

            comparison = _sae_feature_comparison(sae_pe, sae_enc)
            crossref_result = comparison
            frac = comparison["frac_shared_a"]
            print(f"    Fraction of error features aligning with encoder: "
                  f"{frac:.1%}")
            if frac > 0.5:
                print("    -> High alignment: predictor fails along "
                      "dimensions the encoder represents")
            else:
                print("    -> Low alignment: error is orthogonal to "
                      "encoder features")

            save_json(crossref_result, stage_dir / "sae_crossref.json")
        except Exception as e:
            print(f"    Cross-reference failed: {e}")

    result["sae_crossref"] = crossref_result

    # ---- 3.5 Error space clustering -----------------------------------------
    print("\n  3.5 Error space clustering")
    try:
        print("    Fitting error UMAP...")
        error_umap = fit_tform_umap_2d(pred_error)
        np.save(stage_dir / "error_umap.npy", error_umap)

        sample_meta = broadcast_to_samples(ctx["metadata_raw"], ctx["metadata_patient_ids"], subject_ids)
        error_clusters = run_cluster_enrichment(error_umap, sample_meta, ctx["feature_names"])
        n_unlabeled = len(error_clusters.get("unlabeled_clusters", []))
        result["error_clustering"] = {
            "n_clusters": error_clusters["n_clusters"],
            "n_noise": error_clusters["n_noise"],
            "n_unlabeled": n_unlabeled,
        }
        print(f"    Error clusters: {error_clusters['n_clusters']} "
                f"clusters, {error_clusters['n_noise']} noise, "
                f"{n_unlabeled} unlabeled")
    except Exception as e:
        print(f"    Error clustering failed: {e}")

    # Save main results
    save_result = {k: v for k, v in result.items() if k != "sae"}
    save_result["sae"] = {
        k: v for k, v in sae_results.items() if k != "feature_cards"
    }
    save_json(save_result, stage_dir / "error.json")

    return result


# ============================================================================
# Stage 4 - Evaluation & Cross-Layer Synthesis
# ============================================================================

def _check_seed_stability(ctx, output_dir):
    """Check for multi-seed results and compute stability metrics."""
    config = ctx["config"]
    model_dir = ctx["model_dir"]
    current_seed = ctx["exp_seed"]
    current_arch = config.get("model", {}).get("architecture")
    current_dim = config.get("model", {}).get("embed_dim")

    # Find sibling experiments with same architecture but different seeds
    variants = []
    for d in EXPERIMENTS_DIR.iterdir():
        if not d.is_dir() or d == model_dir:
            continue
        cfg_path = d / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if (
                cfg.get("model", {}).get("architecture") == current_arch and
                cfg.get("model", {}).get("embed_dim") == current_dim and
                cfg.get("meta", {}).get("seed") != current_seed
            ):
                analysis_dir = d / "analysis"
                if analysis_dir.exists():
                    variants.append((d, cfg.get("meta", {}).get("seed")))
        except Exception:
            continue

    if not variants:
        print("    No multi-seed variants found")
        return {}

    print(f"    Found {len(variants)} seed variant(s): "
          f"{[s for _, s in variants]}")
    stability = {"seeds_found": [current_seed] + [s for _, s in variants]}

    # Compare probe AUROCs across seeds
    eval_results = {}
    current_eval = output_dir / "evaluation.json"
    if current_eval.exists():
        with open(current_eval) as f:
            eval_results[current_seed] = json.load(f)

    for variant_dir, variant_seed in variants:
        variant_eval = variant_dir / "analysis" / "evaluation.json"
        if variant_eval.exists():
            with open(variant_eval) as f:
                eval_results[variant_seed] = json.load(f)

    if len(eval_results) > 1:
        stability["n_eval_results"] = len(eval_results)
        print(f"    Comparing {len(eval_results)} evaluation results")

    # --- SAE dictionary stability across seeds ---
    from src.analysis.sae import sae_seed_stability
    
    for target in ("pred_error", "z_enc", "z_pred", "z_target"):
        current_ckpt = ctx["sae_eval_dict"].get(target)
        if not current_ckpt:
            continue
        paths_for_target = [current_ckpt]
        for variant_dir, _ in variants:
            # Look for SAE checkpoint in variant dir
            variant_sae = variant_dir / f"sae_{target}" / "sae_checkpoint.pt"
            if variant_sae.exists():
                paths_for_target.append(variant_sae)
        if len(paths_for_target) > 1:
            try:
                sae_stab = sae_seed_stability(paths_for_target)
                stability[f"sae_{target}"] = sae_stab.get("aggregate", {})
                agg = sae_stab.get("aggregate", {})
                print(f"    SAE {target}: "
                      f"mean_cosine={agg.get('mean_cosine', 0):.4f}, "
                      f"frac_stable={agg.get('frac_stable', 0):.1%}")
            except Exception as e:
                print(f"    SAE {target} stability failed: {e}")

    # Cluster stability via adjusted Rand index
    current_cluster = output_dir / "representation" / "cluster_labels.npy"
    if current_cluster.exists():
        from sklearn.metrics import adjusted_rand_score
        current_labels = np.load(current_cluster)
        for variant_dir, variant_seed in variants:
            variant_cluster = (variant_dir / "analysis" / "representation"
                               / "cluster_labels.npy")
            if variant_cluster.exists():
                variant_labels = np.load(variant_cluster)
                if len(variant_labels) == len(current_labels):
                    ari = adjusted_rand_score(current_labels, variant_labels)
                    stability[f"ari_seed_{variant_seed}"] = float(ari)
                    print(f"    Cluster ARI (seed {variant_seed}): {ari:.4f}")

    return stability


def run_eval_synthesis(ctx, output_dir):
    result = {}
    is_sup = ctx["is_supervised"]
    patient_ids = ctx["patient_subject_ids"]
    subject_ids = ctx["subject_ids"]

    # ---- 4.1 Information flow table -----------------------------------------
    print("\n  4.1 Information flow table")
    info_flow = {}

    if is_sup:
        vec_dict = {"z_enc_pooled": ctx["z_enc_pooled"]}
    else:
        vec_dict = {
            "z_enc_pooled": ctx["z_enc_pooled"],
            "z_pred_pooled": ctx["z_pred_pooled"],
            "z_target_pooled": ctx["z_target_pooled"],
            "pred_error_pooled": ctx["pred_error_pooled"],
        }

    for label_name, label_arr in [("escalation", ctx["label_escalation"]),
                                  ("30d", ctx["label_30d"])]:
        n_pos = int(label_arr.sum())
        if n_pos < 5 or (len(label_arr) - n_pos) < 5:
            print(f"    {label_name}: insufficient samples, skipping")
            continue
        res = probe_vectors(
            vec_dict, label_arr, patient_ids,
            pool_to_patient=False
        )
        info_flow[label_name] = {
            name: metrics["mean_auroc"] for name, metrics in res.items()
        }
        for name, metrics in res.items():
            print(f"    {name:25s} {label_name}: "
                  f"AUROC={metrics['mean_auroc']:.4f}")

    result["info_flow"] = info_flow

    # ---- 4.2 Mislabeling gap quantification ---------------------------------
    print("\n  4.2 Mislabeling gap quantification")
    gap = {}

    # (a) LASSO R² from Stage 1
    s1_path = output_dir / "representation" / "stage1_representation.json"
    if s1_path.exists():
        with open(s1_path) as f:
            s1 = json.load(f)
        if "lasso_patient" in s1:
            gap["lasso_r2_patient"] = s1["lasso_patient"].get("mean_r2")
        if "lasso_encounter" in s1:
            gap["lasso_r2_encounter"] = s1["lasso_encounter"].get("mean_r2")
        # (b) Unlabeled clusters
        clustering = s1.get("clustering", {})
        umap_cl = clustering.get("umap", {})
        gap["n_unlabeled_clusters"] = umap_cl.get("n_unlabeled", 0)
        gap["n_total_clusters"] = umap_cl.get("n_clusters", 0)
        # (c) SAE features with no clinical counterpart
        sae_info = s1.get("sae", {})
        n_total = sae_info.get("n_features_active", 0)
        n_nomatch = sae_info.get("n_no_match", 0)
        gap["sae_frac_no_match"] = (
            n_nomatch / n_total if n_total > 0 else None)
    else:
        print("    Stage 1 results not found on disk")

    # Error SAE info from Stage 3
    s3_path = output_dir / "error_decomp" / "error.json"
    if s3_path.exists():
        with open(s3_path) as f:
            s3 = json.load(f)
        sae3 = s3.get("sae", {})
        n_total_3 = sae3.get("n_features_active", 0)
        n_nomatch_3 = sae3.get("n_no_match", 0)
        gap["error_sae_frac_no_match"] = (
            n_nomatch_3 / n_total_3 if n_total_3 > 0 else None)

    result["mislabeling_gap"] = gap
    if gap:
        for k, v in gap.items():
            if v is not None:
                print(f"    {k}: {v}")

    # ---- 4.3 Clinical evaluation --------------------------------------------
    print("\n  4.3 Clinical evaluation")
    clinical = {}

    # Binary probes for escalation and 30d on all pooled vectors
    for label_name, label_arr in [("escalation", ctx["label_escalation"]),
                                  ("30d", ctx["label_30d"])]:
        n_pos = int(label_arr.sum())
        if n_pos < 5 or (len(label_arr) - n_pos) < 5:
            continue
        task_results = {}
        for vec_name, vec in vec_dict.items():
            res = evaluate_binary_probe(vec, label_arr)
            task_results[vec_name] = {
                "auroc": res["mean_auroc"],
                "auprc": res["mean_auprc"],
                "f1": res["mean_f1"],
                "brier": res["mean_brier"],
            }
        clinical[label_name] = task_results

    # ICD block probing on z_pred and z_target
    if not is_sup:
        mask_pos = ctx["mask_pos"]
        try:
            targets, chapter_names = extract_icd_block_targets(
                ctx["sequences_path"], subject_ids, mask_pos)
            for vec_name, vec in [("z_pred", ctx["z_pred"]),
                                  ("z_target", ctx["z_target"])]:
                res = probe_icd_blocks(vec, targets, chapter_names)
                clinical[f"icd_blocks_{vec_name}"] = {
                    "macro_auroc": res["macro_auroc"],
                    "macro_auprc": res.get("macro_auprc"),
                    "n_chapters": res["n_chapters_evaluated"],
                }
                print(f"    ICD blocks ({vec_name}): "
                      f"macro_AUROC={res['macro_auroc']:.4f}")
        except Exception as e:
            print(f"    ICD block probing failed: {e}")

        # Per-criterion escalation probes on z_pred
        criteria_labels = compute_escalation_criterions(ctx["patients_dict"], subject_ids, mask_pos)
        criterion_probes = {}
        for crit, crit_labels in criteria_labels.items():
            n_pos = int(crit_labels.sum())
            if n_pos < 5 or (len(crit_labels) - n_pos) < 5:
                continue
            res = evaluate_binary_probe(ctx["z_pred"], crit_labels)
            criterion_probes[crit] = {
                "auroc": res["mean_auroc"],
                "n_positive": n_pos,
            }
            print(f"    z_pred -> {crit}: AUROC={res['mean_auroc']:.4f} "
                  f"(n={n_pos})")
        clinical["escalation_criteria"] = criterion_probes

    result["clinical"] = clinical

    # ---- 4.4 Per-patient prediction flow ------------------------------------
    if not is_sup:
        print("\n  4.4 Per-patient prediction flow")
        z_pred_pooled = ctx["z_pred_pooled"]
        z_target_pooled = ctx["z_target_pooled"]
        label_esc = ctx["label_escalation"]

        interesting = select_interesting_patients(z_pred_pooled, z_target_pooled, label_esc, n=5)
        result["interesting_patients"] = interesting
        print(f"    Selected {len(interesting)} patients")

        # Build profile figures if any SAE is available
        sae_models = {}
        sae_cards = {}
        for target in ("pred_error", "z_pred", "z_target"):
            ckpt = ctx["sae_eval_dict"].get(target)
            if ckpt:
                try:
                    sae_m = load_sae(ckpt, torch.device("cpu"))
                    sae_models[target] = sae_m
                except Exception:
                    pass

        if sae_models:
            # Build feature cards for available SAEs
            pooled_vecs = {
                "pred_error": ctx["pred_error_pooled"],
                "z_pred": z_pred_pooled,
                "z_target": z_target_pooled,
            }
            for target, sae_m in sae_models.items():
                if target in pooled_vecs:
                    acts = extract_sae_activations(
                        sae_m, pooled_vecs[target])
                    cards = inspect_sae_features(
                        acts, patient_ids, ctx["sequences_path"])
                    sae_cards[target] = cards

            # Build figures
            profile_dir = output_dir / "patient_profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)

            # Get PCA basis from Stage 2 results or fit fresh
            pca_pred, _, _ = fit_pca(z_pred_pooled, PCA_TOP_K)

            for pidx in interesting:
                try:
                    vecs_for_decomp = {
                        k: v for k, v in pooled_vecs.items()
                        if k in sae_models
                    }
                    decomp = decompose_patient(
                        pidx, vecs_for_decomp, sae_models, sae_cards)
                    sid = str(patient_ids[pidx])
                    fig_path = profile_dir / f"patient_{sid}.png"
                    build_patient_profile_figure(
                        pidx, z_pred_pooled, z_target_pooled,
                        decomp, label_esc, pca_pred,
                        show=False, save_path=fig_path)
                    print(f"    Profile saved: {fig_path.name}")
                except Exception as e:
                    print(f"    Profile for patient {pidx} failed: {e}")
        else:
            print("    No SAE checkpoints available for patient profiles")

    # ---- 4.5 Seed stability -------------------------------------------------
    print("\n  4.5 Seed stability")
    seed_result = _check_seed_stability(ctx, output_dir)
    result["seed_stability"] = seed_result

    # Save results
    save_json(result, output_dir / "evaluation.json")
    if seed_result:
        save_json(seed_result, output_dir / "seed_stability.json")

    return result

# ============================================================================
# Context builder
# ============================================================================

def _parse_sae_eval_str(arg_str: str, model_dir: Path):
    """Parse 'target:path,target:path,...' into {target: Path}."""
    result = {}
    if not arg_str:
        return result
    for entry in arg_str.split(","):
        if ":" not in entry:
            continue
        key, path_str = entry.split(":", 1)
        path = Path(model_dir / path_str.strip())
        if not path.exists():
            alt = Path(path_str.strip())
            if alt.exists():
                path = alt
        result[key.strip()] = path
    return result


def build_context(args, model_dir: Path, output_dir: Path, data_dir: Path):
    """Build shared context dict from CLI args and data on disk."""
    from src.utils.io import load_metadata, load_sequences_dict
    from src.analysis.eval_infra import flatten_valid_encounters, pool_to_patients
    
    # -- load config --
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    architecture = config.get("model", {}).get("architecture", "stopgrad")
    is_supervised = architecture == "supervised"
    print(f"  Architecture: {architecture}")
    
    # -- Load embeddings --
    npz, emb_path = load_embeddings(model_dir, args.embeddings)
    subject_ids = npz["subject_ids"].astype(str)
    print(f"  Embeddings:   {emb_path.name}")
    
    # -- SAE checkpoints --
    sae_eval_dict = _parse_sae_eval_str(args.sae_eval, model_dir)
        
    # -- Load patient data --
    sequences_path = Path(data_dir / "sequences.jsonl")
    patients_dict = load_sequences_dict(sequences_path)
    print(f"  Patients: {len(patients_dict)} in sequences.jsonl")
    
    # --- Instantiate context ---
    ctx = {
        "model_dir":        model_dir,
        "output_dir":       output_dir,
        "data_dir":         data_dir,
        "sequences_path":   sequences_path,
        "architecture":     architecture,
        "is_supervised":    is_supervised,
        "model_dim":        config.get("model", {}).get("embed_dim", 0),
        "exp_seed":         config.get("meta", {}).get("seed", 0),
        "patients_dict":    patients_dict,
        "n_patients":       config.get("data", {}).get("n_patients", None),
        "subject_ids":      subject_ids,
        "sae_eval_dict":    sae_eval_dict,
    }

    if is_supervised:
        z_enc_pooled_raw = npz["z_enc_pooled"]
        tensor_dict = { "z_enc_pooled": z_enc_pooled_raw }
    else:
        tensor_dict = {
            "z_encs":         npz["z_encs"],
            "z_pred":         npz["z_pred"],
            "z_target":       npz["z_target"],
            "ctx_pad_mask":  npz["ctx_pad_mask"].astype(bool),
            "mask_pos":       npz["mask_pos"],
        }
    
    # --- Update tensor_dict with eval subset (any/none f-code) ---
    subset = args.eval_subset
    if subset != "all":
        subset_mask = compute_subset_mask(patients_dict, subject_ids, subset)
        subject_ids = subject_ids[subset_mask]
        tensor_dict = { k: v[subset_mask] for k, v in tensor_dict.items() }

    # -- Supervised model: one sample per patient, already pooled --
    if is_supervised:
        z_enc_pooled = tensor_dict["z_enc_pooled"]
        patient_subject_ids = np.unique(subject_ids)
        mask_pos = None

        ctx.update({
            "z_enc_pooled":       z_enc_pooled,
            "patient_subject_ids": patient_subject_ids,
            # No encounter-level data
            "z_encs": None, "z_pred": None, "z_target": None,
            "pred_error": None, "ctx_pad_mask": None, "mask_pos": None,
            "z_enc_flat": None, "enc_subject_ids": None,
            "enc_indices": None, "enc_original_indices": None,
            "z_pred_pooled": None, "z_target_pooled": None,
            "pred_error_pooled": None,
            "label_escalation_enc": None,
            # -- Align labels to target patient ordering --
            "label_30d": load_label(patients_dict, patient_subject_ids, "label_30d"),
            "label_escalation": load_label(patients_dict, patient_subject_ids, "label_escalation"),
        })
    else:
        z_encs = tensor_dict["z_encs"]
        z_pred = tensor_dict["z_pred"]
        z_target = tensor_dict["z_target"]
        pred_error = z_pred - z_target
        ctx_pad_mask = tensor_dict["ctx_pad_mask"]
        mask_pos = tensor_dict["mask_pos"]

        # -- Pool flattened sample z_enc per patient --
        z_enc_flat, enc_subject_ids, enc_indices = \
            flatten_valid_encounters(z_encs, ctx_pad_mask, subject_ids)
        
        z_enc_pooled, _ = pool_to_patients(z_enc_flat, enc_subject_ids)

        # -- Map context position to original encounter index --
        valid_mask = ~ctx_pad_mask.astype(bool)
        sample_idx, ctx_pos = np.where(valid_mask)
        enc_original_indices = np.where(ctx_pos < mask_pos[sample_idx], ctx_pos,
                                        ctx_pos + 1
                                       ).astype(np.int64)

        # -- Pool sample z_pred, z_target, pred_error to p/patient, add "_pooled" suffix --
        pooled, patient_subject_ids = pool_to_patients(
            { "z_pred": z_pred, "z_target": z_target, "pred_error": pred_error }, 
            subject_ids,
            key_suffix="_pooled"
        )
        
        ctx.update({
            "z_encs": z_encs, "z_pred": z_pred, "z_target": z_target, 
            "pred_error": pred_error, "z_enc_flat": z_enc_flat,
            "ctx_pad_mask": ctx_pad_mask, "mask_pos": mask_pos,
            "enc_subject_ids": enc_subject_ids,
            "enc_indices": enc_indices,
            "enc_original_indices": enc_original_indices,
            "patient_subject_ids": patient_subject_ids,
            "z_enc_pooled": z_enc_pooled,
            "z_pred_pooled": pooled["z_pred_pooled"],
            "z_target_pooled": pooled["z_target_pooled"],
            "pred_error_pooled": pooled["pred_error_pooled"],
            # -- Align labels to target patient ordering --
            "label_30d": load_label(patients_dict, patient_subject_ids, "label_30d"),
            "label_escalation": load_label(patients_dict, patient_subject_ids, "label_escalation"),
            "label_escalation_enc": load_escalation_labels(patients_dict, subject_ids, mask_pos)
        })

    # -- Load & align metadata patient ordering to target patient ordering --
    try:
        metadata, feature_names, metadata_patient_ids = load_metadata(data_dir)
        
        meta_idx = {str(pid): i for i, pid in enumerate(metadata_patient_ids)}
        aligned_meta = np.zeros((len(patient_subject_ids), metadata.shape[1]), dtype=metadata.dtype)
        valid_meta = np.zeros(len(patient_subject_ids), dtype=bool)
        for i, pid in enumerate(patient_subject_ids):
            key = str(pid)
            if key in meta_idx:
                aligned_meta[i] = metadata[meta_idx[key]]
                valid_meta[i] = True
        
        n_meta, n_pat = int(valid_meta.sum()), len(patient_subject_ids)
        print(f"  Metadata: {aligned_meta.shape[1]} features, {n_meta}/{n_pat} patients matched")
        
        ctx["metadata"] = aligned_meta
        ctx["metadata_raw"] = metadata
        ctx["metadata_patient_ids"] = metadata_patient_ids
        ctx["feature_names"] = feature_names
    except Exception as e:
        raise Exception(f"Failed to load metadata: {e}")

    # --- Print summary ---
    N_samples = len(ctx["subject_ids"])
    N_patients = len(ctx["patient_subject_ids"])
    D = ctx["z_enc_pooled"].shape[1]
    print(f"  Samples:      {N_samples}")
    print(f"  Patients:     {N_patients}")
    print(f"  Embed dim:    {D}")
    if not is_supervised:
        print(f"  z_enc_flat:   {ctx['z_enc_flat'].shape}")
    if sae_eval_dict:
        print(f"  SAE ckpts:    {list(sae_eval_dict.keys())}")

    return ctx

# ============================================================================
# Summary printers
# ============================================================================

def print_stage_summary(result, stage_num):
    """Print concise summary after a stage completes."""
    print(f"\n  --- Stage {stage_num} Summary ---")

    if stage_num == 1:
        pca_pat = result.get("pca_patient", {})
        pca_enc = result.get("pca_encounter", {})
        print(f"  d_eff: patient={pca_pat.get('effective_dimensionality', '?'):.1f}"
              + (f", encounter={pca_enc.get('effective_dimensionality', '?'):.1f}"
                 if pca_enc else ""))
        icc = result.get("icc", {})
        if icc:
            print(f"  ICC: {len(icc.get('trait_pcs', []))} trait, "
                  f"{len(icc.get('state_pcs', []))} state PCs")
        probing = result.get("probing", {})
        for k, v in probing.items():
            if k == "layer_wise":
                continue
            auroc = v.get("mean_auroc", "?")
            if isinstance(auroc, (int, float)):
                print(f"  Probe {k}: AUROC={auroc:.4f}")
        cl = result.get("clustering", {})
        if "umap" in cl:
            print(f"  Clusters: {cl['umap'].get('n_clusters', '?')} "
                  f"({cl['umap'].get('n_unlabeled', '?')} unlabeled)")
        sae = result.get("sae", {})
        if sae:
            print(f"  SAE: {sae.get('n_clinical_match', 0)} clinical, "
                  f"{sae.get('n_mixed', 0)} mixed, "
                  f"{sae.get('n_no_match', 0)} no match")

    elif stage_num == 2:
        print(f"  CKA = {result.get('cka', '?')}")
        sa = result.get("subspace_alignment", {})
        if sa:
            print(f"  Subspace: mean={sa.get('mean_alignment', '?'):.4f}, "
                  f"min={sa.get('min_alignment', '?'):.4f}")

    elif stage_num == 3:
        pca = result.get("pca_error", {})
        if pca:
            print(f"  Error d_eff = "
                  f"{pca.get('effective_dimensionality', '?'):.1f}")
        mag = result.get("magnitude", {})
        if "mann_whitney_p" in mag:
            print(f"  Magnitude vs escalation: p={mag['mann_whitney_p']:.2e}")
        xref = result.get("sae_crossref", {})
        if xref:
            print(f"  Error-encoder SAE alignment: "
                  f"{xref.get('frac_shared_a', 0):.1%}")

    elif stage_num == 4:
        iflow = result.get("info_flow", {})
        if iflow:
            print("  Information flow (AUROC):")
            for label, vecs in iflow.items():
                parts = [f"{v}={a:.3f}" for v, a in vecs.items()]
                print(f"    {label}: {', '.join(parts)}")


def print_combined_summary(all_results, ctx):
    """Print combined summary table across all stages."""
    arch = ctx["architecture"]
    model_name = ctx["model_dir"].name

    print(f"\n{'=' * 72}")
    print(f"  Combined Summary - {model_name} ({arch})")
    print(f"{'=' * 72}")

    # Representation (Stage 1)
    s1 = all_results.get(1, {})
    if s1:
        print("\n  Representation")
        print("  " + "-" * 68)
        pca_pat = s1.get("pca_patient", {})
        pca_enc = s1.get("pca_encounter", {})
        d_eff_pat = pca_pat.get("effective_dimensionality", "?")
        d_eff_enc = pca_enc.get("effective_dimensionality", "N/A")
        n_sig_pat = pca_pat.get("n_signal_components", "?")
        n_sig_enc = pca_enc.get("n_signal_components", "N/A")
        print(f"    d_eff:           patient={d_eff_pat}, "
              f"encounter={d_eff_enc}")
        print(f"    Signal:          patient={n_sig_pat}, "
              f"encounter={n_sig_enc}")

        icc = s1.get("icc", {})
        if icc:
            print(f"    ICC:             {len(icc.get('trait_pcs', []))} trait, "
                  f"{len(icc.get('state_pcs', []))} state")

        lp = s1.get("lasso_patient", {})
        le = s1.get("lasso_encounter", {})
        if lp:
            r2_pat = lp.get("mean_r2", "?")
            r2_enc = le.get("mean_r2", "N/A") if le else "N/A"
            print(f"    LASSO R²:        patient={r2_pat}, encounter={r2_enc}")

        cl = s1.get("clustering", {})
        if "umap" in cl:
            print(f"    Clusters:        {cl['umap'].get('n_clusters', '?')} "
                  f"({cl['umap'].get('n_unlabeled', '?')} unlabeled)")

        sae = s1.get("sae", {})
        if sae and sae.get("n_features_active"):
            print(f"    SAE:             {sae['n_clinical_match']} clinical, "
                  f"{sae['n_mixed']} mixed, {sae['n_no_match']} no match")

    # Prediction (Stage 2)
    s2 = all_results.get(2, {})
    if s2:
        print("\n  Prediction")
        print("  " + "-" * 68)
        print(f"    CKA:             {s2.get('cka', '?'):.4f}")
        sa = s2.get("subspace_alignment", {})
        if sa:
            print(f"    Subspace:        mean={sa.get('mean_alignment', '?'):.4f}, "
                  f"min={sa.get('min_alignment', '?'):.4f}")
        probing = s2.get("probing", {})
        for label_key in ("patient_escalation", "patient_30d"):
            label_data = probing.get(label_key, {})
            if label_data:
                pred_auc = label_data.get("z_pred_pooled", {}).get(
                    "mean_auroc", "?")
                tgt_auc = label_data.get("z_target_pooled", {}).get(
                    "mean_auroc", "?")
                if isinstance(pred_auc, float) and isinstance(tgt_auc, float):
                    delta = pred_auc - tgt_auc
                    print(f"    Probe {label_key}: z_pred={pred_auc:.4f}, "
                          f"z_target={tgt_auc:.4f}, delta={delta:+.4f}")
        sc = s2.get("sae_comparison", {})
        if sc:
            print(f"    SAE vocab:       "
                  f"{sc.get('frac_shared_a', 0):.0%} shared (cos > 0.8)")

    # Error Decomposition (Stage 3)
    s3 = all_results.get(3, {})
    if s3:
        print("\n  Error Decomposition")
        print("  " + "-" * 68)
        pca_err = s3.get("pca_error", {})
        if pca_err:
            d_eff_err = pca_err.get("effective_dimensionality", "?")
            d_eff_enc_s1 = s1.get("pca_encounter", {}).get(
                "effective_dimensionality") if s1 else None
            ratio = (f"{d_eff_err / d_eff_enc_s1:.2f}"
                     if d_eff_enc_s1 and isinstance(d_eff_err, (int, float))
                     else "N/A")
            print(f"    d_eff:           {d_eff_err} (ratio to encoder: "
                  f"{ratio})")
        mag = s3.get("magnitude", {})
        if "mann_whitney_p" in mag:
            print(f"    Magnitude:       p={mag['mann_whitney_p']:.2e}")
        xref = s3.get("sae_crossref", {})
        if xref:
            print(f"    SAE alignment:   "
                  f"{xref.get('frac_shared_a', 0):.0%} error features "
                  f"align with encoder")

    # Synthesis (Stage 4)
    s4 = all_results.get(4, {})
    if s4:
        print("\n  Synthesis")
        print("  " + "-" * 68)
        iflow = s4.get("info_flow", {})
        if iflow:
            # Build table
            labels = sorted(iflow.keys())
            all_vecs = set()
            for label_data in iflow.values():
                all_vecs.update(label_data.keys())
            all_vecs = sorted(all_vecs)

            header = f"    {'vector':25s}"
            for lb in labels:
                header += f" {lb:>12s}"
            print(header)

            for vec in all_vecs:
                row = f"    {vec:25s}"
                for lb in labels:
                    val = iflow[lb].get(vec)
                    row += f" {val:12.4f}" if val is not None else " " * 13
                print(row)

        gap = s4.get("mislabeling_gap", {})
        if gap:
            print(f"\n    Mislabeling gap:")
            for k, v in gap.items():
                if v is not None:
                    print(f"      {k}: {v}")

        seed_stab = s4.get("seed_stability", {})
        if seed_stab:
            print(f"\n    Seed stability: {seed_stab.get('seeds_found', [])}")

    print()


# ============================================================================
# Main
# ============================================================================

STAGE_CFG = {
    1: ("representation", "Representation Characterization"),
    2: ("prediction", "Predictive Alignment"),
    3: ("error_decomp", "Error Decomposition"),
    4: ("eval_comp", "Evaluation & Cross-Layer Synthesis"),
}

stage_funcs = {
    1: run_representation,
    2: run_pred_alignment,
    3: run_error_decomp,
    4: run_eval_synthesis,
}

parser = argparse.ArgumentParser(description="Interpretability analysis on saved model embeddings")
parser.add_argument("--model", type=str, required=True,
                    help="Experiment directory name under experiments/")
parser.add_argument("--stages", type=str, default="1,2,3,4",
                    help="Comma-separated stage numbers to run (default: all)")
parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                    help="Dataset directory path")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: experiments/<model>/analysis/)")
parser.add_argument("--sae-eval", type=str, default=None,
                    help="SAE checkpoints keyed by target: z_enc:name,pred_error:name,...")
parser.add_argument("--embeddings", type=str, default=None,
                    help="Embeddings filename (e.g. embeddings_40 or embeddings_40.npz). Defaults to highest-epoch.")
parser.add_argument("--eval-subset", type=str, default="all",
                    choices=["all", "fcode", "non_fcode"],
                    help="Filter patients before analysis. 'fcode': patients with at least one F30-39 diagnoses. 'non_fcode': patients with no F30-39 diagnoses. (default: all)")


def main():
    args = parser.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(",")]
    if any(s < 1 or s > 4 for s in stages):
        raise ValueError(f"Unknown stage in {args.stages}.")
    
    model_dir = EXPERIMENTS_DIR / args.model
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    loaded_seed = load_exp_seed(model_dir)
    assert loaded_seed is not None, ValueError(f"Could not load seed for model {args.model}")
    set_global_seed(loaded_seed)

    output_dir = Path(args.output_dir)
    if args.output_dir is None or not Path(args.output_dir).exists():
        output_dir = EXPERIMENTS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Interpretability, but make it JEPA")
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Stages:       {[f"{s}{',' if s != stages[-1] else ''}" for s in stages]}")
    print(f"  Output:       {output_dir}")
    print(f"  Eval subset:  {args.eval_subset}")
    print(f"  Seed:         {loaded_seed}")

    ctx = build_context(args, model_dir, output_dir, data_dir=Path(args.data_dir))

    all_results = {}
    for stage_num in stages:
        stage_dir = output_dir / STAGE_CFG[stage_num][0]
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        if ctx["is_supervised"] and stage_num in (2, 3):
            print(f"\n  Stage {stage_num}: SKIPPED (supervised model - no z_pred/z_target)")
            continue

        print(f"\n{'=' * 72}")
        print(f"  {STAGE_CFG[stage_num][1]}")
        print(f"{'=' * 72}")

        result = stage_funcs[stage_num](ctx, stage_dir)
        all_results[stage_num] = result
        print_stage_summary(result, stage_num)

    if len(all_results) > 1:
        print_combined_summary(all_results, ctx)

    print(f"\n  All results saved to: ../{output_dir.parts[-4]}")
    print("  Done.")


if __name__ == "__main__":
    main()