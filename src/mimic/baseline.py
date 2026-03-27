import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def flatten_sequences(sequences: list[dict], label_key: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """
    Flatten patient sequences into a feature matrix for baseline evaluation.

    Features (per patient):
      - n_encounters: total encounter count
      - n_unique_icd: count of unique ICD codes across all encounters
      - n_unique_meds: count of unique medications across all encounters
      - n_fcode_encounters: encounters containing any F-code
      - fcode_ratio: fraction of encounters with F-code
      - n_unique_fcodes: distinct F-codes seen
      - mean_meds_per_enc: average active meds per encounter
      - mean_icd_per_enc: average ICD codes per encounter
      - span_days: days from first admit to last admit
      - mean_gap_days: average inter-admission gap
      - icd_ch_{A-Z}: binary flag for ICD-10 chapter presence
    """
    records = []
    for seq in sequences:
        encs = seq["encounters"]
        n_enc = len(encs)

        all_icd = []
        all_meds = []
        n_fcode_enc = 0
        all_fcodes = set()
        icd_chapters = set()

        for enc in encs:
            all_icd.extend(enc["icd_codes"])
            all_meds.extend(enc["meds"])
            fcodes = [c for c in enc["icd_codes"] if c.upper().startswith("F")]
            if fcodes:
                n_fcode_enc += 1
                all_fcodes.update(fcodes)
            for code in enc["icd_codes"]:
                if code:
                    icd_chapters.add(code[0].upper())

        unique_icd = set(all_icd)
        unique_meds = set(all_meds)

        # Time features
        times = sorted(enc["admittime"] for enc in encs)
        span_days = (times[-1] - times[0]).total_seconds() / 86400 if n_enc > 1 else 0
        gaps = [(times[i + 1] - times[i]).total_seconds() / 86400 for i in range(len(times) - 1)]
        mean_gap = np.mean(gaps) if gaps else 0

        rec = {
            "subject_id": seq["subject_id"],
            "label": seq.get(label_key, 0),
            "n_encounters": n_enc,
            "n_unique_icd": len(unique_icd),
            "n_unique_meds": len(unique_meds),
            # "n_fcode_encounters": n_fcode_enc,
            # "fcode_ratio": n_fcode_enc / n_enc,
            # "n_unique_fcodes": len(all_fcodes),
            "mean_meds_per_enc": len(all_meds) / n_enc,
            "mean_icd_per_enc": len(all_icd) / n_enc,
            "span_days": span_days,
            "mean_gap_days": mean_gap,
        }

        # ICD chapter presence flags
        # for ch in "ABCDEGHIJKLMNOPQRSTUVWXYZ":
        #     rec[f"icd_ch_{ch}"] = int(ch in icd_chapters)

        records.append(rec)

    df = pd.DataFrame(records)
    y = df.pop("label").astype(int)
    X = df.drop(columns=["subject_id"])

    return X, y


def run_baseline(sequences: list[dict]) -> dict:
    """ Logistic regression baseline on flattened features. """
    
    print("\n" + "=" * 60)
    print("Logistic Regression baseline")
    print("=" * 60)

    X, y = flatten_sequences(sequences)
    
    assert X.shape[0] == len(sequences)
    assert not X.isnull().any().any(), "FAIL: NaN in flattened features"
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples:  {len(y)} (pos={y.sum()}, neg={len(y) - y.sum()})")

    if y.nunique() < 2:
        print("  ERROR: Only one class present. Cannot run.")
        return {"auc": None, "error": "single_class"}

    print(f"  Positive rate: {(y.mean()):.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.replace([np.inf, -np.inf], np.nan).fillna(0))
    X_test_scaled = scaler.transform(X_test.replace([np.inf, -np.inf], np.nan).fillna(0))
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs")
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n  ROC AUC: {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"]))

    # Top features by absolute coefficient
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coef": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    print("  Top 10 features by |coefficient|:")
    for _, row in coef_df.head(10).iterrows():
        dir = "+" if row["coef"] > 0 else "-"
        print(f"    {dir} {row['feature']:30s}  {row['abs_coef']:.4f}")

    return {
        "auc": auc,
        "model": model,
        "scaler": scaler,
        "coef_df": coef_df,
        "y_test": y_test,
        "y_prob": y_prob,
    }