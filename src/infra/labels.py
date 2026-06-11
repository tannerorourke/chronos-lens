"""
Label / target / split / subset preparation for analysis.

All lookups key the patients dict (from sequences.jsonl) by
(subject_id, mask_pos), keeping labels causal per-encounter: the label at
sample i describes encounter mask_pos[i] given only encounters [0, mask_pos-1].
"""
import json
from pathlib import Path

import numpy as np


# =============================================================================
# Patient sequences (dict) -> labels
# =============================================================================

def load_label_30d_at_k(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """
    Load per-sample causal 30d readmission label at each sample's mask position.
    Reads patient["label_30d_per_enc"][k] for each (subject_id, mask_pos=k) pair.
    """
    labels = np.zeros(len(subject_ids), dtype=np.int64)
    for i, (sid, pos) in enumerate(zip(subject_ids, mask_pos)):
        patient = patients_dict[str(sid)]
        per_enc = patient.get("label_30d_per_enc", [])
        pos = int(pos)
        if pos < len(per_enc):
            labels[i] = per_enc[pos]
    return labels


def load_label(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    label_key: str,
    mask_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Load per-sample binary label from patients_dict.

    For per-encounter labels (label_30d, label_escalation), mask_pos is required
    to select the correct encounter position.
    """
    if label_key == "label_30d":
        if mask_pos is not None:
            return load_label_30d_at_k(patients_dict, subject_ids, mask_pos)
        # Fallback for patient-level callers (e.g. supervised): use last encounter
        return np.array([
            patients_dict[str(sid)].get("label_30d_per_enc", [0])[-1]
            for sid in subject_ids
        ], dtype=np.int64)
    elif label_key == "label_escalation":
        if mask_pos is not None:
            return load_escalation_labels(patients_dict, subject_ids, mask_pos)
        return np.array([
            patients_dict[str(sid)].get("label_escalation", 0)
            for sid in subject_ids
        ], dtype=np.int64)
    else:
        raise ValueError(f"[load_label] Unknown label key: {label_key}")


def load_escalation_labels(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """ Load per-sample per-encounter escalation labels at each sample's mask position """
    labels = np.zeros(len(subject_ids), dtype=np.int64)
    for i, (sid, pos) in enumerate(zip(subject_ids, mask_pos)):
        patient = patients_dict[str(sid)]
        per_enc = patient.get("label_escalation_per_enc", [])
        pos = int(pos)
        if pos < len(per_enc):
            labels[i] = per_enc[pos]
    return labels


# =============================================================================
# Re-run the escalation state machine
# =============================================================================

def compute_escalation_criterions(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute per-encounter escalation criteria for each sample.

    Replays the escalation state machine from encounter 0 to mask_pos-1,
    then checks which criteria fire at mask_pos.

    Returns dict mapping criterion_name -> (N,) binary int64 array.
    """
    from src.utils.constants import ESCALATION_CRITERIA
    from src.mimic.labels import _check_enc_escalation, _update_state
    from src.mimic.helper import get_encounter_f_codes


    N = len(subject_ids)
    criteria_labels = {c: np.zeros(N, dtype=np.int64) for c in ESCALATION_CRITERIA}

    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]

        if pos == 0:
            continue

        # Build prior state from encounters 0 .. pos-1
        # Causal assertion: only encounters [0:pos+1] are accessed
        assert pos < len(encs), (
            f"mask_pos {pos} >= n_encounters {len(encs)} for patient {sid}")

        prior_subcats: dict[str, int] = {}
        prior_f_codes: set[str] = set()
        prior_drug_classes: set[str] = set()
        has_prior_psych_meds = False

        for j in range(pos):
            f_codes = get_encounter_f_codes(encs[j], full=True)
            meds = [m.lower() for m in encs[j].get("meds", [])]
            had_meds = _update_state(
                f_codes, meds, prior_subcats, prior_f_codes, prior_drug_classes)
            has_prior_psych_meds = has_prior_psych_meds or had_meds

        # Check escalation at mask_pos (encounter at pos only - no future data)
        f_codes = get_encounter_f_codes(encs[pos], full=True)
        meds = [m.lower() for m in encs[pos].get("meds", [])]
        fired = _check_enc_escalation(
            f_codes, meds,
            prior_subcats, prior_f_codes, prior_drug_classes,
            has_prior_psych_meds,
        )

        for criterion in fired:
            if criterion in criteria_labels:
                criteria_labels[criterion][i] = 1

    return criteria_labels


def compute_subset_mask(patients: dict[str, dict], subject_ids: np.ndarray, subset):
    n_tot = len(subject_ids)
    if subset == "all":
        return np.ones(n_tot, dtype=bool)

    # f-code subset
    fcode_pids: set[str] = set()
    for pid, p in patients.items():
        for enc in p["encounters"]:
            if any(c.upper().startswith("F3") for c in enc.get("icd_codes", [])):
                fcode_pids.add(pid)
                break
    is_fcode = np.array([str(sid) in fcode_pids for sid in subject_ids])
    subset_mask = is_fcode if subset == "fcode" else ~is_fcode
    print(f"  Subset: {subset} -> {int(subset_mask.sum())}/{n_tot} samples")

    return subset_mask
  
  
def get_absolute_enc_times(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Extract days_since_first for each (subject_id, mask_pos) sample."""
    N = len(subject_ids)
    times = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]
        if pos < len(encs):
            times[i] = encs[pos].get("days_since_first", pos)
        else:
            times[i] = pos
    return times
  
def get_relative_enc_times(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Extract days since previous encounter for each (subject_id, mask_pos) sample.
      First encounters get 0.
    """
    N = len(subject_ids)
    rel = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]
        if pos > 0 and pos < len(encs):
            t_cur = encs[pos].get("days_since_first", pos)
            t_prev = encs[pos - 1].get("days_since_first", pos - 1)
            rel[i] = t_cur - t_prev
    return rel
  
# =============================================================================
# Patient sequence computations
# =============================================================================

def compute_temporal_split(
    sequences_path: Path,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split samples into train/test by median latest admission date.

    All encounter windows from the same patient are assigned to the same
    split. Patients whose latest admission is strictly before the median
    cutoff go to train; the rest go to test.

    Returns
    -------
    train_mask  : (N,) bool array over subject_ids
    test_mask   : (N,) bool array over subject_ids
    cutoff_iso  : ISO-format string of the cutoff date
    """
    from datetime import datetime

    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    latest: dict[str, datetime] = {}
    for pid, p in patients.items():
        dates = [datetime.fromisoformat(e["admittime"])
                 for e in p["encounters"] if "admittime" in e]
        if dates:
            latest[pid] = max(dates)

    all_dates = sorted(latest.values())
    cutoff = all_dates[len(all_dates) // 2]

    train_mask = np.array([latest.get(str(sid), cutoff) < cutoff
                           for sid in subject_ids])
    test_mask = ~train_mask

    return train_mask, test_mask, cutoff.isoformat()


# =============================================================================
# ICD-10 chapter target extraction
# =============================================================================

def extract_icd_block_targets(
    sequences_path: Path,
    subject_ids: np.ndarray,
    mask_positions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build a binary multi-label matrix of ICD-10 chapters for masked encounters.

    For each (subject_id, mask_position) pair, looks up the masked encounter
    in sequences.jsonl and extracts the first character of every ICD code
    (the ICD-10 chapter letter).  ICD-9 numeric-prefix codes are ignored.

    Parameters
    ----------
    sequences_path : path to sequences.jsonl
    subject_ids    : (N,) str array - patient IDs per sample
    mask_positions : (N,) int array - which encounter was masked (0-indexed)

    Returns
    -------
    targets       : (N, C) int8 binary matrix, columns = active chapters
    chapter_names : list[str] of length C, sorted chapter letters
    """
    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    N = len(subject_ids)
    chapter_sets: list[set[str]] = []
    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_positions[i])
        enc = patients[sid]["encounters"][pos]
        codes = enc.get("icd_codes", [])
        chapters = {c[0] for c in codes if c and c[0].isalpha()}
        chapter_sets.append(chapters)

    all_chapters = sorted(set().union(*chapter_sets)) if chapter_sets else []
    ch_to_idx = {ch: i for i, ch in enumerate(all_chapters)}

    targets = np.zeros((N, len(all_chapters)), dtype=np.int8)
    for i, chapters in enumerate(chapter_sets):
        for ch in chapters:
            targets[i, ch_to_idx[ch]] = 1

    return targets, all_chapters
