from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset
    
# =============================================================================
# Utilities
# =============================================================================

def build_vocab(
    patients: list[dict], 
    pad_idx: int, 
    dir: Path,
    save: bool = True
) -> dict[str, int]:
    print(f"[build_vocab] building vocab ([PAD]: {pad_idx})...")
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": pad_idx}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i

    if save:
        with open(dir / "vocab.json", "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, indent=2)

    print(f"   vocab len: {len(vocab)} tokens")
    return vocab


def encode_encounter(
    enc: dict, 
    vocab: dict[str, int], 
    PAD_IDX: int,
    modality: str = "all",
    use_np_int32: bool = True,
) -> list[int] | np.ndarray:
    """Return a list of token indices for one encounter dict.

       modality : "all" | "icd_only" | "meds_only"
    """
    if modality == "icd_only":
        codes = enc.get("icd_codes", [])
    elif modality == "meds_only":
        codes = enc.get("meds", [])
    else:
        codes = enc.get("icd_codes", []) + enc.get("meds", [])
    enc_toks = [vocab[c] for c in codes if c in vocab]
    
    if use_np_int32:
        enc_toks = np.array(enc_toks, dtype=np.int32)
    if len(enc_toks) == 0:
        return [PAD_IDX]
    return enc_toks


# =============================================================================
# Primary JEPA Dataset
# =============================================================================


class MimicDataset(Dataset):
    """One sample per (patient, masked-encounter-index).

    For a patient with N encounters, N samples are created. Each 
    sample uses N-1 encounters as context and 1 as the prediction 
    target. Patients with fewer than 2 encounters are skipped.
    
    Supports ICD-code-only and medication-only modality plus evaluation 
    for 30d admission prediction and next block prediction.
    """

    def __init__(
        self, 
        patients: list[dict], 
        vocab: dict[str, int], 
        data_params: dict, 
        pad_idx: int = 0,
        max_encounters: int | None = None, 
        use_np_int32: bool = True
    ):
        max_encounters = data_params["max_encounters"]
        modality       = data_params["modality"]
        
        self.samples: list[dict] = []
        self.pad_idx = pad_idx
        for p in patients:
            encs = p.get("encounters", [])
            if len(encs) < 2:
                continue
            if max_encounters is not None:
                encs = encs[:max_encounters]
            sid    = str(p["subject_id"])
            tokens = [encode_encounter(e, vocab, self.pad_idx, modality, use_np_int32) 
                      for e in encs]

            for mask_pos in range(len(encs)):
                self.samples.append({
                    "context":    [tokens[i] for i in range(len(encs)) if i != mask_pos],
                    "target":     tokens[mask_pos],
                    "mask_pos":   mask_pos,
                    "subject_id": sid,
                })
        assert len(self.samples) > 0, "[MimicDataset] No training samples produced"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length encounter sequences and token lists for batching."""
    B = len(batch)

    # Determine maximum context length and max tokens-per-encounter
    max_ctx = max(len(item["context"]) for item in batch)
    all_enc_lens = [
        len(enc)
        for item in batch for enc in item["context"]
    ] + [len(item["target"]) for item in batch]
    max_tok = max(all_enc_lens) if all_enc_lens else 1

    # ctx_tokens[b, c, t] - token indices for context encounter c in batch b
    ctx_tokens   = torch.zeros(B, max_ctx, max_tok, dtype=torch.long)
    # ctx_tok_mask[b, c, t] - True where a real token exists (for mean-pool)
    ctx_tok_mask = torch.zeros(B, max_ctx, max_tok, dtype=torch.bool)
    # ctx_pad_mask[b, c] - True where encounter slot c is padding (for attn)
    ctx_pad_mask = torch.ones(B, max_ctx, dtype=torch.bool)

    tgt_tokens   = torch.zeros(B, max_tok, dtype=torch.long)
    tgt_tok_mask = torch.zeros(B, max_tok, dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, enc in enumerate(item["context"]):
            n = len(enc)
            ctx_tokens[i, j, :n]   = torch.tensor(enc, dtype=torch.long)
            ctx_tok_mask[i, j, :n] = True
            ctx_pad_mask[i, j]     = False # this slot is a real encounter

        n = len(item["target"])
        tgt_tokens[i, :n]   = torch.tensor(item["target"], dtype=torch.long)
        tgt_tok_mask[i, :n] = True

    mask_pos   = torch.tensor([item["mask_pos"] for item in batch], dtype=torch.long)
    subject_ids = [item["subject_id"] for item in batch]

    return {
        "ctx_tokens":    ctx_tokens,     # (B, max_ctx, max_tok)
        "ctx_tok_mask":  ctx_tok_mask,   # (B, max_ctx, max_tok)
        "ctx_pad_mask":  ctx_pad_mask,   # (B, max_ctx) True=padding slot
        "tgt_tokens":    tgt_tokens,     # (B, max_tok)
        "tgt_tok_mask":  tgt_tok_mask,   # (B, max_tok)
        "mask_pos":      mask_pos,       # (B,)
        "subject_ids":   subject_ids,
    }


# =============================================================================
# Supervised Dataset
# =============================================================================

class SupervisedDataset(Dataset):
    """One sample per patient using all encounters. Includes label=label_key
       for loss computation. Unlike MimicDataset which creates N samples per 
       patient (one per masked encounter), this creates exactly one sample with 
       all encounters as context and the patient label as the target.
    
       Used by the supervised transformer.
    """

    def __init__(
        self, 
        patients: list[dict], 
        vocab: dict[str, int],
        data_params: dict,
        pad_idx: int = 0, 
        max_encounters: int | None = None
    ):
        max_encounters = data_params["max_encounters"]
        modality       = data_params.get("modality", "all")
        label_key      = data_params.get("label_key", "label_30d")

        self.samples: list[dict] = []
        self.pad_idx = pad_idx
        for p in patients:
            encs = p.get("encounters", [])
            if len(encs) < 2:
                continue
            if max_encounters is not None:
                encs = encs[:max_encounters]
            label  = int(p.get(label_key, 0))
            sid    = str(p["subject_id"])
            tokens = [encode_encounter(e, vocab, pad_idx, modality) for e in encs]
            
            self.samples.append({
                "context":    tokens,
                "subject_id": sid,
                "label":      label,
            })
        assert len(self.samples) > 0, "[SupervisedDataset] No training samples produced"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def supervised_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length encounter sequences for batching (no target split)."""
    B = len(batch)

    max_enc = max(len(item["context"]) for item in batch)
    all_enc_lens = [len(enc) for item in batch for enc in item["context"]]
    max_tok = max(all_enc_lens) if all_enc_lens else 1

    ctx_tokens   = torch.zeros(B, max_enc, max_tok, dtype=torch.long)
    ctx_tok_mask = torch.zeros(B, max_enc, max_tok, dtype=torch.bool)
    ctx_pad_mask = torch.ones(B, max_enc, dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, enc in enumerate(item["context"]):
            n = len(enc)
            ctx_tokens[i, j, :n]   = torch.tensor(enc, dtype=torch.long)
            ctx_tok_mask[i, j, :n] = True
            ctx_pad_mask[i, j]     = False

    labels      = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    subject_ids = [item["subject_id"] for item in batch]

    return {
        "ctx_tokens":   ctx_tokens,
        "ctx_tok_mask": ctx_tok_mask,
        "ctx_pad_mask": ctx_pad_mask,
        "labels":       labels,
        "subject_ids":  subject_ids,
    }