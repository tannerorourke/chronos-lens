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
    
    return vocab


def encode_encounter(
    enc: dict, 
    vocab: dict[str, int], 
    PAD_IDX: int,
    use_np_int32: bool = True,
) -> list[int] | np.ndarray:
    """Return a list of token indices for one encounter dict. """
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

    For patient with N encounters, N-2 samples are created. patients
    have 3 encounters minimum (see 'src/mimic). Each sample uses 
    encounters [0..mask_pos-1] as causal context and encounter [mask_pos] 
    as prediction target.
    
    In the non-supervised case, "tgt_x" is applied to target encoder.
    In the supervised setting, "tgt_labels" is added. and the rest
    are used to maintain subject reference.
    """

    def __init__(
        self, 
        patients: list[dict], 
        vocab: dict[str, int], 
        data_params: dict, 
        pad_idx: int = 0,
        max_enc: int | None = 100, 
        is_supervised: bool = False,
        label_key: str | None = None,
        use_np_int32: bool = True
    ):
        self.is_supervised = is_supervised
        self.label_key = (data_params.get("label_key", label_key)
                          if self.is_supervised else None)
        
        max_enc = data_params.get("max_encounters", max_enc)
        
        self.samples: list[dict] = []
        self.pad_idx = pad_idx
        for p in patients:
            encs = p.get("encounters", [])
            if len(encs) < 3:
                continue
            if max_enc is not None:
                encs = encs[:max_enc]
            sid    = str(p["subject_id"])
            tokens = [encode_encounter(e, vocab, self.pad_idx, use_np_int32) 
                      for e in encs]
            times  = [e.get("days_since_first", 0) for e in encs]

            for mask_pos in range(2, len(encs)):
                s = {
                    "context":       tokens[:mask_pos],
                    "context_times": times[:mask_pos],
                    "target":        tokens[mask_pos],
                    "target_time":   times[mask_pos],
                    "subject_id":    sid,
                }
                if self.is_supervised:
                    assert label_key in p, f"[MimicDataset] label key missing for patient: {label_key}"
                    s["label"] = p[label_key]
                self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def mimic_collate(self, batch: list[dict]) -> dict:
        """Pad variable-length encounter sequences and token lists for batching."""
        B = len(batch)

        # Determine maximum context length and max tokens-per-encounter
        max_ctx = max(len(item["context"]) for item in batch)
        all_enc_lens = [len(enc) for item in batch for enc in item["context"]]
        if not self.is_supervised:
            all_enc_lens += [len(item["target"]) for item in batch]
        max_tok = max(all_enc_lens) if all_enc_lens else 1

        # [b, c, t] - token t for context encounter c in batch b
        ctx_tokens   = torch.zeros(B, max_ctx, max_tok, dtype=torch.long)
        # [b, c, t] - True t where token exists
        ctx_tok_mask = torch.zeros(B, max_ctx, max_tok, dtype=torch.bool)
        # [b, c]    - True c where encounter slot c is padding (for attn)
        ctx_pad_mask = torch.ones(B, max_ctx, dtype=torch.bool)
        # [b, c]    - days since last admission for encounter c in batch b
        ctx_times    = torch.zeros(B, max_ctx, dtype=torch.long)

        tgt_tokens   = torch.zeros(B, max_tok, dtype=torch.long)
        tgt_tok_mask = torch.zeros(B, max_tok, dtype=torch.bool)

        for s, sample in enumerate(batch):
            for e, enc in enumerate(sample["context"]):
                n = len(enc)
                ctx_tokens[s, e, :n]   = torch.tensor(enc, dtype=torch.long)
                ctx_tok_mask[s, e, :n] = True
                ctx_pad_mask[s, e]     = False

            ct = sample["context_times"]
            ctx_times[s, :len(ct)] = torch.tensor(ct, dtype=torch.long)

            n = len(sample["target"])
            tgt_tokens[s, :n]   = torch.tensor(sample["target"], dtype=torch.long)
            tgt_tok_mask[s, :n] = True

        tgt_times   = torch.tensor([sample["target_time"] for sample in batch], dtype=torch.long)
        subject_ids = [item["subject_id"] for item in batch]
        
        batch_out = {
            "ctx_tokens":        ctx_tokens,
            "ctx_tok_mask":      ctx_tok_mask,
            "ctx_pad_mask":      ctx_pad_mask,
            "ctx_times":         ctx_times,
            "tgt_tokens":        tgt_tokens,
            "tgt_tok_mask":      tgt_tok_mask,
            "tgt_times":         tgt_times,
            "subject_ids":       subject_ids,
        }
        
        if self.is_supervised:
            labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
            batch_out["tgt_labels"] = labels

        return batch_out