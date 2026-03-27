import torch
from torch.utils.data import Dataset
    

def encode_encounter(enc: dict, vocab: dict[str, int], PAD_IDX: int) -> list[int]:
    """Return a list of token indices for one encounter dict."""
    codes = enc.get("icd_codes", []) + enc.get("meds", [])
    ids = [vocab[c] for c in codes if c in vocab]
    return ids if ids else [PAD_IDX]   # at least one token


class MimicDataset(Dataset):
    """One sample per (patient, masked-encounter-index).

    For a patient with N encounters, N samples are created.  Each sample
    uses N-1 encounters as context and 1 as the prediction target.
    Patients with fewer than 2 encounters are skipped.
    """

    def __init__(self, patients: list[dict], vocab: dict[str, int], data_params: dict, 
                 pad_idx: int = 0,
                 max_encounters: int | None = None, max_tokens: int | None = None,
                 label_key: str = "label"):
        max_encounters = data_params["max_encounters"]
        max_tokens     = data_params["max_tokens"]
        modality       = data_params["modality"]
        eval           = data_params["eval"]
        
        self.samples: list[dict] = []
        self.pad_idx = pad_idx
        self.label_key = label_key
        for p in patients:
            encs = p.get("encounters", [])
            if len(encs) < 2:
                continue
            if max_encounters is not None:
                encs = encs[:max_encounters]
            label  = int(p.get(label_key, 0))
            sid    = str(p["subject_id"])
            tokens = [encode_encounter(e, vocab, self.pad_idx) for e in encs]
            if max_tokens is not None:
                tokens = [t[:max_tokens] for t in tokens]
            for mask_pos in range(len(encs)):
                self.samples.append({
                    "context":    [tokens[i] for i in range(len(encs)) if i != mask_pos],
                    "target":     tokens[mask_pos],
                    "mask_pos":   mask_pos,
                    "subject_id": sid,
                    "label":      label,
                })
        assert len(self.samples) > 0, "[MimicDataset] No training samples produced"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length encounter sequences and token lists for batching."""
    B = len(batch)

    # Determine maximum context length and maximum tokens-per-encounter
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
    labels     = torch.tensor([item["label"]    for item in batch], dtype=torch.long)
    subject_ids = [item["subject_id"] for item in batch]

    return {
        "ctx_tokens":    ctx_tokens,     # (B, max_ctx, max_tok)
        "ctx_tok_mask":  ctx_tok_mask,   # (B, max_ctx, max_tok)
        "ctx_pad_mask":  ctx_pad_mask,   # (B, max_ctx)  True=padding slot
        "tgt_tokens":    tgt_tokens,     # (B, max_tok)
        "tgt_tok_mask":  tgt_tok_mask,   # (B, max_tok)
        "mask_pos":      mask_pos,       # (B,)
        "labels":        labels,         # (B,)
        "subject_ids":   subject_ids,
    }

