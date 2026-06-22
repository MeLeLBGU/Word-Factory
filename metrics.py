from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---- Cosine similarity --------------------------------------------------
def _row_cosine(a, b) -> torch.Tensor:
    """Per-row cosine as a 1-D torch tensor. Accepts numpy or torch; preserves device."""
    a = torch.as_tensor(a, device = "cpu").float()
    b = torch.as_tensor(b, device = "cpu").float()
    return F.cosine_similarity(a, b, dim=-1)

def mean_cosine(activations, canon_activations) -> float:
    """Mean row-wise cosine (scalar canonicity baseline)."""
    return float(_row_cosine(activations, canon_activations).mean())

def row_cosine(activations, canon_activations) -> np.ndarray:
    """Per-row cosine, one canonicity value per item."""
    return _row_cosine(activations, canon_activations).detach().cpu().numpy()

# ---- Metric validation --------------------------------------------------

def next_token_kl(logits_canon, logits_split) -> np.ndarray:
    """Per-row KL(p_canon || p_split) from last-position logits."""
    logp_c = F.log_softmax(logits_canon.float(), dim=-1)
    logp_s = F.log_softmax(logits_split.float(), dim=-1)
    return (logp_c.exp() * (logp_c - logp_s)).sum(-1).cpu().numpy()


def top1_agreement(logits_canon, logits_split) -> np.ndarray:
    """Per-row 1.0 if the argmax next token matches, else 0.0."""
    return (logits_canon.argmax(-1) == logits_split.argmax(-1)).float().cpu().numpy()


def topk_overlap(logits_canon, logits_split, k: int = 5) -> np.ndarray:
    """Per-row |top-k ∩ top-k| / k."""
    a = logits_canon.topk(k, dim=-1).indices
    b = logits_split.topk(k, dim=-1).indices
    overlap = (a.unsqueeze(2) == b.unsqueeze(1)).any(dim=2).sum(dim=1)
    return (overlap.float() / k).cpu().numpy()

# ---- Gap-closed metric --------------------------------------------------

def gap_closed_pct(patched, lo: float, hi: float) -> np.ndarray:
    """Percentage of the (lo -> hi) gap closed by `patched`. Returns an ndarray that
    mirrors the input shape; an (almost) zero gap yields all-NaN."""
    if isinstance(patched, torch.Tensor):
        patched = patched.detach().cpu().numpy()
    patched = np.asarray(patched, dtype=np.float64)
    gap = hi - lo
    if abs(gap) < 1e-9:
        return np.full_like(patched, np.nan)
    return (patched - lo) / gap * 100.0

def first_layer_at_threshold(pct_closed, layers = None, threshold: float = 80.0) -> int | None:
    """First layer where gap-closed reaches `threshold` (the paper's l*80%), or None."""
    pct = np.asarray(pct_closed)
    for i, v in enumerate(pct):
        if v >= threshold:
            return i if layers is None else int(layers[i])
    return None