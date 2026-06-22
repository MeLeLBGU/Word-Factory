"""Load pair-style detokenization datasets into model-ready tensors.

This module is the *loading* layer: it resolves a model to its on-disk dataset tag,
builds the file paths that `build_datasets` writes to, and reads the pair JSON into the
tensor dicts the patching code consumes. It does not run the model for analysis; the
baseline / canon-caching passes live with the patching code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

import csv



def load_words(path: str | Path) -> list[str]:
    """Load words from a newline-delimited text file or a CSV with a 'word' column."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as f:
            return [w.lower() for row in csv.DictReader(f) if (w := (row.get("word") or "").strip())]
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]



# --- Model -> dataset tag ------------------------------------------------------------
# Keys are normalized (lowercased) pretrained ids as they appear on `model.cfg.model_name`.


# Keys: normalize_pretrained_id(model.cfg.model_name). Values: on-disk folder under dataset_root/.
_DATASET_TAG_BY_PRETRAINED_ID: dict[str, str] = {
    "gpt-j-6b":       "gpt-j",
    "pythia-410m":    "pythia-410m",
    "pythia-1b":      "pythia-1b",
    "pythia-6.9b":    "pythia",
    "llama-2-7b-hf":  "llama2",
    "gemma-2-2b":     "gemma2",
    "bloom-7b1":      "bloom-7b1",
    "opt-1.3b":       "opt-1.3b",
    "opt-6.7b":       "opt-6.7b",
    "gpt-neo-1.3b":   "gpt-neo-1.3B",
    "gpt2-large":     "gpt2-large",
    "gpt2-xl":        "gpt",
}

def normalize_pretrained_id(name: str) -> str:
    """Canonical key from model.cfg.model_name (TransformerLens: tail, lowercased)."""
    return str(name).strip().lower().rsplit("/", 1)[-1]

def dataset_tag(model: Any) -> str:
    name = getattr(model.cfg, "model_name", None)
    if not name:
        raise ValueError("model.cfg has no model_name; cannot resolve dataset tag")
    key = normalize_pretrained_id(name)
    try:
        return _DATASET_TAG_BY_PRETRAINED_ID[key]
    except KeyError as e:
        raise KeyError(
            f"No dataset tag for model_name {name!r} (normalized: {key!r}); "
            f"add it to _DATASET_TAG_BY_PRETRAINED_ID."
        ) from e


# --- Path layout ---------------------------------------------------------------------


def vary_pos_pairs_path(
    dataset_root: str | Path,
    n_tokens: int,
    vary_pos: int,
    tag: str,
) -> Path:
    """Path to the pair file for one (n_tokens, vary_pos) under `dataset_root/tag/`."""
    if not (1 <= vary_pos <= n_tokens):
        raise ValueError(f"vary_pos must be in [1, {n_tokens}], got {vary_pos}")
    return (
        Path(dataset_root)
        / tag
        / f"{n_tokens}_tokens_seg"
        / f"vary_pos_{vary_pos}_{tag}.json"
    )


# --- Loading pair datasets -----------------------------------------------------------


def _canon_fields(
    model: Any,
    words: list[str],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Canonical (single-token) ids, embeddings, and full token sequences for `words`.

    The canonical token is the word's first content token (BOS-independent); the full
    sequence follows the model's own `prepend_bos` behaviour.
    """
    canon_ids = torch.tensor(
        [model.to_tokens(w, prepend_bos=False)[0, 0].item() for w in words],
        device=device,
    )
    return {
        "canon_ids": canon_ids,
        "canon_embeds": model.embed.W_E[canon_ids],
        "canon_as_tokens": torch.tensor(
            [model.to_tokens(w)[0].tolist() for w in words],
            device=device,
        ),
    }


def _build_side(
    pairs: list[Mapping[str, Any]],
    suffix: str,
    model: Any,
    device: torch.device | str,
) -> dict[str, Any]:
    """Build one side (high or low) of a pair batch into a tensor dict."""
    words = [p[f"word_{suffix}"] for p in pairs]
    side: dict[str, Any] = {
        "ids": torch.tensor([p[f"tokens_ids_{suffix}"] for p in pairs], device=device),
        "words": words,
        "cos_sims": torch.tensor(
            [p[f"cos_sim_{suffix}"] for p in pairs], device=device
        ),
    }
    side.update(_canon_fields(model, words, device))
    return side


def load_pairs_dataset(
    filepath: str | Path,
    model: Any,
    device: torch.device | str = "cuda",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load one pair JSON into (high, low) tensor dicts.

    Each row has `word_high` / `word_low` and matching `tokens_ids_*` / `cos_sim_*`.
    Returns (None, None) for an empty (but valid) dataset.
    """
    pairs = json.loads(Path(filepath).read_text(encoding="utf-8"))
    if not pairs:
        print(f"empty dataset, skipping: {filepath}")
        return None, None
    return _build_side(pairs, "high", model, device), _build_side(pairs, "low", model, device)


def load_vary_pos_pairs(
    dataset_root: str | Path,
    n_tokens: int,
    vary_pos: int,
    model: Any,
    device: torch.device | str = "cuda",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load the pair file for one (n_tokens, vary_pos)."""
    tag = dataset_tag(model)
    path = vary_pos_pairs_path(dataset_root, n_tokens, vary_pos, tag)
    return load_pairs_dataset(path, model, device)


def load_all_vary_pos(
    dataset_root: str | Path,
    n_tokens: int,
    model: Any,
    device: torch.device | str = "cuda",
) -> dict[str, dict[str, dict[str, Any] | None]]:
    """Load every vary_pos (1..n_tokens) for one n_tokens. Tag is inferred from the model.

    Returns {"vary_1": {"high": ..., "low": ...}, ...}.
    """
    out: dict[str, dict[str, dict[str, Any] | None]] = {}
    for vary_pos in range(1, n_tokens + 1):
        high, low = load_vary_pos_pairs(dataset_root, n_tokens, vary_pos, model, device)
        out[f"vary_{vary_pos}"] = {"high": high, "low": low}
    return out


def load_vary_pos_for_token_counts(
    dataset_root: str | Path,
    n_tokens_list: list[int],
    model: Any,
    device: torch.device | str = "cuda",
) -> dict[str, dict[str, dict[str, dict[str, Any] | None]]]:
    """Load all vary_pos for several token counts.

    Returns {"2_tok": {"vary_1": {"high": ..., "low": ...}, ...}, "3_tok": {...}, ...}.
    """
    out: dict[str, dict[str, dict[str, dict[str, Any] | None]]] = {}
    for n in n_tokens_list:
        out[f"{n}_tok"] = load_all_vary_pos(dataset_root, n, model, device)
    return out