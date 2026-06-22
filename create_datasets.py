"""Build the controlled FST / LST pair datasets for the detokenization experiments.

Pipeline (see `create_datasets`):
  1. enumerate vocab-consistent segmentations of each single-token word into N pieces
  2. score each segmentation by cosine similarity to the word's canonical representation
  3. build high/low pairs that vary one token position and fix the rest

Works with any HookedTransformer model and any segment count N >= 2. The output layout
mirrors `data_utils`, so datasets written here are loaded back by `data_utils.load_*`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import tqdm

import token_utils
import data_utils
from activations import activation_at_position_batch, get_canonicity_layer_idx
from metrics import row_cosine

DEFAULT_BATCH_SIZE = 2048


# ── Segmentation -> rows ─────────────────────────────────────────────


def collect_splits(
    model,
    words: list[str],
    num_segments: int,
) -> list[dict[str, Any]]:
    """
    Returns a list of dictionaries, each containing the word, its canonical IDs, its token IDs, and its token strings.
    """
    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    valid_tokens = set(vocab.keys())
    prefix = token_utils.word_start_prefix(tokenizer)
    bos_ids, bos_strs = token_utils.leading_bos(model)
 
    splits: list[dict[str, Any]] = []
    for word in tqdm.tqdm(words, desc=f"segmenting {num_segments}-tokens words"):
        segmentations = token_utils.segment_word(word, valid_tokens, prefix, num_segments)
        if not segmentations:
            continue
        canonical_ids = model.to_tokens(word)[0].tolist()
        for segmentation in segmentations:
            token_ids = bos_ids + [vocab[p] for p in segmentation]
            split: dict[str, Any] = {
                "word": word,
                "canonical_ids": canonical_ids,
                "tokens_ids": token_ids,
                "tokens_string": "|".join(bos_strs + segmentation),
            }
            splits.append(split)
    return splits


# ── Cosine similarity vs. canonical ──────────────────────────────────

def cos_sim_split_vs_canonical(
    model,
    splits: list[dict[str, Any]],
    canonicity_layer_idx: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    """Cosine similarity of each split word's last-position activation vs its canonical activation.

    Returns one float per split, in the same order as `splits`.
    """
    if not splits:
        return []

    canon_pos = len(token_utils.leading_bos(model)[0])
    model.eval()
    
    # Phase 1: canonical activations per unique word.
    unique_words = list(dict.fromkeys(split["word"] for split in splits))
    word_to_canonical_ids = {split["word"]: split["canonical_ids"] for split in splits}
    canonical_activations_dict: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(unique_words), batch_size):
            word_batch = unique_words[start : start + batch_size]
            canonical_ids_batch = torch.tensor(
                [word_to_canonical_ids[w] for w in word_batch],
                device=model.cfg.device,
                dtype=torch.long,
            )
            canonical_activations_batch = activation_at_position_batch(
                model, canonical_ids_batch, canonicity_layer_idx, pos=canon_pos
            )
            for idx, word in enumerate(word_batch):
                canonical_activations_dict[word] = canonical_activations_batch[idx]

    # Phase 2: each split word's last-position activation and its cosine to canonical.
    cos_sims: list[float] = []
    split_word_last_pos = len(splits[0]["tokens_ids"]) - 1
    with torch.no_grad():
        for start in tqdm.tqdm(
            range(0, len(splits), batch_size),
            desc=f"computing cos_sims for {split_word_last_pos}-tok vs canonical",
        ):
            batch = splits[start : start + batch_size]
            split_word_batch = torch.tensor(
                [split["tokens_ids"] for split in batch], device=model.cfg.device, dtype=torch.long
            )
            split_word_activations_batch = activation_at_position_batch(
                model, split_word_batch, canonicity_layer_idx, pos=split_word_last_pos
            )
            canonical_activations_batch = torch.stack(
                [canonical_activations_dict[split["word"]] for split in batch]
            )
            cos = row_cosine(canonical_activations_batch, split_word_activations_batch)
            cos_sims.extend(cos)
    return cos_sims


# ── Pair building ────────────────────────────────────────────────────
 
 
def filter_max_word_appearances(
    pairs: list[dict[str, Any]], max_appearances: int = 3
) -> list[dict[str, Any]]:
    """Greedily keep pairs (highest cos_diff first), capping each word's appearances."""
    word_count: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda x: -x["cos_diff"]):
        if (
            word_count[pair["word_high"]] >= max_appearances
            or word_count[pair["word_low"]] >= max_appearances
        ):
            continue
        word_count[pair["word_high"]] += 1
        word_count[pair["word_low"]] += 1
        out.append(pair)
    return out
 
 
def build_n_token_pairs(
    splits: list[dict[str, Any]],
    num_segments: int,
    min_cos_diff: float,
    max_word_appearances: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Build high/low pairs for each varied position.
    
    Returns a dictionary of lists of pairs, keyed by the varied position.
    """
    if not splits:
        return {}

    n_bos = len(splits[0]["tokens_ids"]) - num_segments
    result: dict[str, list[dict[str, Any]]] = {}

    for vary_pos in range(1, num_segments + 1):
        fixed = [pos for pos in range(1, num_segments + 1) if pos != vary_pos] # All the fixed positions

        groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        for split in splits:
            key = tuple(split["tokens_ids"][n_bos + pos - 1] for pos in fixed)
            groups[key].append(split) # Group the splits by the fixed positions

        pairs: list[dict[str, Any]] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            for i, split_a in enumerate(group):
                for split_b in group[i + 1:]:
                    if split_a["tokens_ids"][n_bos + vary_pos - 1] == split_b["tokens_ids"][n_bos + vary_pos - 1]:
                        continue
                    cos_diff = abs(split_a["cos_sim"] - split_b["cos_sim"])
                    if cos_diff < min_cos_diff:
                        continue

                    high, low = (split_a, split_b) if split_a["cos_sim"] >= split_b["cos_sim"] else (split_b, split_a)

                    pairs.append({
                        "word_high": high["word"],
                        "cos_sim_high": float(high["cos_sim"]),
                        "word_low": low["word"],
                        "cos_sim_low": float(low["cos_sim"]),
                        "cos_diff": float(cos_diff),
                        "tokens_strings_high": high["tokens_string"],
                        "tokens_strings_low": low["tokens_string"],
                        "tokens_ids_high": high["tokens_ids"],
                        "tokens_ids_low": low["tokens_ids"],
                    })

        pairs = filter_max_word_appearances(pairs, max_word_appearances)
        pairs.sort(key=lambda pair: -pair["cos_diff"])
        result[f"vary_pos_{vary_pos}"] = pairs

    return result
 
 
# ── Pipeline ─────────────────────────────────────────────────────────
 
 
def create_datasets(
    model,
    one_token_words: list[str],
    token_counts: Sequence[int] = (2, 3),
    *,
    canonicity_layer_idx: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_pair_cos_diff: float = 0.2, 
    max_word_appearances: int = 3,
) -> dict[str, Any]:
    """Run the full pipeline and return {"meta", "splits": {n: [...], "pairs": {n: [...]}}.
 
    Each word appears in at most `max_word_appearances` pairs per varied position.
    """
    if not one_token_words:
        raise ValueError("one_token_words is empty")
 
    if canonicity_layer_idx is None:
        canonicity_layer_idx = get_canonicity_layer_idx(model)
    counts = sorted({int(n) for n in token_counts if int(n) >= 2})
 
    datasets: dict[str, Any] = {
        "meta": {
            "canonicity_layer_idx": canonicity_layer_idx,
            "batch_size": int(batch_size),
            "token_counts": counts,
            "min_pair_cos_diff": float(min_pair_cos_diff),
            "max_word_appearances": int(max_word_appearances),
        },
        "splits": {},
        "pairs": {},
    }
 
    for n in counts:
        splits = collect_splits(model, one_token_words, n)
        cos_sims = cos_sim_split_vs_canonical(model, splits, canonicity_layer_idx, batch_size)
        for split, cos_sim in zip(splits, cos_sims):
            split["cos_sim"] = cos_sim
 
        datasets["splits"][n] = splits
        datasets["pairs"][n] = build_n_token_pairs(
            splits, n, min_pair_cos_diff, max_word_appearances
        )
        torch.cuda.empty_cache()
 
    return datasets
 
 
# ── Saving ───────────────────────────────────────────────────────────
 
 
def save_datasets(
    datasets: dict[str, Any],
    dataset_root: str | Path,
    model,
) -> list[str]:
    """Save every pair set under the layout `data_utils` reads back.
 
    Files land at `dataset_root/<tag>/<n>_tokens_seg/vary_pos_<k>_<tag>.json`, where the
    tag is resolved from the model, so the builder and loader can never disagree on paths.
    Returns the list of written paths.
    """
    tag = data_utils.dataset_tag(model)
    saved: list[str] = []
    for n, pair_sets in datasets["pairs"].items():
        for vary_key, pairs in pair_sets.items():
            vary_pos = int(vary_key.rsplit("_", 1)[1])
            path = data_utils.vary_pos_pairs_path(dataset_root, n, vary_pos, tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pairs, indent=4, ensure_ascii=False), encoding="utf-8")
            saved.append(str(path))
    return saved