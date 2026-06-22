"""Early-layer linear probe for detokenization success (paper section 6).

A single class-mean-difference direction, fit on early-layer activations,
predicts whether a split word reaches its canonical representation. The probe is
just two operations: fit a direction on some activations, then score it on
others. The paper's three settings are which activations feed each operation:

    isolated          fit on isolated,     score on isolated    (CV by word)
    in-context        fit on in-context,   score on in-context  (CV by word)
    isolated -> ctx   fit on isolated,     score on in-context  (held-out words)

Labels are the top/bottom canonicity quintiles (middle 60% discarded), measured
at the readout layer (n-2); activations are taken at the probe layer l*.

Pipeline (see `run_probe`):
  1. one deterministic k-piece split per single-token word
  2. isolated + in-context activations at l*, each with quintile labels
  3. fit_direction / score_auroc, within-setting (CV) and isolated -> in-context
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import token_utils
from activations import (
    activation_at_position_batch,
    get_canonicity_layer_idx,
)
from metrics import row_cosine
from data_utils import normalize_pretrained_id

# Probe layer l* per pretrained id and k (paper Table 4). Used to quote the
# headline AUROC; pass `probe_layer` to override.
PAPER_PROBE_LAYERS: dict[str, dict[int, int]] = {
    "gpt-j-6b":      {2: 1, 3: 2, 4: 5},
    "pythia-410m":   {2: 3, 3: 5, 4: 6},
    "pythia-1b":     {2: 3, 3: 5, 4: 6},
    "pythia-6.9b":   {2: 3, 3: 5, 4: 6},
    "llama-2-7b-hf": {2: 2, 3: 5, 4: 5},
    "gemma-2-2b":    {2: 2, 3: 4, 4: 5},
    "bloom-7b1":     {2: 5, 3: 10, 4: 17},
    "opt-1.3b":      {2: 7, 3: 10, 4: 11},
    "opt-6.7b":      {2: 10, 3: 13, 4: 14},
    "gpt-neo-1.3b":  {2: 5, 3: 8, 4: 11},
    "gpt2-large":    {2: 7, 3: 13, 4: 15},
    "gpt2-xl":       {2: 7, 3: 14, 4: 17},
}

# -- Helper functions ----------------------------------------------------

DEFAULT_BATCH_SIZE = 2048
IN_CONTEXT_BATCH_SIZE = 128


def _pad_sequences(model, sequences: list[list[int]]) -> torch.Tensor:
    pad_id = (
        model.tokenizer.pad_token_id
        or model.tokenizer.eos_token_id
        or 0
    )
    max_len = max(len(s) for s in sequences)
    batch = torch.full(
        (len(sequences), max_len),
        pad_id,
        dtype=torch.long,
        device=model.cfg.device,
    )
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=model.cfg.device)
    return batch


def _extract_acts(
    model,
    sequences: list[list[int]],
    position: int | list[int],
    layer: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Variable- or fixed-length seqs -> float32 [N, d] on CPU.
    `position` is an int (same column for all rows) or a per-row list; either form may
    use negative indices, which are resolved against each sequence's true (pre-pad)
    length so the right-padding in `_pad_sequences` is never read."""
    if not sequences:
        return np.empty((0, model.cfg.d_model), dtype=np.float32)
    
    if isinstance(position, int):
        position = [position] * len(sequences)
    position = [int(p) for p in position]
    if len(position) != len(sequences):
        raise ValueError("len(position) must match len(sequences)")
    
    position = [p if p >= 0 else len(s) + p for p, s in zip(position, sequences)]

    chunks: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = _pad_sequences(model, sequences[start : start + batch_size])
        acts = activation_at_position_batch(model, batch, layer, position[start : start + batch_size])
        chunks.append(acts.float().numpy())
    return np.concatenate(chunks, axis=0)

# ---- Probe layer resolution --------------------------------------------

def resolve_probe_layer(model_name: str, num_tokens: int) -> int:
    key = normalize_pretrained_id(model_name)
    try:
        return PAPER_PROBE_LAYERS[key][num_tokens]
    except KeyError as e:
        raise KeyError(
            f"no probe layer for ({model_name!r}, normalized={key!r}, num_tokens={num_tokens}); "
            f"pass probe_layer explicitly"
        ) from e


# -- Labels and the fit / score primitives -----------------------------


def quintile_labels(values: np.ndarray, low_q: float = 0.20, high_q: float = 0.80):
    """Top quintile -> +1 (successful), bottom -> -1 (failed), middle 60% -> 0."""
    lo, hi = float(np.quantile(values, low_q)), float(np.quantile(values, high_q))
    labels = np.zeros(len(values), dtype=np.int8)
    labels[values <= lo] = -1
    labels[values >= hi] = 1
    return labels, lo, hi


def fit_direction(activations: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit direction mean(successful) - mean(failed); y == 1 is successful."""
    direction = activations[y == 1].mean(axis=0) - activations[y == 0].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-8 else np.zeros_like(direction)


def score_auroc(activations: np.ndarray, y: np.ndarray, direction: np.ndarray) -> float:
    """AUROC of the projection onto `direction` against labels y (1 = successful)."""
    if np.linalg.norm(direction) < 1e-8:
        return 0.5
    return float(roc_auc_score(y, activations @ direction))


def cv_auroc(activations, y, groups, cv_folds: int, seed: int):
    """Fit/score across StratifiedGroupKFold splits grouped by word (no word leakage).

    Returns the per-fold AUROCs, their per-fold directions, and the mean unit
    direction over folds (for reuse downstream).
    """
    splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    aurocs, directions = [], []
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("word leakage between train and test")
        direction = fit_direction(activations[train], y[train])
        aurocs.append(score_auroc(activations[test], y[test], direction))
        directions.append(direction)
    aurocs = np.asarray(aurocs, dtype=np.float32)
    mean_direction = np.mean(directions, axis=0)
    mean_direction /= np.linalg.norm(mean_direction) + 1e-8
    return aurocs, np.stack(directions), mean_direction

def shuffle_control_auroc(activations, y, groups, cv_folds, seed, n_shuffles: int = 5) -> float:
    """Mean CV AUROC with labels permuted - should be near 0.5 if the probe's
    separation is real geometry and not an artifact of the fitting procedure."""
    rng = np.random.default_rng(seed)
    aurocs = [
        float(cv_auroc(activations, rng.permutation(y), groups, cv_folds, seed + i)[0].mean())
        for i in range(n_shuffles)
    ]
    return float(np.mean(aurocs))

# -- Word pool: one deterministic k-piece split per single-token word --


def build_split_entries(model, words: Sequence[str], k: int) -> dict[str, dict[str, Any]]:
    """For each word, keep its canonical token id and the first valid k-piece split."""
    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    valid_tokens = set(vocab.keys())
    prefix = token_utils.word_start_prefix(tokenizer)

    entries: dict[str, dict[str, Any]] = {}
    for word in tqdm.tqdm(words, desc=f"k={k} split entries"):
        canonical_id = vocab.get(f"{prefix}{word}" if prefix else word)
        if canonical_id is None:
            continue
        
        segmentations = token_utils.segment_word(word, valid_tokens, prefix, k)
        if not segmentations:
            continue
        split_ids = [vocab[piece] for piece in segmentations[0]]
        entries[word] = {
            "word": word,
            "canonical_token_id": int(canonical_id),
            "split_token_ids": split_ids,
        }
    return entries


# -- WikiText-103 mining (in-context setting) --------------------------

_WORD_RE = re.compile(r"\b[a-z]+\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def _sentences(text_lines, min_chars: int) -> list[str]:
    """WikiText lines -> sentence strings (collapse whitespace, drop headers/short lines)."""
    out: list[str] = []
    for line in text_lines:
        line = re.sub(r"\s+", " ", line.strip())
        if len(line) < min_chars or line.startswith("="):
            continue
        out += [s.strip() for s in _SENTENCE_RE.findall(line) if len(s.strip()) >= min_chars]
    return out


def _occurrences(sentences, word_set: set[str], max_per_word: int):
    """word -> [(sentence_idx, char_start, char_end), ...], one hit per sentence."""
    found: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for idx, sentence in enumerate(tqdm.tqdm(sentences, desc="indexing sentences")):
        seen: set[str] = set()
        for match in _WORD_RE.finditer(sentence.lower()):
            word = match.group(0)
            if word in seen or word not in word_set or len(found[word]) >= max_per_word:
                continue
            start, end = match.start(), match.end()
            if sentence[start:end] == word:  # skip surface mismatches (e.g. capitalized)
                found[word].append((idx, start, end))
                seen.add(word)
    return found


def _build_record(model, sentence, start, end, word, entry, k, min_prefix, min_suffix, max_len):
    """Aligned (canon, split) token sequences for one WikiText hit, or None.

    Re-tokenizes sentence[:end] and requires its last token to be the canonical
    id, then checks the prefix tokenizes identically with the suffix present.
    Trims a window around the target if the sentence is long.
    """
    if sentence[start:end] != word or start == 0:
        return None
    canonical_id = int(entry["canonical_token_id"])
    split_ids = [int(x) for x in entry["split_token_ids"]]

    head = model.to_tokens(sentence[:end], prepend_bos=True)[0].tolist()
    if not head or head[-1] != canonical_id:
        return None
    canon_pos = len(head) - 1
    full = model.to_tokens(sentence, prepend_bos=True)[0].tolist()
    if full[: canon_pos + 1] != head:  # suffix changed how the prefix tokenizes
        return None

    prefix, suffix = full[:canon_pos], full[canon_pos + 1 :]
    if len(prefix) < min_prefix or len(suffix) < min_suffix:
        return None
    canon_seq = full
    split_seq = prefix + split_ids + suffix
    split_last_pos = canon_pos + k - 1

    if len(canon_seq) > max_len:  # centre a window on the target
        half = max_len // 2
        win = max(0, min(canon_pos - half, len(canon_seq) - max_len))
        if (canon_pos - win) < min_prefix or (win + max_len - canon_pos - 1) < min_suffix:
            return None
        gap = len(split_seq) - len(canon_seq)  # = k - 1
        canon_seq, split_seq = canon_seq[win : win + max_len], split_seq[win : win + max_len + gap]
        canon_pos, split_last_pos = canon_pos - win, split_last_pos - win

    if canon_seq[canon_pos] != canonical_id:
        return None
    if split_seq[canon_pos : canon_pos + len(split_ids)] != split_ids:
        return None
    return {
        "word": word,
        "tokens_canon": [int(x) for x in canon_seq],
        "tokens_split": [int(x) for x in split_seq],
        "canon_pos": int(canon_pos),
        "split_last_pos": int(split_last_pos),
    }


def mine_records(
    model,
    word_entries: dict[str, dict[str, Any]],
    k: int,
    *,
    seed: int,
    contexts_per_word: int = 5,
    target_records: int = 5000,
    max_occurrences_per_word: int = 100,
    min_prefix_tokens: int = 5,
    min_suffix_tokens: int = 3,
    max_context_length: int = 160,
    min_sentence_chars: int = 50,
) -> list[dict[str, Any]]:
    """Up to `contexts_per_word` aligned records per word from WikiText-103."""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    sentences = _sentences(dataset["text"], min_sentence_chars)
    occurrences = _occurrences(sentences, set(word_entries), max_occurrences_per_word)

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for word in tqdm.tqdm(sorted(word_entries), desc="building records"):
        hits = list(occurrences.get(word, []))
        rng.shuffle(hits)
        kept = 0
        for idx, start, end in hits:
            if kept >= contexts_per_word:
                break
            record = _build_record(
                model, sentences[idx], start, end, word, word_entries[word], k,
                min_prefix_tokens, min_suffix_tokens, max_context_length,
            )
            if record is not None:
                records.append(record)
                kept += 1

    if len(records) > target_records:
        records = [records[i] for i in rng.choice(len(records), target_records, replace=False)]
    return records


# -- Activations + labels per setting ----------------------------------


def isolated_activations(model, word_entries, k, probe_layer, readout_layer):
    """[BOS, t1..tk] activations at l*, labelled by isolated canonicity at n-2."""
    words = sorted(word_entries)
    bos, _ = token_utils.leading_bos(model)
    canon_seqs = [bos + [word_entries[w]["canonical_token_id"]] for w in words]
    split_seqs = [bos + word_entries[w]["split_token_ids"] for w in words]
    canon_pos, split_last_pos = len(bos), len(bos) + k - 1

    cos = row_cosine(
        _extract_acts(model, canon_seqs, canon_pos, readout_layer),
        _extract_acts(model, split_seqs, split_last_pos, readout_layer),
    )
    labels, lo, hi = quintile_labels(cos)
    keep = np.where(labels != 0)[0]

    activations = _extract_acts(model, [split_seqs[i] for i in keep], split_last_pos, probe_layer)
    y = (labels[keep] == 1).astype(np.int64)
    groups = np.array([words[i] for i in keep])
    return activations, y, groups, (lo, hi)


def in_context_activations(model, records, readout_layer, probe_layer, batch_size: int = IN_CONTEXT_BATCH_SIZE):
    """WikiText-embedded split activations at l*, labelled by in-context canonicity at n-2."""
    cos = row_cosine(
        _extract_acts(model, [r["tokens_canon"] for r in records], [r["canon_pos"] for r in records], readout_layer, batch_size),
        _extract_acts(model, [r["tokens_split"] for r in records], [r["split_last_pos"] for r in records], readout_layer, batch_size),
    )
    labels, lo, hi = quintile_labels(cos)
    keep = np.where(labels != 0)[0]
    kept = [records[i] for i in keep]
    activations = _extract_acts(
        model, [r["tokens_split"] for r in kept], [r["split_last_pos"] for r in kept], probe_layer, batch_size
    )
    y = (labels[keep] == 1).astype(np.int64)
    groups = np.array([r["word"] for r in kept])
    return activations, y, groups, (lo, hi)


# -- Entry point --------------------------------------------------------


def run_probe(
    model,
    one_token_words: Sequence[str],
    *,
    k: int = 2,
    probe_layer: int | None = None,
    cv_folds: int = 5,
    seed: int = 313,
    train_frac: float = 0.80,
) -> dict[str, Any]:
    """Run the section-6 probe at l* for one (model, k).

    Reports AUROC for the isolated and in-context probes (CV by word) and for the
    isolated direction applied to in-context activations of held-out words.
    """
    model.eval()
    torch.set_grad_enabled(False)
    model_name = str(model.cfg.model_name)
    canonicity_layer = get_canonicity_layer_idx(model)
    if probe_layer is None:
        probe_layer = resolve_probe_layer(model_name, num_tokens=k)

    word_entries = build_split_entries(model, list(one_token_words), k)
    records = mine_records(model, word_entries, k, seed=seed)

    iso_acts, iso_y, iso_groups, iso_thr = isolated_activations(model, word_entries, k, probe_layer, canonicity_layer)
    ctx_acts, ctx_y, ctx_groups, ctx_thr = in_context_activations(model, records, canonicity_layer, probe_layer)

    iso_aurocs, _, iso_direction = cv_auroc(iso_acts, iso_y, iso_groups, cv_folds, seed)
    ctx_aurocs, _, _ = cv_auroc(ctx_acts, ctx_y, ctx_groups, cv_folds, seed)

    iso_shuffle = shuffle_control_auroc(iso_acts, iso_y, iso_groups, cv_folds, seed)
    ctx_shuffle = shuffle_control_auroc(ctx_acts, ctx_y, ctx_groups, cv_folds, seed)

    # isolated -> in-context: fit on isolated training words, score in-context held-out words.
    train, _ = train_test_split(np.arange(len(iso_groups)), train_size=train_frac, stratify=iso_y, random_state=seed)
    train_words = set(iso_groups[train])
    in_train = np.array([w in train_words for w in iso_groups])
    held_out = np.array([w not in train_words for w in ctx_groups])
    transfer_direction = fit_direction(iso_acts[in_train], iso_y[in_train])
    transfer_auroc = score_auroc(ctx_acts[held_out], ctx_y[held_out], transfer_direction)

    return {
        "model_name": model_name,
        "k": k,
        "canonicity_layer": canonicity_layer,
        "probe_layer": probe_layer,
        "isolated": {"auroc": float(iso_aurocs.mean()), "std": float(iso_aurocs.std()),
               "shuffle_auroc": iso_shuffle, "thresholds": iso_thr},
        "in_context": {"auroc": float(ctx_aurocs.mean()), "std": float(ctx_aurocs.std()), 
                "shuffle_auroc": ctx_shuffle, "thresholds": ctx_thr},
        "isolated_to_in_context": {"auroc": transfer_auroc, "n_held_out": int(held_out.sum())},
        "iso_direction": iso_direction,
    }