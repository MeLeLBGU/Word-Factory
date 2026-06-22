"""
Tokenization utilities for SentencePiece (e.g. Llama) and BPE (e.g. GPT-2, Pythia) models.
"""

from __future__ import annotations

# SentencePiece (Llama, Gemma): word-initial token is prefixed with "▁".
SPIECE_UNDERLINE = "\u2581"
# GPT-2 / GPT-NeoX (Pythia, OPT, BLOOM): leading space merged into next token as "Ġ".
GPT2_WORD_START_PREFIX = "\u0120"  # "Ġ"
def word_start_prefix(tokenizer) -> str:
    """Return the vocab prefix for the first subword of a word (no leading space in text).
    Inspects the tokenizer vocabulary: SentencePiece models use "▁", GPT-2-style BPE
    uses "Ġ". Returns "" if neither convention appears (none of our supported models).
    """
    vocab = tokenizer.get_vocab()
    if any(k.startswith(SPIECE_UNDERLINE) for k in vocab):
        return SPIECE_UNDERLINE
    if any(k.startswith(GPT2_WORD_START_PREFIX) for k in vocab):
        return GPT2_WORD_START_PREFIX
    return ""


# --- Segmentation --------------------------------------------------------

def _segment(
    text: str,
    valid_tokens: set[str],
    is_start: bool,
    memo: dict,
    prefix: str,
    remaining: int | None,
) -> list[list[str]]:
    """Recursively enumerate vocab-consistent segmentations of `text`.
 
    `remaining` is the exact number of pieces still required; pass None to enumerate
    segmentations of any length. The bounded form prunes branches that cannot reach the
    target count, which matters for the higher segment counts (k = 5, 6).
    """
    if remaining is not None:
        if remaining == 0:
            return [[]] if not text else []
        # Each piece covers at least one character, so `remaining` pieces need at least
        # `remaining` characters left.
        if len(text) < remaining:
            return []
    if not text:
        return [[]]
 
    key = (text, is_start, remaining)
    if key in memo:
        return memo[key]
 
    results: list[list[str]] = []
    max_prefix_len = len(text) if remaining is None else len(text) - (remaining - 1)
    for i in range(1, max_prefix_len + 1):
        candidate = (prefix + text[:i]) if is_start else text[:i]
        if candidate in valid_tokens:
            sub_remaining = None if remaining is None else remaining - 1
            for suffix in _segment(
                text[i:], valid_tokens, False, memo, prefix, sub_remaining
            ):
                results.append([candidate] + suffix)
 
    memo[key] = results
    return results
 
 
def segment_word(
    word: str,
    valid_tokens: set[str],
    prefix: str,
    num_segments: int | None = None,
) -> list[list[str]]:
    """All vocab-consistent segmentations of `text` (a word fragment, no leading space).
 
    Model-free hot-path core: the caller supplies a prebuilt `valid_tokens` set and the
    `prefix` from `word_start_prefix`, so neither is rebuilt per word. Safe to call from
    worker processes that hold no model.
 
    When `num_segments` is given, only segmentations into exactly that many pieces are
    returned (with early pruning); otherwise segmentations of every length are returned.
    """
    return _segment(word, valid_tokens, is_start=True, memo={}, prefix=prefix, remaining=num_segments)

def get_all_segmentations(
    word: str,
    tokenizer,
    num_segments: int | None = None,
) -> list[list[str]]:
    valid_tokens = set(tokenizer.get_vocab().keys())
    prefix = word_start_prefix(tokenizer)
    return segment_word(word, valid_tokens, prefix, num_segments)


# --- BOS-aware TransformerLens helpers -----------------------------------------------


def leading_bos(model) -> tuple[list[int], list[str]]:
    """
    Return ([bos_id], [bos_str]) if this model prepends a BOS, else ([], []).
    """
    with_bos = model.to_tokens("a", prepend_bos=True)[0].tolist()
    without_bos = model.to_tokens("a", prepend_bos=False)[0].tolist()
    if len(with_bos) != len(without_bos) + 1:
        return [], []
    return [with_bos[0]], [model.to_str_tokens("a", prepend_bos=True)[0]]


def text_segmentation_to_tokens(
    segmentation: list[str],
    model,
    prepend_bos: bool = True,
) -> tuple[list[int], list[str]]:
    vocab = model.tokenizer.get_vocab()
    ids, strs = leading_bos(model) if prepend_bos else ([], [])
    for piece in segmentation:
        ids.append(vocab[piece])
        strs.append(piece)
    return ids, strs