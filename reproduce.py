"""Reproduce results from *Inside the LLM Word Factory* (Busigin & Pinter).

Each public function runs one paper experiment (or figure/table), returns a result dict
with measured values and pass/fail checks against `PAPER_TARGETS`, and accepts optional
`max_pairs` / `max_splits` subsampling for fast smoke runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import tqdm
from scipy.stats import spearmanr
from transformer_lens import utils

import activation_patching as ap
import create_datasets
import data_utils
import interventions
import metrics
import probing
import token_utils
from activations import (
    activation_at_position_batch,
    get_canonicity_layer_idx,
    next_token_logits_batch,
)

# ---------------------------------------------------------------------------
# Paths & model registry
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "datasets"
DEFAULT_WORDS_DIR = REPO_ROOT / "words"
DEFAULT_RAW_WORD_LISTS = (
    DEFAULT_WORDS_DIR / "top_english_words_lower_100000.txt",
    DEFAULT_WORDS_DIR / "google-10000-english.txt",
    DEFAULT_WORDS_DIR / "unigram_freq.csv",
)

# (TransformerLens pretrained id, on-disk dataset tag under dataset_root)
ALL_MODELS: list[tuple[str, str]] = [
    ("gpt-j-6b", "gpt-j"),
    ("pythia-410m", "pythia-410m"),
    ("pythia-1b", "pythia-1b"),
    ("pythia-6.9b", "pythia"),
    ("llama-2-7b", "llama2"),
    ("gemma-2-2b", "gemma-2-2b"),
    ("bloom-7b1", "bloom-7b1"),
    ("opt-1.3b", "opt-1.3b"),
    ("opt-6.7b", "opt-6.7b"),
    ("gpt-neo-1.3b", "gpt-neo-1.3B"),
    ("gpt2-large", "gpt2-large"),
    ("gpt2-xl", "gpt"),
]

DEFAULT_BATCH_SIZE = 2048

# Published headline numbers and tolerances (relative unless noted).
PAPER_TARGETS: dict[str, Any] = {
    "llama-2-7b-hf": {
        "pool_size": (8933, 500),
        "lst_pairs": (6409, 800),
        "fst_pairs": (6254, 800),
        "readout_layer": {"rho_kl": (-0.86, 0.08), "rho_top1": (0.45, 0.10), "rho_top5": (0.71, 0.10)},
        "quintile_top1": (0.78, 0.08),
        "quintile_bottom_top1": (0.24, 0.08),
        "quintile_top_kl": (0.09, 0.05),
        "quintile_bottom_kl": (1.86, 0.30),
        "lst_l1_attn_gap": (53.0, 8.0),
        "lst_l2_mlp_peak_gap": (28.0, 8.0),
        "fst_l1_mlp_gap": (53.0, 8.0),
        "fst_l1_attn_gap": (1.0, 3.0),
        "mlp_necessity_l1_canon": (0.78, 0.05),
        "mlp_unmodified_canon": (0.95, 0.03),
        "head_z27_gap": (17.0, 5.0),
        "head_z_combined_gap": (39.0, 8.0),
        "scaling_l80_k2": (2, 1),
        "scaling_l80_k6": (6, 2),
        "probe_isolated_auroc": (0.94, 0.04),
        "probe_in_context_auroc": (0.97, 0.03),
        "probe_transfer_auroc": (0.91, 0.05),
        "l1_resid_gap_pct": (60.5, 10.0),
        "l80_layer": (2, 1),
    },
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _model_key(model) -> str:
    return data_utils.normalize_pretrained_id(str(model.cfg.model_name))


def _device(model) -> torch.device | str:
    return model.cfg.device


def check_target(
    name: str,
    value: float | int | None,
    target: float | int,
    tol: float,
) -> dict[str, Any]:
    """Return a small dict with measured value and pass/fail vs paper target."""
    if value is None:
        return {"name": name, "value": None, "target": target, "tol": tol, "pass": False}
    ok = abs(float(value) - float(target)) <= float(tol)
    return {
        "name": name,
        "value": float(value) if isinstance(value, (int, float, np.floating)) else value,
        "target": float(target),
        "tol": float(tol),
        "pass": bool(ok),
    }


def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for c in checks if c.get("pass"))
    return {"checks": checks, "passed": passed, "total": len(checks), "all_pass": passed == len(checks)}


def build_single_token_pool(
    model,
    raw_list_paths: Sequence[str | Path] | None = None,
) -> list[str]:
    """Merge raw word lists, lowercase/dedup, keep words encoded as one content token."""
    paths = raw_list_paths or DEFAULT_RAW_WORD_LISTS
    seen: set[str] = set()
    candidates: list[str] = []
    for path in paths:
        for word in data_utils.load_words(path):
            if word not in seen:
                seen.add(word)
                candidates.append(word)

    pool: list[str] = []
    for word in candidates:
        ids = model.to_tokens(word, prepend_bos=False)[0].tolist()
        if len(ids) == 1:
            pool.append(word)
    return pool


def _subsample_indices(n: int, max_n: int | None, seed: int = 0) -> np.ndarray:
    if max_n is None or max_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def load_pair_batch(
    model,
    dataset_root: str | Path,
    n_tokens: int,
    vary_pos: int,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load (high, low) pair sides; optionally subsample rows."""
    high, low = data_utils.load_vary_pos_pairs(dataset_root, n_tokens, vary_pos, model)
    if high is None or low is None:
        raise FileNotFoundError(
            f"no pairs for n={n_tokens}, vary_pos={vary_pos} under {dataset_root}"
        )
    idx = _subsample_indices(len(high["words"]), max_pairs, seed)
    if len(idx) == len(high["words"]):
        return high, low

    def _slice(side: dict[str, Any]) -> dict[str, Any]:
        out = {"words": [side["words"][i] for i in idx]}
        for key in ("ids", "cos_sims", "canon_ids", "canon_embeds", "canon_as_tokens"):
            if key in side:
                val = side[key]
                out[key] = val[idx] if isinstance(val, torch.Tensor) else [val[i] for i in idx]
        return out

    return _slice(high), _slice(low)


def _canon_activations(model, high: dict[str, Any], readout_layer: int | None = None) -> torch.Tensor:
    if readout_layer is None:
        readout_layer = get_canonicity_layer_idx(model)
    return ap.cache_metric_activation(
        model, high["canon_as_tokens"], metric_layer=readout_layer, metric_pos=-1
    )


def gap_closed_layer_sweep(
    model,
    high: dict[str, Any],
    low: dict[str, Any],
    *,
    act_name: str,
    patch_pos: int,
    layers: Sequence[int],
    metric_pos: int = -1,
) -> dict[str, Any]:
    """Layer sweep with gap-closed % vs source canonical activations."""
    readout_layer = get_canonicity_layer_idx(model)
    canon = _canon_activations(model, high, readout_layer)
    lo, hi = ap.baseline_lo_hi(
        model, low["ids"], high["ids"], canon, "resid_post", readout_layer, metric_pos
    )
    scores = ap.run_patching(
        model,
        target_ids=low["ids"],
        source_ids=high["ids"],
        canon_activations=canon,
        patch_pos=patch_pos,
        act_name=act_name,
        layers_to_patch=list(layers),
        metric_pos=metric_pos,
    )
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    pct = metrics.gap_closed_pct(scores, lo, hi)
    pct_list = [float(x) for x in np.asarray(pct).ravel()]
    layer_list = list(layers)
    return {
        "layers": layer_list,
        "scores": [float(x) for x in np.asarray(scores).ravel()],
        "gap_closed_pct": pct_list,
        "lo": lo,
        "hi": hi,
        "peak_layer": layer_list[int(np.nanargmax(pct_list))],
        "peak_pct": float(np.nanmax(pct_list)),
    }


def collect_flat_splits(
    model,
    words: Sequence[str],
    n_tokens: int = 2,
    *,
    max_splits: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """All valid k-piece segmentations with canonicity scores."""
    splits = create_datasets.collect_splits(model, list(words), n_tokens)
    readout = get_canonicity_layer_idx(model)
    cos_sims = create_datasets.cos_sim_split_vs_canonical(model, splits, readout)
    for split, cos in zip(splits, cos_sims):
        split["cos_sim"] = cos
    if max_splits is not None and len(splits) > max_splits:
        idx = _subsample_indices(len(splits), max_splits, seed)
        splits = [splits[i] for i in idx]
    return splits


def ensure_token_count_datasets(
    model,
    words: Sequence[str],
    token_counts: Sequence[int],
    dataset_root: str | Path,
    *,
    min_pair_cos_diff: float = 0.2,
) -> list[str]:
    """Build and save pair datasets for missing token counts; return written paths."""
    tag = data_utils.dataset_tag(model)
    missing = [
        n for n in token_counts
        if not data_utils.vary_pos_pairs_path(dataset_root, n, 1, tag).exists()
    ]
    if not missing:
        return []
    datasets = create_datasets.create_datasets(
        model, list(words), token_counts=missing, min_pair_cos_diff=min_pair_cos_diff
    )
    return create_datasets.save_datasets(datasets, dataset_root, model)


# ---------------------------------------------------------------------------
# Table 1 / Appendix A
# ---------------------------------------------------------------------------


def table1_pool_counts(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    raw_list_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    pool = build_single_token_pool(model, raw_list_paths)
    high_lst, _ = data_utils.load_vary_pos_pairs(dataset_root, 2, 1, model)
    high_fst, _ = data_utils.load_vary_pos_pairs(dataset_root, 2, 2, model)
    lst_n = len(high_lst["words"]) if high_lst else 0
    fst_n = len(high_fst["words"]) if high_fst else 0
    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = []
    if "pool_size" in targets:
        checks.append(check_target("pool_size", len(pool), *targets["pool_size"]))
    if "lst_pairs" in targets:
        checks.append(check_target("lst_pairs", lst_n, *targets["lst_pairs"]))
    if "fst_pairs" in targets:
        checks.append(check_target("fst_pairs", fst_n, *targets["fst_pairs"]))
    return {
        "pool_size": len(pool),
        "lst_pairs": lst_n,
        "fst_pairs": fst_n,
        **summarize_checks(checks),
    }


# ---------------------------------------------------------------------------
# Appendix B: readout layer & quintiles (Fig 7, Fig 8)
# ---------------------------------------------------------------------------


def readout_layer_sweep(
    model,
    words: Sequence[str] | None = None,
    *,
    n_tokens: int = 2,
    max_splits: int | None = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if words is None:
        words = build_single_token_pool(model)
    splits = collect_flat_splits(model, words, n_tokens, max_splits=max_splits, seed=seed)
    readout = get_canonicity_layer_idx(model)
    n_layers = model.cfg.n_layers
    canon_pos = len(token_utils.leading_bos(model)[0])

    split_ids = torch.tensor([s["tokens_ids"] for s in splits], device=_device(model))
    canon_ids = torch.tensor([s["canonical_ids"] for s in splits], device=_device(model))
    last_pos = split_ids.shape[1] - 1

    canon_cos_by_layer: list[np.ndarray] = []
    kl_by_layer: list[np.ndarray] = []
    top1_by_layer: list[np.ndarray] = []
    top5_by_layer: list[np.ndarray] = []

    with torch.no_grad():
        for layer in tqdm.tqdm(range(n_layers), desc="readout sweep"):
            split_act = activation_at_position_batch(model, split_ids, layer, pos=last_pos)
            canon_act = activation_at_position_batch(model, canon_ids, layer, pos=canon_pos)
            canon_cos_by_layer.append(metrics.row_cosine(canon_act, split_act))

        split_logits = next_token_logits_batch(model, split_ids, pos=last_pos)
        canon_logits = next_token_logits_batch(model, canon_ids, pos=canon_pos)
        kl_base = metrics.next_token_kl(canon_logits, split_logits)
        top1_base = metrics.top1_agreement(canon_logits, split_logits)
        top5_base = metrics.topk_overlap(canon_logits, split_logits, k=5)

    rho_kl, rho_top1, rho_top5 = [], [], []
    for layer in range(n_layers):
        cos = canon_cos_by_layer[layer]
        rho_kl.append(float(spearmanr(cos, -kl_base).correlation or 0.0))
        rho_top1.append(float(spearmanr(cos, top1_base).correlation or 0.0))
        rho_top5.append(float(spearmanr(cos, top5_base).correlation or 0.0))

    best_kl_layer = int(np.argmax(rho_kl))
    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    rt = targets.get("readout_layer", {})
    checks = []
    if rt:
        checks.extend([
            check_target("rho_kl_at_n-2", rho_kl[readout], rt["rho_kl"][0], rt["rho_kl"][1]),
            check_target("rho_top1_at_n-2", rho_top1[readout], rt["rho_top1"][0], rt["rho_top1"][1]),
            check_target("rho_top5_at_n-2", rho_top5[readout], rt["rho_top5"][0], rt["rho_top5"][1]),
        ])
    return {
        "readout_layer": readout,
        "best_kl_layer": best_kl_layer,
        "rho_kl": rho_kl,
        "rho_top1": rho_top1,
        "rho_top5": rho_top5,
        "canonicity_by_layer": [float(np.mean(c)) for c in canon_cos_by_layer],
        **summarize_checks(checks),
    }


def quintile_behavioral(
    model,
    words: Sequence[str] | None = None,
    *,
    n_tokens: int = 2,
    max_splits: int | None = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if words is None:
        words = build_single_token_pool(model)
    splits = collect_flat_splits(model, words, n_tokens, max_splits=max_splits, seed=seed)
    readout = get_canonicity_layer_idx(model)
    canon_pos = len(token_utils.leading_bos(model)[0])
    split_ids = torch.tensor([s["tokens_ids"] for s in splits], device=_device(model))
    canon_ids = torch.tensor([s["canonical_ids"] for s in splits], device=_device(model))
    last_pos = split_ids.shape[1] - 1

    with torch.no_grad():
        split_act = activation_at_position_batch(model, split_ids, readout, pos=last_pos)
        canon_act = activation_at_position_batch(model, canon_ids, readout, pos=canon_pos)
        cos = metrics.row_cosine(canon_act, split_act)
        split_logits = next_token_logits_batch(model, split_ids, pos=last_pos)
        canon_logits = next_token_logits_batch(model, canon_ids, pos=canon_pos)
        top1 = metrics.top1_agreement(canon_logits, split_logits)
        kl = metrics.next_token_kl(canon_logits, split_logits)

    labels, lo_thr, hi_thr = probing.quintile_labels(cos)
    top_mask = labels == 1
    bot_mask = labels == -1
    result = {
        "top_quintile": {
            "mean_canonicity": float(cos[top_mask].mean()),
            "top1_agreement": float(top1[top_mask].mean()),
            "median_kl": float(np.median(kl[top_mask])),
        },
        "bottom_quintile": {
            "mean_canonicity": float(cos[bot_mask].mean()),
            "top1_agreement": float(top1[bot_mask].mean()),
            "median_kl": float(np.median(kl[bot_mask])),
        },
        "thresholds": (lo_thr, hi_thr),
    }
    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = [
        check_target("top_quintile_top1", result["top_quintile"]["top1_agreement"],
                     targets.get("quintile_top1", (0.78, 0.08))[0],
                     targets.get("quintile_top1", (0.78, 0.08))[1]),
        check_target("bottom_quintile_top1", result["bottom_quintile"]["top1_agreement"],
                     targets.get("quintile_bottom_top1", (0.24, 0.08))[0],
                     targets.get("quintile_bottom_top1", (0.24, 0.08))[1]),
    ]
    return {**result, **summarize_checks(checks)}


# ---------------------------------------------------------------------------
# Fig 2a,b: two-stage layer sweep
# ---------------------------------------------------------------------------


def two_stage_layer_sweep(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = None,
    n_layers: int = 7,
    seed: int = 0,
) -> dict[str, Any]:
    layers = list(range(n_layers))
    patch_pos = 2  # last subword position for k=2

    high_lst, low_lst = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    high_fst, low_fst = load_pair_batch(model, dataset_root, 2, 2, max_pairs=max_pairs, seed=seed)

    lst_attn = gap_closed_layer_sweep(
        model, high_lst, low_lst, act_name="attn_out", patch_pos=patch_pos, layers=layers
    )
    lst_mlp = gap_closed_layer_sweep(
        model, high_lst, low_lst, act_name="mlp_out", patch_pos=patch_pos, layers=layers
    )
    fst_attn = gap_closed_layer_sweep(
        model, high_fst, low_fst, act_name="attn_out", patch_pos=patch_pos, layers=layers
    )
    fst_mlp = gap_closed_layer_sweep(
        model, high_fst, low_fst, act_name="mlp_out", patch_pos=patch_pos, layers=layers
    )

    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = [
        check_target("lst_l1_attn", lst_attn["gap_closed_pct"][1], *targets.get("lst_l1_attn_gap", (53, 8))),
        check_target("lst_l2_mlp_peak", max(lst_mlp["gap_closed_pct"][1:3]), *targets.get("lst_l2_mlp_peak_gap", (28, 8))),
        check_target("fst_l1_mlp", fst_mlp["gap_closed_pct"][1], *targets.get("fst_l1_mlp_gap", (53, 8))),
        check_target("fst_l1_attn", fst_attn["gap_closed_pct"][1], *targets.get("fst_l1_attn_gap", (1, 3))),
    ]
    return {
        "lst_attn": lst_attn,
        "lst_mlp": lst_mlp,
        "fst_attn": fst_attn,
        "fst_mlp": fst_mlp,
        **summarize_checks(checks),
    }


# ---------------------------------------------------------------------------
# Table 2: behavioral patching (extension)
# ---------------------------------------------------------------------------


def _patch_and_capture_logits(
    model,
    target_ids: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    act_name: str,
    layer: int,
    patch_pos: int,
    readout_pos: int = -1,
) -> torch.Tensor:
    """Patch one layer/component and return last-position next-token logits."""
    hook_name = utils.get_act_name(act_name, layer=layer)
    with torch.no_grad():
        _, source_cache = model.run_with_cache(
            source_ids, names_filter=lambda n: n == hook_name, return_type=None
        )
    model.reset_hooks(including_permanent=True)
    model.add_hook(
        hook_name,
        ap._make_source_patch_hook(source_cache[hook_name], [patch_pos]),
        is_permanent=True,
    )
    with torch.no_grad():
        logits = model(target_ids, return_type="logits")[:, readout_pos, :].detach().cpu()
    model.reset_hooks(including_permanent=True)
    return logits


def patch_behavioral(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Behavioral gap-closed for L1 attn_out (LST) and L1 mlp_out (FST)."""
    high_lst, low_lst = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    high_fst, low_fst = load_pair_batch(model, dataset_root, 2, 2, max_pairs=max_pairs, seed=seed)

    def _behavioral_block(high, low, act_name: str, layer: int = 1, patch_pos: int = 2):
        canon_logits = next_token_logits_batch(model, high["canon_as_tokens"], pos=-1)
        failed_logits = next_token_logits_batch(model, low["ids"], pos=-1)
        success_logits = next_token_logits_batch(model, high["ids"], pos=-1)
        patched_logits = _patch_and_capture_logits(
            model, low["ids"], high["ids"], act_name=act_name, layer=layer, patch_pos=patch_pos
        )
        # RANDOM: patch from a cyclically shifted unrelated source in the batch
        perm = torch.roll(high["ids"], shifts=1, dims=0)
        random_logits = _patch_and_capture_logits(
            model, low["ids"], perm, act_name=act_name, layer=layer, patch_pos=patch_pos
        )

        def _rates(logits):
            return {
                "top1": float(metrics.top1_agreement(canon_logits, logits).mean()),
                "top5": float(metrics.topk_overlap(canon_logits, logits, k=5).mean()),
                "kl": float(metrics.next_token_kl(canon_logits, logits).mean()),
            }

        def _gap_closed(patched_val, lo_val, hi_val):
            gap = hi_val - lo_val
            if abs(gap) < 1e-9:
                return float("nan")
            return (patched_val - lo_val) / gap * 100.0

        failed = _rates(failed_logits)
        success = _rates(success_logits)
        patched = _rates(patched_logits)
        random = _rates(random_logits)
        return {
            "failed": failed,
            "success": success,
            "patched": patched,
            "random": random,
            "gap_closed_top1": _gap_closed(patched["top1"], failed["top1"], success["top1"]),
            "gap_closed_random_top1": _gap_closed(random["top1"], failed["top1"], success["top1"]),
        }

    lst = _behavioral_block(high_lst, low_lst, "attn_out")
    fst = _behavioral_block(high_fst, low_fst, "mlp_out")
    checks = [
        check_target("lst_attn_top1_gap_closed", lst["gap_closed_top1"], 42.0, 12.0),
        check_target("lst_attn_random_top1_gap_closed", lst["gap_closed_random_top1"], 1.0, 5.0),
        check_target("fst_mlp_top1_gap_closed", fst["gap_closed_top1"], 40.0, 12.0),
    ]
    return {"lst_attn_out": lst, "fst_mlp_out": fst, **summarize_checks(checks)}


# ---------------------------------------------------------------------------
# Fig 2c + MLP necessity (§3.4)
# ---------------------------------------------------------------------------


def mlp_interventions(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    high, _ = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    readout = get_canonicity_layer_idx(model)
    canon = _canon_activations(model, high, readout)
    patch_pos = 2

    necessity = interventions.scale_component(
        model, high["ids"], canon,
        positions=[patch_pos], layers=list(range(model.cfg.n_layers)), alphas=[0.0],
        readout_pos=-1,
    )
    alphas = [float(x) for x in np.linspace(0, 4, 9)]
    continuity = {}
    for label, positions in [("pos1", [1]), ("pos2", [2]), ("both", [1, 2])]:
        continuity[label] = interventions.scale_component(
            model, high["ids"], canon,
            positions=positions, layers=[1], alphas=alphas, readout_pos=-1,
        )[1]

    unmodified = float(metrics.mean_cosine(
        activation_at_position_batch(model, high["ids"], readout, pos=-1), canon
    ))
    l1_ablated = necessity[1][0.0]
    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = [
        check_target("unmodified_canonicity", unmodified, *targets.get("mlp_unmodified_canon", (0.95, 0.03))),
        check_target("l1_mlp_ablation", l1_ablated, *targets.get("mlp_necessity_l1_canon", (0.78, 0.05))),
    ]
    return {
        "unmodified_canonicity": unmodified,
        "necessity": {layer: vals[0.0] for layer, vals in necessity.items()},
        "continuity": continuity,
        **summarize_checks(checks),
    }


# ---------------------------------------------------------------------------
# §3.3 Per-head patching
# ---------------------------------------------------------------------------


def per_head_patching(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    high, low = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    readout = get_canonicity_layer_idx(model)
    canon = _canon_activations(model, high, readout)
    lo, hi = ap.baseline_lo_hi(model, low["ids"], high["ids"], canon, "resid_post", readout, -1)
    patch_pos = 2

    def _head_gap(heads: list[tuple[int, int]], act_name: str = "z") -> float:
        score = ap.run_patching(
            model, low["ids"], high["ids"], canon,
            patch_pos=patch_pos, act_name=act_name, heads=heads,
        )
        return float(metrics.gap_closed_pct(score, lo, hi))

    results = {
        "z_head27": _head_gap([(1, 27)]),
        "z_heads_24_28": _head_gap([(1, 24), (1, 28)]),
        "z_heads_24_27_28": _head_gap([(1, 24), (1, 27), (1, 28)]),
    }
    # Value vectors at position 1 (all heads jointly at L1)
    v_score = ap.run_patching(
        model, low["ids"], high["ids"], canon,
        patch_pos=1, act_name="v", heads=[(1, h) for h in range(model.cfg.n_heads)],
    )
    results["v_pos1_all_heads"] = float(metrics.gap_closed_pct(v_score, lo, hi))
    for act in ("q", "k"):
        score = ap.run_patching(
            model, low["ids"], high["ids"], canon,
            patch_pos=patch_pos, act_name=act,
            heads=[(1, h) for h in range(model.cfg.n_heads)],
        )
        results[f"{act}_all_heads"] = float(metrics.gap_closed_pct(score, lo, hi))

    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = [
        check_target("z_head27", results["z_head27"], *targets.get("head_z27_gap", (17, 5))),
        check_target("z_combined", results["z_heads_24_27_28"], *targets.get("head_z_combined_gap", (39, 8))),
    ]
    return {**results, **summarize_checks(checks)}


def head_canonical_alignment(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean cosine of dominant L1 head outputs (z @ W_O) to canonical embedding."""
    high, _ = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    dominant = {24, 27, 28}
    hook_name = utils.get_act_name("z", layer=1)
    with torch.no_grad():
        _, cache = model.run_with_cache(high["ids"], names_filter=lambda n: n == hook_name)
        z = cache[hook_name][:, 1, :, :]  # pos 1, all heads
        w_o = model.W_O[1]
        head_out = torch.einsum("bhd,hdv->bhv", z, w_o)
        canon_embed = high["canon_embeds"].to(head_out.device)
        cos_all = []
        for h in range(model.cfg.n_heads):
            cos_h = torch.nn.functional.cosine_similarity(head_out[:, h, :], canon_embed, dim=-1)
            cos_all.append(float(cos_h.mean()))
    dom_mean = float(np.mean([cos_all[h] for h in dominant]))
    other_mean = float(np.mean([cos_all[h] for h in range(model.cfg.n_heads) if h not in dominant]))
    return {"per_head_cosine": cos_all, "dominant_mean": dom_mean, "other_mean": other_mean}


# ---------------------------------------------------------------------------
# Fig 3: token-count scaling
# ---------------------------------------------------------------------------


def token_count_scaling(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    words: Sequence[str] | None = None,
    *,
    token_counts: Sequence[int] = (2, 3, 4, 5, 6),
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if words is None:
        words = build_single_token_pool(model)
    ensure_token_count_datasets(model, words, token_counts, dataset_root)
    layers = list(range(min(13, model.cfg.n_layers)))
    curves: dict[int, dict[str, Any]] = {}
    for k in token_counts:
        high, low = load_pair_batch(model, dataset_root, k, 1, max_pairs=max_pairs, seed=seed)
        sweep = gap_closed_layer_sweep(
            model, high, low, act_name="resid_post", patch_pos=-1, layers=layers, metric_pos=-1
        )
        l80 = metrics.first_layer_at_threshold(sweep["gap_closed_pct"], layers, threshold=80.0)
        curves[k] = {**sweep, "l80_layer": l80}

    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = []
    if 2 in curves and curves[2]["l80_layer"] is not None:
        checks.append(check_target("l80_k2", curves[2]["l80_layer"], *targets.get("scaling_l80_k2", (2, 1))))
    if 6 in curves and curves[6]["l80_layer"] is not None:
        checks.append(check_target("l80_k6", curves[6]["l80_layer"], *targets.get("scaling_l80_k6", (6, 2))))
    return {"curves": curves, **summarize_checks(checks)}


# ---------------------------------------------------------------------------
# Fig 4 / Fig 9: intermediate relays (reverse patch)
# ---------------------------------------------------------------------------


def intermediate_relays(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    token_counts: Sequence[int] = (3, 4, 5),
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Corrupt successful runs by patching failed intermediate resid_post (reverse direction)."""
    layers = list(range(min(13, model.cfg.n_layers)))
    out: dict[int, Any] = {}
    for k in token_counts:
        high, low = load_pair_batch(model, dataset_root, k, 1, max_pairs=max_pairs, seed=seed)
        readout = get_canonicity_layer_idx(model)
        canon = _canon_activations(model, high, readout)
        baseline = float(metrics.mean_cosine(
            activation_at_position_batch(model, high["ids"], readout, pos=-1), canon
        ))
        intermediate = list(range(2, k))  # positions 2 .. k-1
        single: dict[int, list[float]] = {}
        for pos in intermediate:
            # reverse: source=failed, target=successful
            scores = ap.run_patching(
                model, high["ids"], low["ids"], canon,
                patch_pos=pos, act_name="resid_post", layers_to_patch=layers, metric_pos=-1,
            )
            retained = [float(s / baseline * 100.0) if baseline > 0 else float("nan") for s in scores]
            single[pos] = retained
        if len(intermediate) > 1:
            joint_scores = ap.run_patching(
                model, high["ids"], low["ids"], canon,
                patch_pos=intermediate, act_name="resid_post", layers_to_patch=layers, metric_pos=-1,
            )
            joint = [float(s / baseline * 100.0) if baseline > 0 else float("nan") for s in joint_scores]
        else:
            joint = single.get(intermediate[0], [])
        out[k] = {"baseline": baseline, "single_position": single, "joint": joint, "layers": layers}
    return out


# ---------------------------------------------------------------------------
# Fig 5 / Table 3 & Fig 10: cross-architecture
# ---------------------------------------------------------------------------


def cross_arch_depth(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    high, low = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    layers = list(range(model.cfg.n_layers))
    sweep = gap_closed_layer_sweep(
        model, high, low, act_name="resid_post", patch_pos=-1, layers=layers, metric_pos=-1
    )
    l1_pct = sweep["gap_closed_pct"][1] if len(sweep["gap_closed_pct"]) > 1 else float("nan")
    l80 = metrics.first_layer_at_threshold(sweep["gap_closed_pct"], layers, threshold=80.0)
    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, {})
    checks = []
    if "l1_resid_gap_pct" in targets:
        checks.append(check_target("l1_resid_gap", l1_pct, *targets["l1_resid_gap_pct"]))
    if "l80_layer" in targets:
        checks.append(check_target("l80_layer", l80, *targets["l80_layer"]))
    return {"l1_gap_closed_pct": l1_pct, "l80_layer": l80, "sweep": sweep, **summarize_checks(checks)}


def cross_arch_two_stage(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = None,
    n_layers: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Attention read at pos 1 (% gap closed) and MLP damage (% toward failed)."""
    high, low = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs, seed=seed)
    readout = get_canonicity_layer_idx(model)
    canon = _canon_activations(model, high, readout)
    lo, hi = ap.baseline_lo_hi(model, low["ids"], high["ids"], canon, "resid_post", readout, -1)
    layers = list(range(min(n_layers, model.cfg.n_layers)))

    attn_scores = ap.run_patching(
        model, low["ids"], high["ids"], canon,
        patch_pos=1, act_name="resid_post", layers_to_patch=layers, metric_pos=-1,
    )
    attn_pct = [float(x) for x in metrics.gap_closed_pct(attn_scores, lo, hi)]

    mlp_damage = []
    for layer in layers:
        score = ap.run_patching(
            model, high["ids"], low["ids"], canon,
            patch_pos=-1, act_name="mlp_out", layers_to_patch=[layer], metric_pos=-1,
        )
        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().numpy()
        score = float(np.asarray(score).ravel()[0])
        damage = (hi - score) / (hi - lo) * 100.0 if abs(hi - lo) > 1e-9 else float("nan")
        mlp_damage.append(float(damage))

    return {"layers": layers, "attention_read_pct": attn_pct, "mlp_damage_pct": mlp_damage}


def run_cross_arch_suite(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    models: Sequence[str] | None = None,
    *,
    max_pairs: int = 500,
    load_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Load each model and run cross_arch_depth + cross_arch_two_stage."""
    from transformer_lens import HookedTransformer

    selected = models or [m[0] for m in ALL_MODELS]
    results: dict[str, Any] = {}
    for pretrained in selected:
        model = HookedTransformer.from_pretrained(pretrained, dtype=load_dtype)
        try:
            results[pretrained] = {
                "depth": cross_arch_depth(model, dataset_root, max_pairs=max_pairs),
                "two_stage": cross_arch_two_stage(model, dataset_root, max_pairs=max_pairs),
            }
        finally:
            del model
            torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# §6 Probe (Fig 6 / Fig 12)
# ---------------------------------------------------------------------------


def headline_probe(
    model,
    words: Sequence[str] | None = None,
    *,
    k: int = 2,
    seed: int = 313,
) -> dict[str, Any]:
    if words is None:
        words = build_single_token_pool(model)
    result = probing.run_probe(model, list(words), k=k, seed=seed)
    key = _model_key(model)
    targets = PAPER_TARGETS.get(key, PAPER_TARGETS.get("llama-2-7b-hf", {}))
    checks = [
        check_target("isolated_auroc", result["isolated"]["auroc"], *targets.get("probe_isolated_auroc", (0.94, 0.04))),
        check_target("in_context_auroc", result["in_context"]["auroc"], *targets.get("probe_in_context_auroc", (0.97, 0.03))),
        check_target("transfer_auroc", result["isolated_to_in_context"]["auroc"], *targets.get("probe_transfer_auroc", (0.91, 0.05))),
    ]
    return {**result, **summarize_checks(checks)}


def probe_layer_sweep(
    model,
    words: Sequence[str] | None = None,
    *,
    k: int = 2,
    seed: int = 313,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Layerwise isolated / in-context / transfer AUROC (Fig 6, Fig 12)."""
    if words is None:
        words = build_single_token_pool(model)
    readout = get_canonicity_layer_idx(model)
    word_entries = probing.build_split_entries(model, list(words), k)
    records = probing.mine_records(model, word_entries, k, seed=seed)

    iso_acts0, iso_y0, iso_groups0, _ = probing.isolated_activations(
        model, word_entries, k, probing.resolve_probe_layer(str(model.cfg.model_name), k), readout
    )
    train_idx, _ = probing.train_test_split(
        np.arange(len(iso_groups0)), train_size=0.80, stratify=iso_y0, random_state=seed
    )
    train_words = set(iso_groups0[train_idx])

    n_layers = model.cfg.n_layers
    layers = list(range(n_layers))
    iso_curve, ctx_curve, transfer_curve = [], [], []

    for layer in tqdm.tqdm(layers, desc="probe layer sweep"):
        iso_acts, iso_y_l, iso_groups_l, _ = probing.isolated_activations(
            model, word_entries, k, layer, readout
        )
        ctx_acts, ctx_y_l, ctx_groups_l, _ = probing.in_context_activations(
            model, records, readout, layer
        )
        iso_auroc = probing.cv_auroc(iso_acts, iso_y_l, iso_groups_l, cv_folds, seed)[0].mean()
        ctx_auroc = probing.cv_auroc(ctx_acts, ctx_y_l, ctx_groups_l, cv_folds, seed)[0].mean()
        in_train = np.array([w in train_words for w in iso_groups_l])
        direction = probing.fit_direction(iso_acts[in_train], iso_y_l[in_train])
        held_out = np.array([w not in train_words for w in ctx_groups_l])
        transfer = probing.score_auroc(ctx_acts[held_out], ctx_y_l[held_out], direction)
        iso_curve.append(float(iso_auroc))
        ctx_curve.append(float(ctx_auroc))
        transfer_curve.append(float(transfer))

    probe_layer = probing.resolve_probe_layer(str(model.cfg.model_name), k)
    return {
        "layers": layers,
        "probe_layer_lstar": probe_layer,
        "isolated_auroc": iso_curve,
        "in_context_auroc": ctx_curve,
        "transfer_auroc": transfer_curve,
    }


# ---------------------------------------------------------------------------
# README validation
# ---------------------------------------------------------------------------

README_CORRECTIONS: list[str] = [
    "README quick-start references `words/single_token_words.txt`, which is not shipped; "
    "use `reproduce.build_single_token_pool(model)` or the three lists under `words/`.",
    "README does not document `build_single_token_pool`; pair dataset construction assumes "
    "a pre-filtered single-token word list.",
    "Default `dataset_root` for reproduction is `../datasets` (sibling of the repo), not "
    "a `datasets/` folder inside the repo.",
]


def validate_readme_snippet1(
    model,
    words: Sequence[str] | None = None,
    *,
    temp_root: str | Path | None = None,
    token_counts: Sequence[int] = (2,),
    sample_words: int = 200,
) -> dict[str, Any]:
    """Run README snippet 1 on a small pool; verify save/load round-trip."""
    import tempfile

    if words is None:
        pool = build_single_token_pool(model)
        words = pool[:sample_words]
    root = Path(temp_root) if temp_root else Path(tempfile.mkdtemp(prefix="word_factory_"))
    datasets = create_datasets.create_datasets(model, list(words), token_counts=token_counts)
    paths = create_datasets.save_datasets(datasets, root, model)
    high, low = data_utils.load_vary_pos_pairs(root, token_counts[0], 1, model)
    ok = high is not None and low is not None and len(high["words"]) > 0
    return {"saved_paths": paths, "round_trip_ok": ok, "n_pairs": len(high["words"]) if high else 0}


def validate_readme_snippet2(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    max_pairs: int | None = 500,
) -> dict[str, Any]:
    """Run README snippet 2 (LST attn_out layer sweep)."""
    high, low = load_pair_batch(model, dataset_root, 2, 1, max_pairs=max_pairs)
    layer = get_canonicity_layer_idx(model)
    canon = activation_at_position_batch(model, high["canon_as_tokens"], layer, pos=-1)
    scores = ap.run_patching(
        model, low["ids"], high["ids"], canon,
        patch_pos=2, act_name="attn_out", layers_to_patch=range(7),
    )
    lo, hi = ap.baseline_lo_hi(model, low["ids"], high["ids"], canon, "resid_post", layer, -1)
    pct = metrics.gap_closed_pct(scores, lo, hi)
    l1 = float(np.asarray(pct).ravel()[1])
    return {"gap_closed_l1_attn": l1, "pass": abs(l1 - 53.0) <= 8.0}


def validate_readme_snippet3(
    model,
    words: Sequence[str] | None = None,
    *,
    k: int = 2,
    sample_words: int = 500,
) -> dict[str, Any]:
    """Run README snippet 3 (headline probe)."""
    if words is None:
        words = build_single_token_pool(model)[:sample_words]
    result = probing.run_probe(model, list(words), k=k)
    iso = result["isolated"]["auroc"]
    ctx = result["in_context"]["auroc"]
    return {
        "isolated_auroc": iso,
        "in_context_auroc": ctx,
        "pass": abs(iso - 0.94) <= 0.06 and abs(ctx - 0.97) <= 0.05,
    }


def validate_all_readme_snippets(
    model,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    return {
        "snippet1": validate_readme_snippet1(model),
        "snippet2": validate_readme_snippet2(model, dataset_root),
        "snippet3": validate_readme_snippet3(model),
        "corrections": README_CORRECTIONS,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (optional; requires matplotlib)
# ---------------------------------------------------------------------------


def plot_gap_closed_sweep(sweep: dict[str, Any], title: str = "", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sweep["layers"], sweep["gap_closed_pct"])
    ax.set_xlabel("Layer patched")
    ax.set_ylabel("Gap closed (%)")
    ax.set_title(title)
    return ax


def plot_probe_curves(sweep: dict[str, Any], ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    layers = sweep["layers"]
    ax.plot(layers, sweep["isolated_auroc"], label="Isolated")
    ax.plot(layers, sweep["in_context_auroc"], label="In-context")
    ax.plot(layers, sweep["transfer_auroc"], label="Isolated→In-context", linestyle="--")
    ax.axvline(sweep["probe_layer_lstar"], color="gray", linestyle=":", label="l*")
    ax.axhline(0.5, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.legend()
    return ax
