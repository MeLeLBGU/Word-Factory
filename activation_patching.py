"""Activation patching for the detokenization experiments (paper's sections 3, 4, 5).

A single entry point, `run_patching`, patches a chosen component from a source
(successful) run into a target (failed) run, completes the target forward pass, and
scores how close the target's readout-layer residual gets to the source's canonical
representation (canonicity_src). It dispatches to three patching modes:

    layer sweep      one (layer, pos) at a time           (3.1-3.2, 5)
    per head         q/k/v/z vectors or attention pattern (3.3)
    multi position   several positions jointly, per layer (4)
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Sequence

import pandas as pd
import torch
from transformer_lens import utils
from transformer_lens.patching import (
    generic_activation_patch,
    layer_pos_patch_setter,
    layer_pos_head_vector_patch_setter,
    layer_head_pos_pattern_patch_setter,
)

from metrics import mean_cosine, _row_cosine
from activations import activation_at_position_batch, get_canonicity_layer_idx

# Components patched whole at one (layer, pos); vs. per-head vectors; vs. attention patterns.
_POS_COMPONENTS = {"resid_pre", "resid_mid", "resid_post", "mlp_out", "attn_out"}
_HEAD_VECTOR_COMPONENTS = {"q", "k", "v", "z"}


# -- Patch-setter / index selection ------------------------------------


def _patch_setter(act_name: str):
    if act_name in _POS_COMPONENTS:
        return layer_pos_patch_setter
    if act_name in _HEAD_VECTOR_COMPONENTS:
        return layer_pos_head_vector_patch_setter
    if act_name == "pattern":
        return layer_head_pos_pattern_patch_setter
    raise ValueError(f"no patch setter mapped for activation {act_name!r}")


def _layer_index_df(act_name: str, patch_pos: int, layers: Sequence[int], num_heads: int) -> pd.DataFrame:
    """One row per (layer[, head]) at `patch_pos`, in the column order each setter expects."""
    if act_name in _HEAD_VECTOR_COMPONENTS:
        rows = [(layer, patch_pos, head) for layer in layers for head in range(num_heads)]
        return pd.DataFrame(rows, columns=["layer", "pos", "head"])
    if act_name == "pattern":
        rows = [(layer, head, patch_pos) for layer in layers for head in range(num_heads)]
        return pd.DataFrame(rows, columns=["layer", "head", "pos"])
    rows = [(layer, patch_pos) for layer in layers]
    return pd.DataFrame(rows, columns=["layer", "pos"])


# -- Hooks and metric --------------------------------------------------


def _capture_readout_hook(activation, hook, readout_capture: dict, pos: int) -> None:
    """Stash the readout activation at `pos` so the metric can read it after the pass."""
    readout_capture["readout_act"] = activation[:, pos, :].detach().cpu()


def _cosine_to_canon(logits, readout_capture: dict, canon_activations: torch.Tensor) -> torch.Tensor:
    """Mean cosine of the captured readout activation against the canonical one (ignores logits)."""
    return _row_cosine(readout_capture["readout_act"], canon_activations).mean()


def _make_source_patch_hook(source_activation: torch.Tensor, positions: Sequence[int], heads: list[int] | None = None):
    """Hook overwriting `positions` with the source run's activations.

    heads=None patches the whole vector at each position (resid/mlp_out/attn_out);
    a head list patches only those per-head sub-vectors (q/k/v/z).
    """
    positions = list(positions)

    def hook(activation, hook):
        patched = activation.clone()
        for pos in positions:
            if heads is None:
                patched[:, pos, :] = source_activation[:, pos, :]
            else:
                for head in heads:
                    patched[:, pos, head] = source_activation[:, pos, head]
        return patched

    return hook


# -- Source / metric caching -------------------------------------------


def cache_source_activations(model, source_ids, act_name: str, layers: Sequence[int]):
    """Cache `act_name` at the given layers for the source run, for use as patch source."""
    needed = {utils.get_act_name(act_name, layer=layer) for layer in layers}
    with torch.no_grad():
        _, source_cache = model.run_with_cache(
            source_ids, names_filter=lambda n: n in needed, return_type=None
        )
    return source_cache


def cache_metric_activation(
    model,
    ids,
    metric_name: str = "resid_post",
    metric_layer: int | None = None,
    metric_pos: int = -1,
) -> torch.Tensor:
    """[N, d] readout activation at `metric_pos` on any forward (canonical or split)."""
    if metric_layer is None:
        metric_layer = get_canonicity_layer_idx(model)
    return activation_at_position_batch(model, ids, metric_layer, pos=metric_pos, act_name=metric_name)


# -- Patching modes ----------------------------------------------------


def _patch_layerwise(model, target_ids, source_cache, act_name, patch_pos, metric, layers_to_patch):
    """Sweep `act_name` at `patch_pos` across the given layers; one metric value per index row."""
    index_df = _layer_index_df(act_name, patch_pos, layers_to_patch, model.cfg.n_heads)
    with torch.no_grad():
        return generic_activation_patch(
            model,
            target_ids,
            source_cache,
            metric,
            patch_setter=_patch_setter(act_name),
            activation_name=act_name,
            index_df=index_df,
        )


def _patch_heads(model, target_ids, source_cache, heads, patch_pos, metric, act_name):
    """Patch the given (layer, head) vectors at `patch_pos` jointly; return a single metric value."""
    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        heads_by_layer[layer].append(head)

    for layer, layer_heads in heads_by_layer.items():
        hook_name = utils.get_act_name(act_name, layer=layer)
        model.add_hook(
            hook_name,
            _make_source_patch_hook(source_cache[hook_name], [patch_pos], heads=layer_heads),
            is_permanent=True,
        )
    with torch.no_grad():
        logits = model(target_ids, return_type=None)
        return metric(logits).item()  # type: ignore


def _patch_multiple_positions_layerwise(model, target_ids, source_cache, act_name, positions, metric, layers_to_patch):
    """Patch `positions` jointly, one layer at a time; one metric value per layer."""
    scores = []
    for layer in layers_to_patch:
        hook_name = utils.get_act_name(act_name, layer=layer)
        model.add_hook(
            hook_name,
            _make_source_patch_hook(source_cache[hook_name], positions),
            is_permanent=False,
        )
        with torch.no_grad():
            logits = model(target_ids, return_type=None)
        scores.append(metric(logits).item())
        model.reset_hooks(including_permanent=False)
    return scores


# -- Baselines ---------------------------------------------------------


def baseline_lo_hi(model, target_ids, source_ids, canon_activations, metric_name, metric_layer, metric_pos):
    """Endpoints for gap-closed %: cos(target, canon) (lo) and cos(source, canon) (hi)."""
    lo_act = cache_metric_activation(model, target_ids, metric_name, metric_layer, metric_pos)
    hi_act = cache_metric_activation(model, source_ids, metric_name, metric_layer, metric_pos)
    return mean_cosine(lo_act, canon_activations), mean_cosine(hi_act, canon_activations)


# -- Entry point -------------------------------------------------------


def run_patching(
    model,
    target_ids,
    source_ids,
    canon_activations: torch.Tensor,
    patch_pos: int | Sequence[int] = -1,
    *,
    act_name: str = "resid_post",
    layers_to_patch: Sequence[int] | None = None,
    metric_name: str = "resid_post",
    metric_layer: int | None = None,
    metric_pos: int = -1,
    heads: list[tuple[int, int]] | None = None,  # (layer, head) tuples
):
    """Patch `act_name` from source into target and score canonicity_src against
    `canon_activations` at (`metric_name`, `metric_layer`, `metric_pos`).

    `patch_pos` is a single position (layer sweep / per-head modes) or several positions
    (multi-position mode). `heads` selects per-head patching and requires a q/k/v/z
    `act_name`. Returns a per-layer tensor/list of scores, or a single value for `heads`.
    """
    if layers_to_patch is None:
        layers_to_patch = list(range(model.cfg.n_layers))
    if metric_layer is None:
        metric_layer = get_canonicity_layer_idx(model)
    patching_positions = [patch_pos] if isinstance(patch_pos, int) else list(patch_pos)

    metric_capture: dict = {}
    metric = partial(_cosine_to_canon, readout_capture=metric_capture, canon_activations=canon_activations)
    metric_hook_name = utils.get_act_name(metric_name, layer=metric_layer)

    model.reset_hooks(including_permanent=True)
    model.add_hook(
        metric_hook_name,
        partial(_capture_readout_hook, readout_capture=metric_capture, pos=metric_pos),
        is_permanent=True,
    )

    try:
        if heads is not None:
            if act_name not in _HEAD_VECTOR_COMPONENTS:
                raise ValueError(
                    f"head patching needs a per-head component {_HEAD_VECTOR_COMPONENTS}, "
                    f"got act_name={act_name!r}"
                )
            layers_to_patch = [layer for layer, _ in heads]
            source_cache = cache_source_activations(model, source_ids, act_name, layers_to_patch)
            return _patch_heads(model, target_ids, source_cache, heads, patching_positions[0], metric, act_name)

        source_cache = cache_source_activations(model, source_ids, act_name, layers_to_patch)
        if len(patching_positions) > 1:
            return _patch_multiple_positions_layerwise(
                model, target_ids, source_cache, act_name, patching_positions, metric, layers_to_patch
            )
        return _patch_layerwise(model, target_ids, source_cache, act_name, patching_positions[0], metric, layers_to_patch)
    finally:
        model.reset_hooks(including_permanent=True)
        torch.cuda.empty_cache()