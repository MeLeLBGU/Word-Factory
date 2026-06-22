"""Single-run interventions for the detokenization experiments (paper §3.4).

Unlike activation patching, which transfers activations between a successful and a
failed run, `scale_component` modifies one run in place and reads out raw canonicity.
Scaling a component at given positions by a factor recovers both §3.4 experiments:

    necessity   (zero-ablation)  -> alphas=[0.0], sweep `layers`
    continuity  (alpha-scaling)  -> layers=[1],   sweep `alphas` in [0, 4]

alpha=0 zeroes the component; alpha=1 is the unmodified baseline.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformer_lens import utils

from activations import get_canonicity_layer_idx
from metrics import mean_cosine

DEFAULT_BATCH_SIZE = 2048


def _scale_component_hook(positions: Sequence[int], factor: float):
    """Hook that multiplies the component at `positions` by `factor` (0.0 = ablate)."""
    positions = list(positions)

    def hook(activation, hook):
        patched = activation.clone()
        for pos in positions:
            patched[:, pos, :] = activation[:, pos, :] * factor
        return patched

    return hook


@torch.inference_mode()
def _readout_with_hook(
    model,
    token_ids: torch.Tensor,
    hook_name: str,
    hook_fn,
    readout_layer: int,
    readout_pos: int,
    batch_size: int,
) -> torch.Tensor:
    """Run `token_ids` with `hook_fn` on `hook_name`; return readout-layer resid_post at
    `readout_pos` as [N, d] on CPU. Batched; the hook is batch-independent."""
    readout_name = utils.get_act_name("resid_post", layer=readout_layer)
    capture: dict = {}

    def capture_hook(activation, hook):
        capture["act"] = activation[:, readout_pos, :].detach().cpu()

    chunks: list[torch.Tensor] = []
    model.reset_hooks(including_permanent=True)
    model.add_hook(hook_name, hook_fn, is_permanent=True)
    model.add_hook(readout_name, capture_hook, is_permanent=True)
    try:
        for start in range(0, token_ids.shape[0], batch_size):
            model(token_ids[start : start + batch_size], return_type=None)
            chunks.append(capture["act"])
    finally:
        model.reset_hooks(including_permanent=True)
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def scale_component(
    model,
    token_ids: torch.Tensor,
    canon_activations: torch.Tensor,
    *,
    positions: Sequence[int],
    layers: Sequence[int],
    alphas: Sequence[float],
    component: str = "mlp_out",
    readout_layer: int | None = None,
    readout_pos: int = -1,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, dict[float, float]]:
    """Canonicity after scaling `component` at `positions` by each alpha, per layer.

    Returns {layer: {alpha: mean canonicity}}. Two §3.4 uses:
      necessity   -> alphas=[0.0], layers=range(model.cfg.n_layers)
      continuity  -> layers=[1],   alphas=[0.0, 0.5, 1.0, ... 4.0]
    """
    if readout_layer is None:
        readout_layer = get_canonicity_layer_idx(model)

    out: dict[int, dict[float, float]] = {}
    for layer in layers:
        hook_name = utils.get_act_name(component, layer=layer)
        out[int(layer)] = {}
        for alpha in alphas:
            readout = _readout_with_hook(
                model, token_ids, hook_name, _scale_component_hook(positions, float(alpha)),
                readout_layer, readout_pos, batch_size,
            )
            out[int(layer)][float(alpha)] = mean_cosine(readout, canon_activations)
    return out