"""Single-pass HookedTransformer activation / logit extraction at chosen positions."""

from __future__ import annotations

from typing import Sequence

import torch
from transformer_lens import utils


def get_canonicity_layer_idx(model) -> int:
    """The readout layer n-2 (Llama2-7B: layer 30 of 32)."""
    n_layers = getattr(model.cfg, "n_layers", None)
    if n_layers is None:
        raise ValueError("model.cfg has no n_layers")
    layer_idx = int(n_layers) - 2
    if layer_idx <= 0:
        raise ValueError(f"n_layers - 2 must be > 0, got {layer_idx} (n_layers={n_layers})")
    return layer_idx


@torch.inference_mode()
def next_token_logits_batch(model, token_batch, pos: int = -1) -> torch.Tensor:
    """One forward; [B, V] logits at `pos` (for the behavioral metrics in `metrics`)."""
    return model(token_batch, return_type="logits")[:, pos, :].detach().cpu()


@torch.inference_mode()
def activation_at_position_batch(
    model,
    token_batch: torch.Tensor,
    layer: int,
    pos: int | Sequence[int] = -1,
    act_name: str = "resid_post",
) -> torch.Tensor:
    """One forward; [B, d] activation of `act_name` at layer `layer`.

    `pos` is either a single column applied to every row (int, negative ok) or one
    position per row (sequence of length B). Per-row positions are used as given, so
    callers that right-pad must pass already-resolved (non-negative) positions.
    """
    hook_name = utils.get_act_name(act_name, layer=layer)
    _, cache = model.run_with_cache(
        token_batch, names_filter=lambda n: n == hook_name, return_type=None
    )
    acts = cache[hook_name]

    if isinstance(pos, int):
        out = acts[:, pos, :]
    else:
        if len(pos) != acts.shape[0]:
            raise ValueError("len(pos) must match batch size")
        rows = torch.arange(acts.shape[0], device=acts.device)
        cols = torch.as_tensor(pos, dtype=torch.long, device=acts.device)
        out = acts[rows, cols, :]

    out = out.detach().cpu()
    del cache
    return out