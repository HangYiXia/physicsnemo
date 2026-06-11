"""
amr_router.py — Adaptive Mesh Refinement Token Router (M3)
==========================================================
Turns the AMR-Transformer "progressive quadtree subdivision" into a two-level
fold/keep decision on the hierarchical METIS tree built in M1/M2:

    levels = [L0_assign (K0 coarse segments), L1_assign (K1 fine segments)]

The L1 partition is a *nested* refinement of L0 (each L1 segment lives inside
exactly one L0 segment), so every L1 segment has a unique L0 parent.

Routing (Design Doc 4.7 / 7.4.3):
    Step 1  aggregate per L1 segment: agg_X[k] = max_{i in seg k} |X_i|
    Step 2  active[k] = (agg_G>T_G) | (agg_omega>T_omega) | (agg_M>T_M) | (agg_S>T_S)
    Step 3  active L1 segment  -> kept as its own (fine, depth=1) token
            calm   L1 segment  -> folded back into its L0 parent (depth=0);
                                  all calm siblings of one L0 parent share 1 token

Output: kept_assign[N] in [0, T), kept_depth[T] in {0,1}, T (64<=T<=256),
        token_batch[T] (graph id per token; single graph in M3 -> all zeros).

Implementation uses only core torch (scatter_reduce_, advanced indexing); no
torch_scatter dependency (consistent with physics_ops.py).
"""

from __future__ import annotations

import torch
from torch import Tensor


# Threshold sampling ranges. NOTE on scale (decision gate D3): the paper's
# ranges are in NORMALIZED units; this dataset's physical |omega| is O(1e2).
# `omega` was CALIBRATED on real data (calibrate_thresholds.py, M5 step 5:
# per-seg |omega| p40~p85 over 4 cases) to the physical range below, giving
# token counts T in a healthy [K0,K1] mid-band. G/M/S still hold the paper's
# normalized ranges and should be calibrated the same way before use.
DEFAULT_RANGES = {
    "G": (0.1, 2.0),        # TODO(D3): calibrate to physical scale
    "omega": (2.83, 25.8),  # calibrated (physical), M5 step 5
    "M": (0.5, 10.0),       # TODO(D3): calibrate to physical scale
    "S": (0.2, 4.0),        # TODO(D3): calibrate to physical scale
}

_KEYS = ("G", "omega", "M", "S")


def aggregate_per_segment(
    phys: dict, assign: Tensor, num_seg: int, reduce: str = "max"
) -> dict:
    """Aggregate per-node physical quantities to per-segment scalars.

    phys   : dict of [N] tensors (G, omega, M, S). |.| is taken first, since
             omega has a sign while the router thresholds on magnitude.
    assign : [N] long, segment id per node in [0, num_seg).
    reduce : "max" (default) or "mean".

    Returns dict of [num_seg] tensors. Empty segments map to 0.
    """
    if reduce not in ("max", "mean"):
        raise ValueError(f"unsupported reduce: {reduce}")

    out = {}
    for key, val in phys.items():
        a = val.abs()
        agg = torch.zeros(num_seg, dtype=a.dtype, device=a.device)
        if reduce == "max":
            agg.scatter_reduce_(0, assign, a, reduce="amax", include_self=False)
        else:  # mean
            agg.scatter_reduce_(0, assign, a, reduce="mean", include_self=False)
        out[key] = agg
    return out


def sample_thresholds(
    ranges: dict | None = None,
    training: bool = True,
    fixed: dict | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Sample subdivision thresholds.

    training=True : each threshold ~ Uniform[lo, hi] (paper's mechanism: the
                    model sees every granularity -> robust to test-time tuning).
                    Pass a seeded `generator` for reproducibility.
    training=False: return `fixed` if given, else the midpoint of each range.
    """
    ranges = ranges if ranges is not None else DEFAULT_RANGES

    if not training:
        if fixed is not None:
            return dict(fixed)
        return {k: 0.5 * (lo + hi) for k, (lo, hi) in ranges.items()}

    out = {}
    for k, (lo, hi) in ranges.items():
        r = torch.rand((), generator=generator).item()
        out[k] = lo + r * (hi - lo)
    return out


def build_l1_to_l0(levels: list) -> Tensor:
    """Map each L1 segment to its L0 parent: l1_to_l0[k] in [0, K0).

    Relies on the nested property (all nodes of one L1 segment share an L0
    label), so a single scatter assignment is well-defined.
    """
    L0, L1 = levels[0], levels[1]
    K1 = int(L1.max().item()) + 1
    l1_to_l0 = torch.full((K1,), -1, dtype=torch.long, device=L1.device)
    l1_to_l0[L1] = L0
    return l1_to_l0


def route(levels: list, phys: dict, thresholds: dict, reduce: str = "max"):
    """Fold/keep routing over the two-level partition.

    levels     : [L0_assign[N], L1_assign[N]] (long tensors, nested).
    phys       : dict of [N] tensors {G, omega, M, S}.
    thresholds : dict of scalars per key (from `sample_thresholds`).

    Returns
    -------
    kept_assign : [N] long, final token id per node in [0, T).
    kept_depth  : [T] long, 1 if the token is a kept fine L1 segment,
                  0 if it is a folded-back coarse L0 parent.
    T           : int, number of tokens (K0 <= T <= K1).
    token_batch : [T] long, graph id per token (single graph -> all zeros).
    """
    L0, L1 = levels[0], levels[1]
    device = L1.device
    K1 = int(L1.max().item()) + 1
    K0 = int(L0.max().item()) + 1

    agg = aggregate_per_segment(phys, L1, K1, reduce=reduce)

    active = torch.zeros(K1, dtype=torch.bool, device=device)
    for key in _KEYS:
        if key in agg and key in thresholds:
            active |= agg[key] > thresholds[key]

    l1_to_l0 = build_l1_to_l0(levels)

    token_of_l1 = torch.full((K1,), -1, dtype=torch.long, device=device)
    fold_token_of_l0 = torch.full((K0,), -1, dtype=torch.long, device=device)
    depths = []
    next_id = 0
    for k in range(K1):
        if bool(active[k]):
            token_of_l1[k] = next_id
            depths.append(1)
            next_id += 1
        else:
            p = int(l1_to_l0[k].item())
            if int(fold_token_of_l0[p]) == -1:
                fold_token_of_l0[p] = next_id
                depths.append(0)
                next_id += 1
            token_of_l1[k] = fold_token_of_l0[p]

    T = next_id
    kept_assign = token_of_l1[L1]
    kept_depth = torch.tensor(depths, dtype=torch.long, device=device)
    token_batch = torch.zeros(T, dtype=torch.long, device=device)
    return kept_assign, kept_depth, T, token_batch
