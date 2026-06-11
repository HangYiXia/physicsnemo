"""
Unit tests for amr_m4gn/amr_router.py (M3).

Four routing cases (Design Doc 7.4.3):
    (1) all phys = 0          -> everything folded   -> T == K0
    (2) all phys huge         -> everything kept     -> T == K1
    (3) half active           -> K0 < T < K1, active tokens come from L1
    (4) threshold sampling in-range and reproducible (fixed seed)

Synthetic nested partition (faithful to the real K0=64 / K1=256):
    N = 256 nodes, L1_assign = [0..255] (one node per fine segment),
    L0_assign = [k // 4]  -> each L0 parent owns exactly 4 L1 children
    => K0 = 64, K1 = 256.

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_amr_router.py -v
    # or:  python tests/test_amr_router.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_m4gn.amr_router import (
    aggregate_per_segment,
    sample_thresholds,
    build_l1_to_l0,
    route,
    DEFAULT_RANGES,
)


def _nested_levels(n_per_l0=4, k0=64):
    """Build nested [L0_assign, L1_assign]; K1 = k0 * n_per_l0, one node per L1."""
    k1 = k0 * n_per_l0
    l1 = torch.arange(k1, dtype=torch.long)
    l0 = (l1 // n_per_l0).to(torch.long)
    return [l0, l1], k0, k1


def _zero_phys(n):
    z = torch.zeros(n)
    return {"G": z.clone(), "omega": z.clone(), "M": z.clone(), "S": z.clone()}


def _const_phys(n, val):
    v = torch.full((n,), float(val))
    return {"G": v.clone(), "omega": v.clone(), "M": v.clone(), "S": v.clone()}


# Thresholds well inside DEFAULT_RANGES, used as a fixed test threshold.
_FIXED_T = {"G": 1.0, "omega": 2.0, "M": 5.0, "S": 2.0}


def test_all_calm_folds_to_K0():
    levels, k0, k1 = _nested_levels()
    n = k1
    phys = _zero_phys(n)
    kept_assign, kept_depth, T, token_batch = route(levels, phys, _FIXED_T)
    assert T == k0, f"expected T==K0={k0}, got {T}"
    assert kept_depth.tolist() == [0] * k0  # all folded (coarse)
    assert int(kept_assign.max()) + 1 == T
    assert token_batch.shape[0] == T


def test_all_active_keeps_K1():
    levels, k0, k1 = _nested_levels()
    n = k1
    phys = _const_phys(n, 1e3)  # far above every threshold
    kept_assign, kept_depth, T, _ = route(levels, phys, _FIXED_T)
    assert T == k1, f"expected T==K1={k1}, got {T}"
    assert kept_depth.tolist() == [1] * k1  # all kept (fine)


def test_half_active_intermediate():
    levels, k0, k1 = _nested_levels()
    n = k1
    # First half of fine segments active (huge), second half calm (zero).
    g = torch.zeros(n)
    g[: n // 2] = 1e3
    phys = {"G": g, "omega": torch.zeros(n), "M": torch.zeros(n), "S": torch.zeros(n)}
    kept_assign, kept_depth, T, _ = route(levels, phys, _FIXED_T)

    assert k0 < T < k1, f"expected {k0} < T < {k1}, got {T}"
    # active half: 128 fine tokens; calm half: 128 nodes in L0 parents 32..63,
    # 4 calm children each -> 32 folded tokens. T == 128 + 32 == 160.
    assert T == (n // 2) + (k0 // 2)
    n_fine = int((kept_depth == 1).sum())
    n_coarse = int((kept_depth == 0).sum())
    assert n_fine == n // 2          # active tokens are fine (from L1)
    assert n_coarse == k0 // 2


def test_threshold_sampling_inrange_and_reproducible():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    t1 = sample_thresholds(training=True, generator=g1)
    t2 = sample_thresholds(training=True, generator=g2)
    for k, (lo, hi) in DEFAULT_RANGES.items():
        assert lo <= t1[k] <= hi, f"{k}={t1[k]} out of [{lo},{hi}]"
        assert abs(t1[k] - t2[k]) < 1e-12, f"{k} not reproducible"
    # test-time default = range midpoints
    tt = sample_thresholds(training=False)
    for k, (lo, hi) in DEFAULT_RANGES.items():
        assert abs(tt[k] - 0.5 * (lo + hi)) < 1e-12


def test_aggregate_max_takes_abs():
    levels, k0, k1 = _nested_levels()
    n = k1
    omega = torch.zeros(n)
    omega[0] = -50.0  # negative vorticity, |.| should win
    phys = {"omega": omega}
    agg = aggregate_per_segment(phys, levels[1], k1)
    assert abs(float(agg["omega"][0]) - 50.0) < 1e-6


def test_l1_to_l0_parent_map():
    levels, k0, k1 = _nested_levels()
    l1_to_l0 = build_l1_to_l0(levels)
    assert l1_to_l0.shape[0] == k1
    # fine segments 0,1,2,3 -> parent 0 ; 4,5,6,7 -> parent 1 ; ...
    assert l1_to_l0[0:4].tolist() == [0, 0, 0, 0]
    assert l1_to_l0[4:8].tolist() == [1, 1, 1, 1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
