"""
Unit tests for amr_m4gn/physics_ops.py (M2).

Strategy: build a regular triangulated grid with ANALYTIC velocity fields whose
gradients are known in closed form, then check G / omega / S against truth.
For a *linear* field the 1-ring least-squares gradient is exact (up to float
round-off) at every node, so tolerances are tight.

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_physics_ops.py -v
    # or:  python tests/test_physics_ops.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_m4gn.physics_ops import (
    compute_ns_quantities,
    lstsq_gradient,
    virtual_step,
    denormalize_velocity,
)


def _grid_mesh(n=20, lo=0.0, hi=1.0):
    """Regular n x n grid on [lo, hi]^2, triangulated (2 tris per cell).

    Returns pos [N,2], edge_index [2,E] (undirected, deduplicated).
    """
    xs = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    pos = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)

    def vid(i, j):
        return i * n + j

    edges = set()
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            # two triangles: (a,b,d) and (a,d,c)
            for (p, q) in [(a, b), (b, d), (a, d), (d, c), (a, c)]:
                edges.add((p, q))
    src, dst = [], []
    for (p, q) in edges:
        src += [p, q]
        dst += [q, p]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return torch.tensor(pos, dtype=torch.float32), edge_index


def _interior_mask(pos, lo=0.0, hi=1.0, margin=1e-6):
    p = pos.numpy()
    return (
        (p[:, 0] > lo + margin) & (p[:, 0] < hi - margin)
        & (p[:, 1] > lo + margin) & (p[:, 1] < hi - margin)
    )


def test_linear_gradient_exact():
    """Linear field f = 3x + 2y  ->  grad = (3, 2) everywhere (LSQ exact)."""
    pos, ei = _grid_mesh(15)
    f = 3.0 * pos[:, 0] + 2.0 * pos[:, 1]
    grad = lstsq_gradient(f, pos, ei)  # [N,1,2]
    err = (grad[:, 0, :] - torch.tensor([3.0, 2.0])).abs().max().item()
    assert err < 1e-3, f"linear gradient error too large: {err}"


def test_shear_flow_vorticity():
    """u = y, v = 0  ->  omega = -1 ; G = 1 ; strain-mag S = 1.

    Simple shear = rotation(omega=-1) + strain(S=1); omega and S now DIFFER,
    confirming the strain magnitude is independent of vorticity.
    """
    pos, ei = _grid_mesh(20)
    u, v = pos[:, 1], torch.zeros(pos.shape[0])
    q = compute_ns_quantities(u, v, pos, ei)
    m = _interior_mask(pos)
    assert (q["omega"][m] - (-1.0)).abs().max() < 5e-2
    assert (q["S"][m] - (1.0)).abs().max() < 5e-2
    assert (q["G"][m] - 1.0).abs().max() < 5e-2


def test_rotation_vorticity():
    """u = -y, v = x  ->  omega = 2 ; strain-mag S = 0 (pure rotation)."""
    pos, ei = _grid_mesh(20)
    u, v = -pos[:, 1], pos[:, 0]
    q = compute_ns_quantities(u, v, pos, ei)
    m = _interior_mask(pos)
    assert (q["omega"][m] - 2.0).abs().max() < 5e-2
    assert q["S"][m].abs().max() < 5e-2   # rigid rotation has zero strain


def test_uniform_flow_zero_gradient():
    """u = 1, v = 0 (constant)  ->  G ~ 0, omega ~ 0, S ~ 0."""
    pos, ei = _grid_mesh(20)
    u, v = torch.ones(pos.shape[0]), torch.zeros(pos.shape[0])
    q = compute_ns_quantities(u, v, pos, ei)
    assert q["G"].abs().max() < 1e-3
    assert q["omega"].abs().max() < 1e-3
    assert q["S"].abs().max() < 1e-3


def test_momentum_indicator():
    """M = rho |U| area. With area=1, rho=1, u=3,v=4 -> M = 5 everywhere."""
    pos, ei = _grid_mesh(10)
    n = pos.shape[0]
    u, v = 3.0 * torch.ones(n), 4.0 * torch.ones(n)
    q = compute_ns_quantities(u, v, pos, ei, area=torch.ones(n), rho=1.0)
    assert (q["M"] - 5.0).abs().max() < 1e-4


def test_denormalize_velocity():
    """phys = norm * std + mean."""
    u = torch.tensor([0.0, 1.0, -1.0])
    v = torch.tensor([0.0, 1.0, -1.0])
    up, vp = denormalize_velocity(u, v, vel_mean=[2.0, -3.0], vel_std=[10.0, 5.0])
    assert torch.allclose(up, torch.tensor([2.0, 12.0, -8.0]))
    assert torch.allclose(vp, torch.tensor([-3.0, 2.0, -8.0]))


def test_virtual_step():
    """uv' = uv_t + (uv_t - uv_prev); None prev -> unchanged."""
    uv_t = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    uv_prev = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = virtual_step(uv_t, uv_prev)
    assert torch.allclose(out, torch.tensor([[2.0, 1.0], [3.0, 0.0]]))
    assert torch.allclose(virtual_step(uv_t, None), uv_t)


def test_no_nan_tiny_graph():
    """Regularization keeps a 3-node degenerate graph NaN-free."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ei = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    q = compute_ns_quantities(pos[:, 0], pos[:, 1], pos, ei)
    for k in q:
        assert torch.isfinite(q[k]).all(), f"{k} has non-finite values"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
