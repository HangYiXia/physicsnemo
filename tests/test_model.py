"""
Integration test for amr_m4gn/model.py (M4, Design Doc 7.4.6).

Single-graph single-step forward + backward on a SYNTHETIC mesh and cache
(no dataset / no preprocess dependency):
    (1) forward -> pred [N, 3], all finite (no NaN);
    (2) loss.backward() -> every parameter receives a finite gradient.

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_model.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data

from amr_m4gn.model import AMRM4GN


def _nested_assign(N, k0=8, n_per_l0=4):
    """Assign N nodes to a nested L0/L1 partition (K1 = k0*n_per_l0)."""
    k1 = k0 * n_per_l0
    l1 = torch.arange(N) % k1
    l0 = l1 // n_per_l0
    return l0.long(), l1.long(), k0, k1


def _synthetic(N=40, k0=8, n_per_l0=4, hidden=16, steps=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    # ring graph
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    x = torch.randn(N, 6, generator=g)
    edge_attr = torch.randn(edge_index.shape[1], 3, generator=g)
    pos = torch.rand(N, 2, generator=g)
    graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, num_nodes=N)

    L0, L1, K0, K1 = _nested_assign(N, k0, n_per_l0)
    cache = {
        "levels": [L0, L1],
        "rwse": {"L0": torch.randn(K0, steps, generator=g),
                 "L1": torch.randn(K1, steps, generator=g)},
        "centroid": {"L0": torch.rand(K0, 2, generator=g),
                     "L1": torch.rand(K1, 2, generator=g)},
        "area": torch.rand(N, generator=g) + 0.1,
        "pos": pos,
    }
    return graph, cache


def _model(hidden=16):
    return AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=hidden,
                   processor_size=2, rwse_steps=16,
                   transformer_layers=2, transformer_heads=2, transformer_ffn=32)


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    graph, cache = _synthetic()
    model = _model().eval()
    # fixed thresholds so some segments are active, some folded
    thr = {"G": 1e9, "omega": 0.0, "M": 1e9, "S": 1e9}  # omega>0 -> most active
    with torch.no_grad():
        pred = model(graph, cache, thresholds=thr)
    assert pred.shape == (graph.num_nodes, 3)
    assert torch.isfinite(pred).all()


def test_backward_all_params_have_grad():
    torch.manual_seed(0)
    graph, cache = _synthetic()
    model = _model().train()
    thr = {"G": 0.5, "omega": 0.5, "M": 1e9, "S": 0.5}
    pred = model(graph, cache, thresholds=thr)
    target = torch.zeros_like(pred)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"
    for n, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"{n} grad has NaN/Inf"


def test_all_folded_and_all_fine_run():
    torch.manual_seed(0)
    graph, cache = _synthetic()
    model = _model().eval()
    with torch.no_grad():
        # all calm -> folded to L0
        p_calm = model(graph, cache, thresholds={"G": 1e9, "omega": 1e9, "M": 1e9, "S": 1e9})
        # all active -> kept fine
        p_fine = model(graph, cache, thresholds={"G": -1.0, "omega": -1.0, "M": -1.0, "S": -1.0})
    assert p_calm.shape == p_fine.shape == (graph.num_nodes, 3)
    assert torch.isfinite(p_calm).all() and torch.isfinite(p_fine).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
