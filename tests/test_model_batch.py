"""
Batch integration test for amr_m4gn/model.py (M5 small-step 2).

Builds a PyG Batch of 2 synthetic graphs + a list of 2 caches and checks:
    (1) forward -> pred [N_total, 3], all finite;
    (2) backward -> every parameter has a finite gradient;
    (3) batch forward == per-graph single forwards concatenated (eval mode,
        fixed thresholds) -> proves no cross-graph leakage and that the
        single-graph path stays consistent with the batched one.

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_model_batch.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data, Batch

from amr_m4gn.model import AMRM4GN


def _nested(N, k0, n_per_l0):
    k1 = k0 * n_per_l0
    l1 = (torch.arange(N) % k1).long()
    l0 = (l1 // n_per_l0).long()
    return l0, l1, k0, k1


def _graph_and_cache(N, k0, n_per_l0, steps=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(N); dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    x = torch.randn(N, 6, generator=g)
    edge_attr = torch.randn(edge_index.shape[1], 3, generator=g)
    data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, num_nodes=N)
    L0, L1, K0, K1 = _nested(N, k0, n_per_l0)
    cache = {
        "levels": [L0, L1],
        "rwse": {"L0": torch.randn(K0, steps, generator=g),
                 "L1": torch.randn(K1, steps, generator=g)},
        "centroid": {"L0": torch.rand(K0, 2, generator=g),
                     "L1": torch.rand(K1, 2, generator=g)},
        "area": torch.rand(N, generator=g) + 0.1,
        "pos": torch.rand(N, 2, generator=g),
    }
    return data, cache


def _model(hidden=16):
    return AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=hidden,
                   processor_size=2, rwse_steps=16,
                   transformer_layers=2, transformer_heads=2, transformer_ffn=32)


_THR = {"G": 1e9, "omega": 0.0, "M": 1e9, "S": 1e9}  # omega>0 active


def test_batch_forward_finite_and_grad():
    torch.manual_seed(0)
    d0, c0 = _graph_and_cache(36, 6, 4, seed=1)
    d1, c1 = _graph_and_cache(28, 7, 4, seed=2)
    batch = Batch.from_data_list([d0, d1])
    model = _model().train()
    pred = model(batch, [c0, c1], thresholds=_THR)
    assert pred.shape == (36 + 28, 3)
    assert torch.isfinite(pred).all()
    pred.pow(2).mean().backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, f"{n} no grad"
        assert torch.isfinite(p.grad).all(), f"{n} grad NaN/Inf"


def test_batch_equals_per_graph():
    torch.manual_seed(0)
    d0, c0 = _graph_and_cache(36, 6, 4, seed=1)
    d1, c1 = _graph_and_cache(28, 7, 4, seed=2)
    model = _model().eval()
    with torch.no_grad():
        batch = Batch.from_data_list([d0, d1])
        out_batch = model(batch, [c0, c1], thresholds=_THR)
        out0 = model(d0, c0, thresholds=_THR)
        out1 = model(d1, c1, thresholds=_THR)
    ref = torch.cat([out0, out1], dim=0)
    assert torch.allclose(out_batch, ref, atol=1e-5), \
        "batched forward must equal per-graph forwards concatenated"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
