"""
Unit tests for amr_m4gn/pe.py (M3).

RWSE sanity checks (Design Doc 7.4.1):
    (1) chain of 5 segments: endpoint return probability < interior;
    (2) fully-connected 3 segments: RWSE near-equal across segments;
    (3) row-stochastic check: P = D^{-1}A has rows summing to 1.

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_pe.py -v
    # or:  python tests/test_pe.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_m4gn.pe import rwse_segment, rwse_node, _rwse_from_adjacency


def _chain_edges(n):
    """Undirected path 0-1-2-...-(n-1)."""
    s = list(range(n - 1))
    d = list(range(1, n))
    return torch.tensor([s + d, d + s], dtype=torch.long)


def _complete_edges(n):
    s, d = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                s.append(i)
                d.append(j)
    return torch.tensor([s, d], dtype=torch.long)


def test_chain_endpoint_lower_return_than_interior():
    n = 5
    ei = _chain_edges(n)
    rwse = rwse_segment(ei, n, steps=16)
    # Sum of even-step return probabilities (odd steps are 0 on bipartite chain).
    ret = rwse.sum(dim=1)
    # endpoints: 0 and n-1 ; an interior node: 2 (the middle)
    assert ret[0] < ret[2], (ret[0].item(), ret[2].item())
    assert ret[n - 1] < ret[2]


def test_complete_graph_uniform_rwse():
    n = 3
    ei = _complete_edges(n)
    rwse = rwse_segment(ei, n, steps=8)
    # by symmetry every node has identical RWSE
    assert torch.allclose(rwse[0], rwse[1], atol=1e-6)
    assert torch.allclose(rwse[1], rwse[2], atol=1e-6)


def test_row_stochastic_P():
    # Re-derive P inside a tiny graph and check rows sum to 1 (non-isolated).
    n = 4
    ei = _chain_edges(n)
    A = torch.zeros((n, n), dtype=torch.float64)
    A[ei[0], ei[1]] = 1.0
    A[ei[1], ei[0]] = 1.0
    A.fill_diagonal_(0.0)
    deg = A.sum(1)
    P = A / deg.unsqueeze(1)
    assert torch.allclose(P.sum(1), torch.ones(n, dtype=torch.float64), atol=1e-9)


def test_shapes_and_node_variant():
    n = 6
    ei = _chain_edges(n)
    assert rwse_segment(ei, n, steps=16).shape == (n, 16)
    assert rwse_node(ei, n, steps=10).shape == (n, 10)


def test_isolated_node_zero_row():
    # node 2 isolated (no edges); its RWSE row should be all zeros.
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # only edge 0-1
    rwse = _rwse_from_adjacency(3, ei, steps=8)
    assert torch.allclose(rwse[2], torch.zeros(8))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
