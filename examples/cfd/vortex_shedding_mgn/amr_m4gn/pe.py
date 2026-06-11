"""
pe.py — Positional / structural encodings (M3, used from M4 onward)
===================================================================
Random-Walk Structural Encoding (RWSE) at two granularities:

    rwse_segment : per-segment structural encoding on the segment-adjacency
                   graph (used by the SegmentEncoder, Design Doc 4.8).
    rwse_node    : per-node absolute PE on the mesh graph (Encoder input, 4.4).

RWSE definition (Dwivedi et al., "Graph Neural Networks with Learnable
Structural and Positional Representations"):
    P = D^{-1} A   (row-stochastic random-walk matrix)
    rwse[i] = [ (P^1)_ii, (P^2)_ii, ..., (P^steps)_ii ]
i.e. the return probability of a random walk of length 1..steps starting at i.

Computed offline and cached. Pure dense torch (segment/node counts are small:
K1<=256, N~2000), no torch.sparse dependency required.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rwse_from_adjacency(num: int, edge_index: Tensor, steps: int) -> Tensor:
    """Shared core: build P = D^{-1} A (undirected, self-loop-free) and return
    diag(P^1 .. P^steps) as [num, steps].

    Isolated nodes (degree 0) get an all-zero row (no return probability).
    """
    A = torch.zeros((num, num), dtype=torch.float64, device=edge_index.device)
    if edge_index.numel() > 0:
        s, d = edge_index[0], edge_index[1]
        A[s, d] = 1.0
        A[d, s] = 1.0  # force symmetry (undirected)
    A.fill_diagonal_(0.0)

    deg = A.sum(dim=1)
    inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    P = A * inv.unsqueeze(1)  # row-normalised: P[i,j] = A[i,j]/deg[i]

    out = torch.empty((num, steps), dtype=torch.float64, device=edge_index.device)
    Pk = P.clone()
    for k in range(steps):
        out[:, k] = torch.diagonal(Pk)
        if k < steps - 1:
            Pk = Pk @ P
    return out.to(torch.float32)


def rwse_segment(seg_adj: Tensor, num_segments: int, steps: int = 16) -> Tensor:
    """Segment-level RWSE.

    seg_adj : [2, E_seg] segment adjacency edge_index (from build_partition_tree).
    Returns [num_segments, steps].
    """
    return _rwse_from_adjacency(num_segments, seg_adj, steps)


def rwse_node(edge_index: Tensor, num_nodes: int, steps: int = 16) -> Tensor:
    """Node-level RWSE (absolute PE) on the mesh graph.

    edge_index : [2, E] mesh edges.
    Returns [num_nodes, steps].
    """
    return _rwse_from_adjacency(num_nodes, edge_index, steps)
