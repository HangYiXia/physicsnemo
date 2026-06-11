"""
model.py — AMRM4GN top-level model (M4)
=======================================
Wires the pieces (Design Doc 4.x / 7.4.6):

    h_node = MicroGNN(x, edge_attr, graph)                 # [N, d]  local 15-hop
    u,v    = denormalize(x[:,:2], vel_mean, vel_std)       # D1: physical velocity
    phys   = compute_ns_quantities(u, v, pos, edge_index, area)
    kept_assign, depth, T = route(levels, phys, thresholds)
    rwse_t, cen_t         = assemble per-token PE (fine->L1, coarse->L0)
    h_seg  = SegmentEncoder(h_node, kept_assign, T, rwse_t, depth, cen_t)
    h_seg  = MacroTransformer(h_seg)
    h_cat  = dispatch(h_seg, kept_assign, h_node)          # [N, 2d]
    pred   = decoder(h_cat)                                # [N, 3] = (du, dv, p)

`cache` is the per-case dict from preprocess_partitions.py (levels, rwse,
centroid, area, l1_to_l0, ...). Single graph (batch_size=1) in M4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .micro_gnn import MicroGNN
from .macro_transformer import SegmentEncoder, MacroTransformer, dispatch
from .physics_ops import compute_ns_quantities, denormalize_velocity
from .amr_router import route, sample_thresholds, DEFAULT_RANGES


class AMRM4GN(nn.Module):
    def __init__(
        self,
        in_nodes: int = 6,
        in_edges: int = 3,
        out_dim: int = 3,
        hidden: int = 128,
        processor_size: int = 15,
        rwse_steps: int = 16,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ffn: int = 512,
        threshold_ranges: dict | None = None,
        vel_mean=None,
        vel_std=None,
    ):
        super().__init__()
        self.hidden = hidden
        self.micro = MicroGNN(in_nodes=in_nodes, in_edges=in_edges,
                              hidden=hidden, processor_size=processor_size)
        pe_in = rwse_steps + 1 + 2  # rwse + depth + centroid(x,y)
        self.seg_enc = SegmentEncoder(d=hidden, pe_in=pe_in)
        self.macro = MacroTransformer(d=hidden, layers=transformer_layers,
                                      heads=transformer_heads, ffn=transformer_ffn)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )
        self.ranges = threshold_ranges if threshold_ranges is not None else DEFAULT_RANGES

        # D1: physical velocity needs denormalization at train time. Store as
        # buffers (None -> skip denorm, e.g. when x is already physical).
        if vel_mean is not None:
            self.register_buffer("vel_mean", torch.as_tensor(vel_mean, dtype=torch.float32))
            self.register_buffer("vel_std", torch.as_tensor(vel_std, dtype=torch.float32))
        else:
            self.vel_mean = None
            self.vel_std = None

    def _assemble_token_pe(self, cache: dict, kept_assign: Tensor,
                           depth: Tensor, T: int):
        """Per-token RWSE/centroid: fine token (depth=1) -> its L1 segment,
        coarse token (depth=0) -> its L0 parent segment."""
        L0, L1 = cache["levels"][0], cache["levels"][1]
        device = kept_assign.device
        # representative node per token (any node works; nested partition)
        rep = torch.full((T,), -1, dtype=torch.long, device=device)
        rep[kept_assign] = torch.arange(kept_assign.shape[0], device=device)
        l1_of_tok = L1[rep]
        l0_of_tok = L0[rep]

        rwse_L0, rwse_L1 = cache["rwse"]["L0"], cache["rwse"]["L1"]
        cen_L0, cen_L1 = cache["centroid"]["L0"], cache["centroid"]["L1"]

        fine = (depth == 1).unsqueeze(1)
        rwse_t = torch.where(fine, rwse_L1[l1_of_tok], rwse_L0[l0_of_tok])
        cen_t = torch.where(fine, cen_L1[l1_of_tok], cen_L0[l0_of_tok])
        return rwse_t, cen_t

    def forward(self, graph, cache: dict, thresholds: dict | None = None) -> Tensor:
        h_node = self.micro(graph.x, graph.edge_attr, graph)

        # physical velocity for the indicators (D1)
        u, v = graph.x[:, 0], graph.x[:, 1]
        if self.vel_mean is not None:
            u, v = denormalize_velocity(u, v, self.vel_mean, self.vel_std)
        phys = compute_ns_quantities(
            u=u, v=v, pos=cache["pos"], edge_index=graph.edge_index,
            area=cache["area"],
        )

        if thresholds is None:
            thresholds = sample_thresholds(self.ranges, training=self.training)
        kept_assign, depth, T, _ = route(cache["levels"], phys, thresholds)

        rwse_t, cen_t = self._assemble_token_pe(cache, kept_assign, depth, T)
        h_seg = self.seg_enc(h_node, kept_assign, T, rwse_t, depth, cen_t)
        h_seg = self.macro(h_seg)
        h_cat = dispatch(h_seg, kept_assign, h_node)
        return self.decoder(h_cat)
