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
centroid, area, ...). Single graph: pass one dict. PyG batch (M5): pass a list
of dicts (one per graph) and a batched `graph` carrying `ptr`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .micro_gnn import MicroGNN
from .macro_transformer import (
    SegmentEncoder, MacroTransformer, dispatch, run_macro_batched,
)
from .physics_ops import compute_ns_quantities, denormalize_velocity, virtual_step
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
        use_amr: bool = True,
        use_transformer: bool = True,
        use_rwse: bool = True,
        use_overlap: bool = False,
        use_virtual_step: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        # ----- M6 ablation switches (Design Doc 6.5) -----
        # use_amr=False        : keep every L1 segment fine (fixed K=K1, no fold)
        # use_transformer=False: decode from micro features only (zero global)
        # use_rwse=False       : zero the segment-level RWSE positional encoding
        # use_overlap=True     : δ=1 overlap (1-ring halo) in segment pool/dispatch
        # use_virtual_step=True: route on the forward-Euler virtual velocity
        #                        field (needs graph.x_prev; else falls back to x)
        self.use_amr = use_amr
        self.use_transformer = use_transformer
        self.use_rwse = use_rwse
        self.use_overlap = use_overlap
        self.use_virtual_step = use_virtual_step
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

    def forward(self, graph, cache, thresholds: dict | None = None) -> Tensor:
        """Single graph (cache = dict) or a PyG batch (cache = list of dicts,
        one per graph; graph must carry `ptr`). Returns pred [N_total, 3].

        micro runs on the whole batch at once; physical quantities use the
        concatenated per-graph pos/area + batched edge_index (edges never cross
        graphs); routing is done per graph and token ids are globally offset;
        the macro transformer attends per graph (run_macro_batched). The
        single-graph path is numerically identical to M4.
        """
        caches = cache if isinstance(cache, list) else [cache]
        B = len(caches)
        N = graph.x.shape[0]
        device = graph.x.device

        ptr = getattr(graph, "ptr", None)
        if ptr is None:
            if B != 1:
                raise ValueError("batch of caches requires graph.ptr (PyG batch)")
            ptr = torch.tensor([0, N], device=device)

        h_node = self.micro(graph.x, graph.edge_attr, graph)

        # M6 ablation "w/o Transformer": skip routing + macro entirely, decode
        # from the local feature only (global branch zeroed -> ~plain MGN).
        if not self.use_transformer:
            h_cat = torch.cat([h_node, torch.zeros_like(h_node)], dim=1)
            return self.decoder(h_cat)

        # physical velocity for the indicators (D1). With virtual step, route on
        # the forward-Euler field uv' = uv_t + (uv_t - uv_prev) (Eq.11) to
        # pre-refine regions about to become active; uv_prev = graph.x_prev
        # (normalized previous-frame velocity; falls back to uv_t if absent).
        uv = graph.x[:, 0:2]
        if self.use_virtual_step:
            uv_prev = getattr(graph, "x_prev", None)
            uv = virtual_step(uv, uv_prev if uv_prev is not None else uv)
        u, v = uv[:, 0], uv[:, 1]
        if self.vel_mean is not None:
            u, v = denormalize_velocity(u, v, self.vel_mean, self.vel_std)
        pos_g = caches[0]["pos"] if B == 1 else torch.cat([c["pos"] for c in caches], 0)
        area_g = caches[0]["area"] if B == 1 else torch.cat([c["area"] for c in caches], 0)
        phys = compute_ns_quantities(
            u=u, v=v, pos=pos_g, edge_index=graph.edge_index, area=area_g,
        )

        # M6 ablation "w/o AMR": force every L1 segment to stay fine (fixed
        # K=K1, no folding) by routing with -inf thresholds (everything active).
        all_active = {"G": float("-inf"), "omega": float("-inf"),
                      "M": float("-inf"), "S": float("-inf")}

        # per-graph routing, then globally offset the token ids
        kept_parts, depth_parts, rwse_parts, cen_parts, tb_parts = [], [], [], [], []
        tok_off = 0
        for b in range(B):
            s, e = int(ptr[b]), int(ptr[b + 1])
            phys_b = {k: val[s:e] for k, val in phys.items()}
            if not self.use_amr:
                thr_b = all_active
            else:
                thr_b = thresholds if thresholds is not None else \
                    sample_thresholds(self.ranges, training=self.training)
            ka, dep, T, _ = route(caches[b]["levels"], phys_b, thr_b)
            rw, ce = self._assemble_token_pe(caches[b], ka, dep, T)
            kept_parts.append(ka + tok_off)
            depth_parts.append(dep)
            rwse_parts.append(rw)
            cen_parts.append(ce)
            tb_parts.append(torch.full((T,), b, dtype=torch.long, device=device))
            tok_off += T

        kept_assign = torch.cat(kept_parts, 0)
        depth = torch.cat(depth_parts, 0)
        rwse_t = torch.cat(rwse_parts, 0)
        cen_t = torch.cat(cen_parts, 0)
        token_batch = torch.cat(tb_parts, 0)
        T_total = tok_off

        if not self.use_rwse:
            rwse_t = torch.zeros_like(rwse_t)   # M6 ablation "w/o RWSE PE"

        ov_edges = graph.edge_index if self.use_overlap else None
        h_seg = self.seg_enc(h_node, kept_assign, T_total, rwse_t, depth, cen_t,
                             overlap_edges=ov_edges)
        if B == 1:
            h_seg = self.macro(h_seg)                       # identical to M4
        else:
            h_seg = run_macro_batched(self.macro, h_seg, token_batch, B)
        h_cat = dispatch(h_seg, kept_assign, h_node, overlap_edges=ov_edges)
        return self.decoder(h_cat)
