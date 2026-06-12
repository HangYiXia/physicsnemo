"""
macro_transformer.py — Segment Encoding + Macro Transformer + Dispatch (M4)
===========================================================================
Design Doc 4.8 / 7.4.5.

SegmentEncoder : permutation-invariant per-token feature.
    h_seg_k = node_mlp( mean_{i in token k} h_node_i )
              + pe_proj( [ rwse_k (16), depth_k (1), centroid_k (2) ] )
    mean-pooling is O(Nd) and permutation-invariant (vs EAGLE's GRU).

MacroTransformer : Pre-LN TransformerEncoder over the T tokens (fully-connected
    token graph). Default 4 layers, 8 heads, d_model=128, FFN=512.

dispatch : each node takes its token's transformer output and concatenates with
    its own micro feature -> h_cat [N, 2d] (concat, not add, so the decoder
    learns to fuse local + global scales).

Batching (M5, Design Doc 7.2-C): SegmentEncoder/dispatch are batch-ready once
`kept_assign` uses global contiguous token ids. Per-graph attention isolation
is handled by `pack_segments` / `run_macro_batched` (pad to [B,Tmax,d] + mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SegmentEncoder(nn.Module):
    """Mean-pool per token + additive segment-level positional encoding."""

    def __init__(self, d: int = 128, pe_in: int = 19, pe_hidden: int = 128):
        super().__init__()
        self.d = d
        self.node_mlp = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
        )
        # pe_in = rwse(16) + depth(1) + centroid(2) = 19 by default
        self.pe_proj = nn.Sequential(
            nn.Linear(pe_in, pe_hidden), nn.SiLU(), nn.Linear(pe_hidden, d)
        )

    def forward(
        self,
        h_node: Tensor,        # [N, d]
        kept_assign: Tensor,   # [N] long in [0, T)
        T: int,
        rwse: Tensor,          # [T, 16]
        depth: Tensor,         # [T]
        centroid: Tensor,      # [T, 2]
        overlap_edges: Tensor | None = None,  # [2,E] -> δ=1 halo pooling
        overlap_w: float = 1.0,
    ) -> Tensor:               # [T, d]
        d = h_node.shape[1]
        seg_sum = torch.zeros(T, d, dtype=h_node.dtype, device=h_node.device)
        seg_sum.index_add_(0, kept_assign, h_node)
        cnt = torch.zeros(T, dtype=h_node.dtype, device=h_node.device)
        cnt.index_add_(0, kept_assign, torch.ones_like(kept_assign, dtype=h_node.dtype))
        if overlap_edges is not None:
            # δ=1 overlap: each node `src` also contributes (weight overlap_w) to
            # the token of every graph neighbor `dst`, so token features absorb a
            # 1-ring halo and segment boundaries are smoothed.
            src, dst = overlap_edges[0], overlap_edges[1]
            tok = kept_assign[dst]
            seg_sum.index_add_(0, tok, overlap_w * h_node[src])
            cnt.index_add_(0, tok, overlap_w * torch.ones_like(src, dtype=h_node.dtype))
        seg_mean = seg_sum / cnt.clamp(min=1.0).unsqueeze(1)

        h_seg = self.node_mlp(seg_mean)
        pe_in = torch.cat(
            [rwse, depth.to(h_node.dtype).unsqueeze(1), centroid.to(h_node.dtype)],
            dim=1,
        )
        h_seg = h_seg + self.pe_proj(pe_in)
        return h_seg


class MacroTransformer(nn.Module):
    """Pre-LN multi-head self-attention over the T segment tokens."""

    def __init__(self, d: int = 128, layers: int = 4, heads: int = 8,
                 ffn: int = 512, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,  # Pre-LN
        )
        # Pre-LN (norm_first=True) makes nested-tensor fast-path unusable, so
        # disable it explicitly to silence the "enable_nested_tensor is True but
        # ... norm_first was True" warning. No functional change.
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=layers, enable_nested_tensor=False
        )

    def forward(self, h_seg: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """h_seg [T, d] (single graph) or [B, T, d]. key_padding_mask [B, T]
        (True = ignore). Returns same shape as input (minus the batch dim if
        the input was 2-D)."""
        squeeze = h_seg.dim() == 2
        if squeeze:
            h_seg = h_seg.unsqueeze(0)  # [1, T, d]
        out = self.encoder(h_seg, src_key_padding_mask=key_padding_mask)
        return out.squeeze(0) if squeeze else out


def dispatch(h_seg_out: Tensor, kept_assign: Tensor, h_node: Tensor,
             overlap_edges: Tensor | None = None, overlap_w: float = 1.0) -> Tensor:
    """h_seg_out [T, d], kept_assign [N], h_node [N, d] -> h_cat [N, 2d].

    Each node grabs its token's global feature and concatenates with its own
    local feature (concat, not add). With `overlap_edges` (δ=1), the global
    feature is the (weighted) mean of the node's own token plus its 1-ring
    neighbors' tokens, matching the overlap pooling in SegmentEncoder."""
    h_global = h_seg_out[kept_assign]          # [N, d]
    if overlap_edges is not None:
        src, dst = overlap_edges[0], overlap_edges[1]
        acc = h_global.clone()
        deg = torch.ones(h_node.shape[0], dtype=h_node.dtype, device=h_node.device)
        acc.index_add_(0, src, overlap_w * h_seg_out[kept_assign[dst]])
        deg.index_add_(0, src, overlap_w * torch.ones_like(src, dtype=h_node.dtype))
        h_global = acc / deg.unsqueeze(1)
    return torch.cat([h_node, h_global], dim=1)  # [N, 2d]


# ---------------------------------------------------------------------------
# Batched macro transformer (M5)
# ---------------------------------------------------------------------------
# SegmentEncoder / dispatch already work for a batch as long as `kept_assign`
# uses GLOBAL contiguous token ids in [0, T_total) and T = T_total. The only
# part that must be batch-aware is the transformer's ATTENTION: tokens of one
# graph must not attend to tokens of another. We pack the variable-length
# per-graph token sequences into [B, Tmax, d] + key_padding_mask, run the
# encoder (so attention is confined per graph), then unpack back to [T_total,d].
#
# Requirement: `token_batch` is SEGMENTED (tokens of graph 0 first, then graph 1,
# ...), which is how model.py concatenates per-graph routing results.


def pack_segments(h_seg: Tensor, token_batch: Tensor, num_graphs: int | None = None):
    """[T_total, d] + token_batch[T_total] -> (packed[B,Tmax,d], mask[B,Tmax],
    index) where mask True = padding. `index` is reused by unpack_segments."""
    T_total, d = h_seg.shape
    B = int(token_batch.max()) + 1 if num_graphs is None else num_graphs
    counts = torch.bincount(token_batch, minlength=B)            # [B]
    Tmax = int(counts.max())
    offsets = torch.zeros(B, dtype=torch.long, device=h_seg.device)
    offsets[1:] = torch.cumsum(counts, 0)[:-1]
    pos = torch.arange(T_total, device=h_seg.device) - offsets[token_batch]
    packed = h_seg.new_zeros(B, Tmax, d)
    mask = torch.ones(B, Tmax, dtype=torch.bool, device=h_seg.device)
    packed[token_batch, pos] = h_seg
    mask[token_batch, pos] = False
    return packed, mask, (token_batch, pos)


def unpack_segments(packed: Tensor, index) -> Tensor:
    """Inverse of pack_segments: [B,Tmax,d] -> [T_total, d]."""
    token_batch, pos = index
    return packed[token_batch, pos]


def run_macro_batched(macro: "MacroTransformer", h_seg: Tensor,
                      token_batch: Tensor, num_graphs: int | None = None) -> Tensor:
    """Per-graph self-attention over a batch of variable-length token sets.

    h_seg [T_total, d], token_batch [T_total] (segmented) -> [T_total, d].
    Attention is confined within each graph via the padding mask.
    """
    packed, mask, idx = pack_segments(h_seg, token_batch, num_graphs)
    out = macro(packed, key_padding_mask=mask)   # [B, Tmax, d]
    return unpack_segments(out, idx)

