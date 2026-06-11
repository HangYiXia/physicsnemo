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

M4 runs batch_size=1; batched segment-id offset / padding is M5 (Design Doc
7.2-C). MacroTransformer still accepts a key_padding_mask for forward-compat.
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
    ) -> Tensor:               # [T, d]
        d = h_node.shape[1]
        seg_sum = torch.zeros(T, d, dtype=h_node.dtype, device=h_node.device)
        seg_sum.index_add_(0, kept_assign, h_node)
        cnt = torch.zeros(T, dtype=h_node.dtype, device=h_node.device)
        cnt.index_add_(0, kept_assign, torch.ones_like(kept_assign, dtype=h_node.dtype))
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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, h_seg: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """h_seg [T, d] (single graph) or [B, T, d]. key_padding_mask [B, T]
        (True = ignore). Returns same shape as input (minus the batch dim if
        the input was 2-D)."""
        squeeze = h_seg.dim() == 2
        if squeeze:
            h_seg = h_seg.unsqueeze(0)  # [1, T, d]
        out = self.encoder(h_seg, src_key_padding_mask=key_padding_mask)
        return out.squeeze(0) if squeeze else out


def dispatch(h_seg_out: Tensor, kept_assign: Tensor, h_node: Tensor) -> Tensor:
    """h_seg_out [T, d], kept_assign [N], h_node [N, d] -> h_cat [N, 2d].

    Each node grabs its token's global feature and concatenates with its own
    local feature (concat, not add)."""
    h_global = h_seg_out[kept_assign]          # [N, d]
    return torch.cat([h_node, h_global], dim=1)  # [N, 2d]
