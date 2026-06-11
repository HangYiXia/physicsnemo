"""
micro_gnn.py — Local GNN backbone (MeshGraphNet, decoder bypassed) (M4)
======================================================================
Wraps PhysicsNeMo's `MeshGraphNet` but stops at the `processor` output, giving
per-node features `h_node [N, hidden]` that carry the 15-hop local detail
(Design Doc 4.3 / 7.4.4). The built-in `node_decoder` is intentionally dropped
so the micro branch produces features, not the final prediction (the unified
decoder in model.py consumes [h_node ; h_global]).

Decision gate D4 (uncertainty U2): `MeshGraphNet` exposes the submodules
`edge_encoder`, `node_encoder`, `processor`, `node_decoder`; its forward is
`edge_encoder -> node_encoder -> processor -> node_decoder`. We confirmed these
attribute names on the installed version (path (a) of Design Doc 7.2-D). We hold
ONLY the three pre-decoder submodules, so `node_decoder` is neither stored nor
counted in `parameters()` (keeps "every parameter receives a gradient" true).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.models.meshgraphnet import MeshGraphNet


class MicroGNN(nn.Module):
    """MeshGraphNet encoder + processor, decoder bypassed -> h_node [N, hidden]."""

    def __init__(
        self,
        in_nodes: int = 6,
        in_edges: int = 3,
        hidden: int = 128,
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        # Build a temporary MeshGraphNet, then keep only its pre-decoder parts.
        # output_dim is irrelevant (the decoder is discarded); use `hidden`.
        backbone = MeshGraphNet(
            input_dim_nodes=in_nodes,
            input_dim_edges=in_edges,
            output_dim=hidden,
            processor_size=processor_size,
            hidden_dim_processor=hidden,
            hidden_dim_node_encoder=hidden,
            hidden_dim_edge_encoder=hidden,
            mlp_activation_fn=mlp_activation_fn,
            recompute_activation=recompute_activation,
        )
        self._adopt(backbone)

    def _adopt(self, backbone: MeshGraphNet):
        # Register only the three pre-decoder submodules; drop node_decoder.
        self.edge_encoder = backbone.edge_encoder
        self.node_encoder = backbone.node_encoder
        self.processor = backbone.processor

    @classmethod
    def from_backbone(cls, backbone: MeshGraphNet) -> "MicroGNN":
        """Reuse an existing MeshGraphNet's submodules (e.g. to share/compare
        weights with a baseline). The decoder is still dropped."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.hidden = None  # not needed when adopting an existing backbone
        obj._adopt(backbone)
        return obj

    def forward(self, x: Tensor, edge_attr: Tensor, graph) -> Tensor:
        """x [N, in_nodes], edge_attr [E, in_edges], graph (PyG Data with
        edge_index). Returns h_node [N, hidden]."""
        edge_feat = self.edge_encoder(edge_attr)
        node_feat = self.node_encoder(x)
        h_node = self.processor(node_feat, edge_feat, graph)
        return h_node
