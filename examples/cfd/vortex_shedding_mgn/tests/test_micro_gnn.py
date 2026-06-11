"""
Unit tests for amr_m4gn/micro_gnn.py (M4, Design Doc 7.4.4).

    (1) shape: x[N,6] + edge_attr[E,3] -> h_node[N,hidden];
    (2) decoder bypassed: MicroGNN has no node_decoder params;
    (3) matches a MeshGraphNet's encoder->processor output (decision gate D4);
    (4) backward: every parameter receives a finite (non-NaN) gradient.

These need torch + physicsnemo (the `gnn` env). Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_micro_gnn.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data

from physicsnemo.models.meshgraphnet import MeshGraphNet
from amr_m4gn.micro_gnn import MicroGNN


def _tiny_graph(N=12, in_nodes=6, in_edges=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    # simple ring graph (each node connected to next), bidirectional
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    x = torch.randn(N, in_nodes, generator=g)
    edge_attr = torch.randn(edge_index.shape[1], in_edges, generator=g)
    graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, num_nodes=N)
    return graph, x, edge_attr


def test_shape():
    micro = MicroGNN(in_nodes=6, in_edges=3, hidden=32, processor_size=2).eval()
    graph, x, edge_attr = _tiny_graph()
    with torch.no_grad():
        h = micro(x, edge_attr, graph)
    assert h.shape == (x.shape[0], 32)


def test_no_decoder_params():
    micro = MicroGNN(in_nodes=6, in_edges=3, hidden=32, processor_size=2)
    names = [n for n, _ in micro.named_parameters()]
    assert not any("node_decoder" in n for n in names), \
        "node_decoder must be dropped in the micro branch"
    assert any("processor" in n for n in names)
    assert any("node_encoder" in n for n in names)
    assert any("edge_encoder" in n for n in names)


def test_matches_meshgraphnet_processor():
    # D4: forward must equal edge_encoder -> node_encoder -> processor of MGN.
    mgn = MeshGraphNet(input_dim_nodes=6, input_dim_edges=3, output_dim=32,
                       processor_size=2, hidden_dim_processor=32,
                       hidden_dim_node_encoder=32, hidden_dim_edge_encoder=32).eval()
    micro = MicroGNN.from_backbone(mgn).eval()
    graph, x, edge_attr = _tiny_graph()
    with torch.no_grad():
        e = mgn.edge_encoder(edge_attr)
        n = mgn.node_encoder(x)
        h_ref = mgn.processor(n, e, graph)
        h = micro(x, edge_attr, graph)
    assert torch.allclose(h, h_ref, atol=1e-6)


def test_all_params_have_finite_grad():
    micro = MicroGNN(in_nodes=6, in_edges=3, hidden=32, processor_size=2)
    graph, x, edge_attr = _tiny_graph()
    h = micro(x, edge_attr, graph)
    h.sum().backward()
    for name, p in micro.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
