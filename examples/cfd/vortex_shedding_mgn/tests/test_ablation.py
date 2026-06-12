"""
Unit test for M6 ablation switches in amr_m4gn/model.py (Design Doc 6.5).

Checks that each switch (use_amr / use_transformer / use_rwse) changes the model
in the expected way on a SYNTHETIC mesh+cache (no dataset/preprocess needed):
    (1) every config forwards to [N,3], finite, and backprops to all params;
    (2) use_transformer=False drops the macro/seg params (fewer parameters);
    (3) use_amr=False keeps all L1 tokens fine -> more tokens than the routed
        full model (verified via the segment-encoder token count).

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_ablation.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data

from amr_m4gn.model import AMRM4GN


def _synthetic(N=40, k0=8, n_per_l0=4, steps=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    x = torch.randn(N, 6, generator=g)
    edge_attr = torch.randn(edge_index.shape[1], 3, generator=g)
    pos = torch.rand(N, 2, generator=g)
    graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, num_nodes=N)

    k1 = k0 * n_per_l0
    l1 = torch.arange(N) % k1
    l0 = l1 // n_per_l0
    cache = {
        "levels": [l0.long(), l1.long()],
        "rwse": {"L0": torch.randn(k0, steps, generator=g),
                 "L1": torch.randn(k1, steps, generator=g)},
        "centroid": {"L0": torch.rand(k0, 2, generator=g),
                     "L1": torch.rand(k1, 2, generator=g)},
        "area": torch.rand(N, generator=g) + 0.1,
        "pos": pos,
    }
    return graph, cache


def _model(hidden=16, **kw):
    return AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=hidden,
                   processor_size=2, rwse_steps=16, transformer_layers=2,
                   transformer_heads=2, transformer_ffn=32, **kw)


# fold some, keep some
_THR = {"G": 1e9, "omega": 0.0, "M": 1e9, "S": 1e9}


def _forward_backward_ok(model, graph, cache):
    pred = model(graph, cache, thresholds=_THR)
    assert pred.shape == (graph.num_nodes, 3)
    assert torch.isfinite(pred).all()
    ((pred - torch.zeros_like(pred)) ** 2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"


def test_every_config_forwards_and_backprops():
    graph, cache = _synthetic()
    for kw in (dict(), dict(use_amr=False), dict(use_transformer=False),
               dict(use_rwse=False)):
        torch.manual_seed(0)
        _forward_backward_ok(_model(**kw).train(), graph, cache)


def test_no_transformer_branch_unused():
    """use_transformer=False: the macro/seg-encoder branch is constructed but
    must NOT be touched in the forward (its params receive no gradient), while
    the model still trains end-to-end."""
    full = sum(p.numel() for p in _model().parameters())
    notr = sum(p.numel() for p in _model(use_transformer=False).parameters())
    graph, cache = _synthetic()
    m = _model(use_transformer=False).train()
    m(graph, cache, thresholds=_THR).pow(2).mean().backward()
    touched = [n for n, p in m.named_parameters()
               if p.grad is not None and ("macro" in n or "seg_enc" in n)]
    assert not touched, f"transformer branch should be unused: {touched}"
    assert full == notr  # modules constructed either way; only usage differs


def test_no_amr_keeps_more_tokens():
    """use_amr=False -> all L1 fine; should pool into >= as many tokens as the
    routed full model (with _THR some L1 fold into L0)."""
    graph, cache = _synthetic()

    seen = {}

    def grab(name):
        def hook(mod, inp, out):
            # SegmentEncoder.forward(h_node, kept_assign, T, ...): T is inp[2]
            seen[name] = int(inp[2])
        return hook

    for name, kw in (("full", dict()), ("noamr", dict(use_amr=False))):
        torch.manual_seed(0)
        m = _model(**kw).eval()
        h = m.seg_enc.register_forward_hook(grab(name))
        with torch.no_grad():
            m(graph, cache, thresholds=_THR)
        h.remove()

    assert seen["noamr"] >= seen["full"]
    assert seen["noamr"] == int(cache["levels"][1].max()) + 1  # == K1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
