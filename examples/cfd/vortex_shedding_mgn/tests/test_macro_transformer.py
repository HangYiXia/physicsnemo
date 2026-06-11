"""
Unit tests for amr_m4gn/macro_transformer.py (M4, Design Doc 7.4.5).

    (1) SegmentEncoder mean-pool is permutation-invariant;
    (2) T=1 degenerates to a single global-average token;
    (3) MacroTransformer key_padding_mask isolates masked tokens;
    (4) dispatch: h_cat[:, :d] == h_node.

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_macro_transformer.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_m4gn.macro_transformer import SegmentEncoder, MacroTransformer, dispatch


def _seg_inputs(N=20, T=4, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    h_node = torch.randn(N, d, generator=g)
    kept_assign = torch.randint(0, T, (N,), generator=g)
    # ensure every token gets >=1 node
    kept_assign[:T] = torch.arange(T)
    rwse = torch.randn(T, 16, generator=g)
    depth = torch.randint(0, 2, (T,))
    centroid = torch.randn(T, 2, generator=g)
    return h_node, kept_assign, T, rwse, depth, centroid


def test_segment_encoder_permutation_invariant():
    torch.manual_seed(0)
    enc = SegmentEncoder(d=16).eval()
    h_node, kept_assign, T, rwse, depth, centroid = _seg_inputs()
    with torch.no_grad():
        out1 = enc(h_node, kept_assign, T, rwse, depth, centroid)
        perm = torch.randperm(h_node.shape[0])
        out2 = enc(h_node[perm], kept_assign[perm], T, rwse, depth, centroid)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_segment_encoder_T1_is_global_mean():
    torch.manual_seed(0)
    enc = SegmentEncoder(d=16).eval()
    N, d = 15, 16
    h_node = torch.randn(N, d)
    kept_assign = torch.zeros(N, dtype=torch.long)
    rwse = torch.zeros(1, 16); depth = torch.zeros(1, dtype=torch.long)
    centroid = torch.zeros(1, 2)
    with torch.no_grad():
        out = enc(h_node, kept_assign, 1, rwse, depth, centroid)
        # the node_mlp input should be the global mean of h_node
        ref = enc.node_mlp(h_node.mean(0, keepdim=True)) + enc.pe_proj(
            torch.cat([rwse, depth.float().unsqueeze(1), centroid], dim=1))
    assert torch.allclose(out, ref, atol=1e-5)


def test_macro_transformer_padding_mask_isolates():
    torch.manual_seed(0)
    mt = MacroTransformer(d=16, layers=2, heads=2, ffn=32).eval()
    d = 16
    h2 = torch.randn(2, d)
    # run with token 1 masked out -> token 0 output must equal running token 0 alone
    with torch.no_grad():
        mask = torch.tensor([[False, True]])  # ignore token 1
        out_masked = mt(h2.unsqueeze(0), key_padding_mask=mask)[0, 0]
        out_alone = mt(h2[:1])  # only token 0
    assert torch.allclose(out_masked, out_alone[0], atol=1e-5)


def test_dispatch_keeps_local_feature():
    torch.manual_seed(0)
    N, T, d = 12, 3, 8
    h_node = torch.randn(N, d)
    kept_assign = torch.randint(0, T, (N,))
    h_seg_out = torch.randn(T, d)
    h_cat = dispatch(h_seg_out, kept_assign, h_node)
    assert h_cat.shape == (N, 2 * d)
    assert torch.allclose(h_cat[:, :d], h_node)
    # global half equals the node's token feature
    assert torch.allclose(h_cat[:, d:], h_seg_out[kept_assign])


def test_segment_encoder_shape():
    enc = SegmentEncoder(d=16)
    h_node, kept_assign, T, rwse, depth, centroid = _seg_inputs()
    out = enc(h_node, kept_assign, T, rwse, depth, centroid)
    assert out.shape == (T, 16)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
