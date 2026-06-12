"""
Unit tests for the batched macro transformer (M5, Design Doc 7.2-C).

    (1) pack/unpack round-trip: unpack(pack(h)) == h;
    (2) padding mask correct (True exactly on padding slots);
    (3) per-graph attention isolation: run_macro_batched on a 2-graph batch
        equals concatenating the per-graph single-graph results
        (i.e. tokens of one graph do NOT attend to another's).

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_batched_macro.py -v
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_m4gn.macro_transformer import (
    MacroTransformer, pack_segments, unpack_segments, run_macro_batched,
)


def _segmented_token_batch(sizes):
    """token_batch like [0,0,0,1,1,...] from per-graph token counts."""
    return torch.cat([torch.full((n,), b, dtype=torch.long)
                      for b, n in enumerate(sizes)])


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    d = 8
    sizes = [3, 5, 2]
    tb = _segmented_token_batch(sizes)
    h = torch.randn(sum(sizes), d)
    packed, mask, idx = pack_segments(h, tb)
    assert packed.shape == (3, 5, d)        # B=3, Tmax=5
    assert torch.equal(unpack_segments(packed, idx), h)


def test_padding_mask_correct():
    sizes = [3, 5, 2]
    tb = _segmented_token_batch(sizes)
    h = torch.randn(sum(sizes), 4)
    _, mask, _ = pack_segments(h, tb)
    # number of valid (non-pad) slots per row == per-graph token count
    valid = (~mask).sum(dim=1)
    assert valid.tolist() == sizes


def test_per_graph_attention_isolation():
    torch.manual_seed(0)
    d = 16
    macro = MacroTransformer(d=d, layers=2, heads=2, ffn=32).eval()
    sizes = [4, 7]
    tb = _segmented_token_batch(sizes)
    h = torch.randn(sum(sizes), d)

    with torch.no_grad():
        out_batched = run_macro_batched(macro, h, tb)
        # reference: run each graph's tokens alone, concatenate
        h0, h1 = h[:4], h[4:]
        ref = torch.cat([macro(h0), macro(h1)], dim=0)
    assert torch.allclose(out_batched, ref, atol=1e-5), \
        "batched result must equal per-graph single runs (no cross-graph leak)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
