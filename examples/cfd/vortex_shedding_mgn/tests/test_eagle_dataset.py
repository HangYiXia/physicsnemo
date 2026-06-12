"""
Unit test for the EAGLE reader (M7, eagle_dataset.py).

Validates the PLUMBING (shapes / graph format / stats) on a tiny SYNTHETIC
EAGLE-format .npz — it does NOT validate real-EAGLE physics (no real data on
the dev machine). Confirms the reader yields the exact graph format the model
already consumes (x[N,6], y[N,3], edge_attr[E,3], pos, gidx, x_prev).

Run:
    cd examples/cfd/vortex_shedding_mgn
    pytest tests/test_eagle_dataset.py -v
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eagle_dataset import (
    load_eagle_case, EagleDatasetAMR, EAGLE_KEYS, EAGLE_TRIANGLE_KEYS,
)


def _write_fake_eagle(path, T=5, seed=0):
    """A 4-node square split into 2 triangles, with EAGLE-format keys.
    Triangles stored under an in-npz key (EAGLE keeps them in a separate file;
    the reader's resolver tries in-npz keys first, which keeps this test
    self-contained)."""
    rng = np.random.default_rng(seed)
    N = 4
    pos = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    mesh_pos = np.tile(pos[None], (T, 1, 1))              # [T,N,2] (static mesh)
    VX = rng.standard_normal((T, N)).astype(np.float32)
    VY = rng.standard_normal((T, N)).astype(np.float32)
    PS = rng.standard_normal((T, N)).astype(np.float32)
    node_type = np.array([1, 0, 0, 1], dtype=np.int64)    # boundary / normal
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    np.savez(path, **{
        EAGLE_KEYS["pos"]: mesh_pos,
        EAGLE_KEYS["vx"]: VX,
        EAGLE_KEYS["vy"]: VY,
        EAGLE_KEYS["pressure"]: PS,
        EAGLE_KEYS["node_type"]: node_type,
        EAGLE_TRIANGLE_KEYS[0]: cells,        # "cells"
    })


def test_load_eagle_case_shapes():
    with tempfile.TemporaryDirectory() as d:
        _write_fake_eagle(os.path.join(d, "0.npz"))
        out = load_eagle_case(d, "train", 0, timestep=0)
        assert out["mesh_pos"].shape == (4, 2)
        assert out["cells"].shape == (2, 3)
        assert out["edge_index"].shape[0] == 2
        assert set(np.unique(out["node_type"])).issubset({0, 1, 2, 3})
        assert out["num_nodes"] == 4


def test_eagle_dataset_graph_format():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        for i in range(2):
            _write_fake_eagle(os.path.join(d, f"{i}.npz"), seed=i)
        os.chdir(d)  # stats json land here
        try:
            ds = EagleDatasetAMR(data_dir=d, split="train", num_samples=2,
                                 num_steps=5, noise_std=0.0)
            assert len(ds) == 2 * (5 - 1)
            g = ds[0]
            assert g.x.shape == (4, 6)          # vel(2) + node-type one-hot(4)
            assert g.y.shape == (4, 3)          # vel_diff(2) + pressure(1)
            assert g.edge_attr.shape[1] == 3
            assert g.pos.shape == (4, 2)
            assert g.x_prev.shape == (4, 2)
            assert int(g.gidx) == 0
            for k in ("velocity_mean", "velocity_diff_std", "pressure_std"):
                assert k in ds.node_stats
            assert torch.isfinite(g.x).all() and torch.isfinite(g.y).all()
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
