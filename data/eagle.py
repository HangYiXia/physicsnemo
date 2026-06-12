"""
eagle_dataset.py — EAGLE dataset reader for AMR-M4GN (M7, Design Doc §八 M7)
===========================================================================
Loads the EAGLE turbulence dataset (Janny et al., ICLR 2023) into the SAME
graph format the model already consumes from the cylinder-flow pipeline, so the
exact same AMRM4GN / preprocessing / training / rollout code runs on EAGLE.

‼️  STATUS: keys CONFIRMED against the official EAGLE VERSION1 page
    (eagle-dataset.github.io); still **NOT validated on real EAGLE files** (no
    data on the dev machine). The on-disk layout is isolated in the ADAPTER
    section below — if your `.npz` keys, node-type ids, or the SEPARATE triangles
    file layout differ, edit `EAGLE_KEYS` / `EAGLE_TRIANGLE_*` /
    `eagle_node_type_to_class` only; the rest of the pipeline is format-agnostic.

Official format (EAGLE VERSION1): one `.npz` per simulation with keys
    mesh_pos [T,N,2], VX [T,N], VY [T,N], PS (dynamic) / PG (static) pressure,
    node_type [N] (boundary-or-not integer); the TRIANGLES are in a SEPARATE
    file. We use PS (dynamic pressure) as the pressure channel.

Documented assumptions (Design Doc D9):
  A1. EAGLE meshes are DYNAMIC (positions/cells can change per frame). AMR-M4GN's
      partition cache is built ONCE per sim. We therefore use the t=0 mesh
      (pos[0], cells[0]) as the fixed connectivity/partition for the whole sim
      and let only the FIELDS (velocity, pressure) vary in time. This matches the
      cylinder-flow "stationary mesh" assumption. True per-frame remeshing would
      need a per-frame partition (future work; see D9 / X-MGN Halo note §十一).
  A2. 2-D fields (pointcloud [T,N,2]) -> in_nodes=6, in_edges=3 unchanged.
  A3. node_type is mapped EAGLE-ids -> {0:normal, 1:inflow, 2:outflow, 3:wall}
      so the 4-way one-hot (x dim 6) is unchanged. Adjust the map if needed.
  A4. rollout updates interior (mapped type 0) nodes; boundaries keep GT.

Layout assumption (configurable):
  {data_dir}/{split}.txt   : one example id per line (optional; else glob *.npz)
  {data_dir}/{id}.npz      : one simulation per file

Use it exactly like VortexSheddingDatasetAMR (same attributes: node_stats, cells,
rollout_mask, mesh_pos, get_cache, and __getitem__ -> graph(+pos,+gidx,+x_prev)).
"""

from __future__ import annotations

import os
import glob

import numpy as np
import torch

from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from physicsnemo.datapipes.gnn.utils import load_json, save_json


# ===========================================================================
# ADAPTER — edit ONLY this section if your EAGLE files differ
# ===========================================================================
# Confirmed against the official EAGLE VERSION1 page (eagle-dataset.github.io):
#   "Simulations data are stored in a single .npz file with keys:
#      mesh_pos : 2D positions of the nodes
#      VX, VY   : velocity field at node positions
#      PS, PG   : dynamic and static pressure
#      node_type: integer per node = boundary or not
#    Triangles are stored in a SEPARATE file."
EAGLE_KEYS = {
    "pos": "mesh_pos",     # [T, N, 2]  node positions over time
    "vx": "VX",            # [T, N]     x-velocity
    "vy": "VY",            # [T, N]     y-velocity
    "pressure": "PS",      # [T, N]     DYNAMIC pressure (use "PG" for static)
    "node_type": "node_type",  # [N] or [T,N]  boundary-or-not integer
}

# Triangles live in a SEPARATE file. We try, in order:
#   1) a key inside the .npz ("cells"/"triangles"/"faces"), then
#   2) a sibling file matching one of these templates ({stem}=npz name w/o ext,
#      {dir}=its folder), then
#   3) Delaunay of the t=0 positions (fallback, with a warning).
# Adjust the templates to your actual layout (e.g. a per-geometry triangles file).
EAGLE_TRIANGLE_KEYS = ("cells", "triangles", "faces")
EAGLE_TRIANGLE_FILES = (
    "{stem}_triangles.npy", "{stem}.cells.npy",
    "{dir}/triangles/{stem}.npy", "{dir}/triangles.npy",
)

# node_type is documented as "boundary or not". Default: 0 -> normal(0), any
# nonzero -> wall(3). If your data distinguishes inflow/outflow ids, add them.
_EAGLE_INFLOW = set()        # e.g. {4}
_EAGLE_OUTFLOW = set()       # e.g. {5}


def eagle_node_type_to_class(nt: np.ndarray) -> np.ndarray:
    """Map raw EAGLE node-type ids -> {0 normal,1 inflow,2 outflow,3 wall}.
    Default: 0 stays normal, every other id is treated as a wall/boundary."""
    out = np.where(nt == 0, 0, 3).astype(np.int64)   # boundary-or-not -> normal/wall
    if _EAGLE_INFLOW:
        out[np.isin(nt, list(_EAGLE_INFLOW))] = 1
    if _EAGLE_OUTFLOW:
        out[np.isin(nt, list(_EAGLE_OUTFLOW))] = 2
    return out
# ===========================================================================


def _list_examples(data_dir: str, split: str):
    txt = os.path.join(data_dir, f"{split}.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        return [os.path.join(data_dir, f"{i}.npz") for i in ids]
    return sorted(glob.glob(os.path.join(data_dir, "*.npz")))


def _load_triangles(d, npz_path: str):
    """Resolve the triangle connectivity for one sim (separate file in EAGLE)."""
    for k in EAGLE_TRIANGLE_KEYS:           # 1) inside the npz
        if k in d:
            return np.asarray(d[k])
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    folder = os.path.dirname(npz_path)
    for tmpl in EAGLE_TRIANGLE_FILES:       # 2) sibling file
        cand = tmpl.format(stem=stem, dir=folder)
        if os.path.exists(cand):
            return np.load(cand, allow_pickle=True)
    # 3) fallback: Delaunay of the t=0 positions
    from scipy.spatial import Delaunay
    raise_pos = d[EAGLE_KEYS["pos"]]
    p0 = np.asarray(raise_pos[0] if raise_pos.ndim == 3 else raise_pos, dtype=np.float64)
    print(f"[eagle_dataset] WARNING: no triangles file for {npz_path}; "
          f"falling back to Delaunay triangulation of t=0 positions. "
          f"Point EAGLE_TRIANGLE_FILES at your real triangles file.")
    return Delaunay(p0).simplices


def _read_eagle_npz(path: str):
    """Return (pos[T,N,2], vel[T,N,2], pressure[T,N,1], node_type[N], cells[F,3]).
    Uses the t=0 mesh for connectivity (assumption A1)."""
    d = np.load(path, allow_pickle=True)
    K = EAGLE_KEYS
    pos = np.asarray(d[K["pos"]], dtype=np.float32)            # [T,N,2]
    vx = np.asarray(d[K["vx"]], dtype=np.float32)              # [T,N]
    vy = np.asarray(d[K["vy"]], dtype=np.float32)
    vel = np.stack([vx, vy], axis=-1)                          # [T,N,2]
    pressure = np.asarray(d[K["pressure"]], dtype=np.float32)[..., None]  # [T,N,1]
    cells = np.asarray(_load_triangles(d, path))
    if cells.ndim == 3:                                        # [T,F,3] -> t=0
        cells = cells[0]
    nt = np.asarray(d[K["node_type"]])
    if nt.ndim == 2:                                           # [T,N] -> t=0
        nt = nt[0]
    return pos, vel, pressure, nt.astype(np.int64), cells.astype(np.int64)


def load_eagle_case(data_dir: str, split: str, case_idx: int, timestep: int = 0):
    """Single-case loader matching visualize_partition.load_single_case so it
    plugs straight into preprocess_partitions.build_cache."""
    paths = _list_examples(data_dir, split)
    if case_idx >= len(paths):
        raise IndexError(f"case {case_idx} >= {len(paths)} EAGLE examples in {data_dir}")
    pos, vel, _, nt_raw, cells = _read_eagle_npz(paths[case_idx])
    src, dst = VortexSheddingDataset.cell_to_adj(cells)
    import torch_geometric as pyg
    edge_index = pyg.utils.to_undirected(
        torch.stack([torch.tensor(src), torch.tensor(dst)], 0).long())
    return {
        "mesh_pos": pos[timestep],                  # np [N,2]
        "cells": cells,                             # np [F,3]
        "edge_index": edge_index,                   # tensor [2,E]
        "node_type": eagle_node_type_to_class(nt_raw),  # np [N] in {0..3}
        "num_nodes": pos.shape[1],
    }


class EagleDatasetAMR(VortexSheddingDataset):
    """EAGLE counterpart of VortexSheddingDatasetAMR. Same public surface."""

    def __init__(self, name="eagle", data_dir=None, split="train",
                 num_samples=10, num_steps=100, noise_std=0.0,
                 cache_dir=None, stats_prefix="eagle"):
        self.name = name
        self.data_dir = data_dir
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.length = num_samples * (num_steps - 1)
        self.cache_dir = cache_dir
        self._cache = {}
        self._node_file = f"node_stats_{stats_prefix}.json"
        self._edge_file = f"edge_stats_{stats_prefix}.json"

        import torch_geometric as pyg
        self.pyg = pyg

        paths = _list_examples(data_dir, split)[:num_samples]
        if len(paths) < num_samples:
            raise IndexError(f"need {num_samples} EAGLE examples, found {len(paths)}")

        self.graphs, self.cells, self.node_type = [], [], []
        self.mesh_pos, self.rollout_mask = [], []
        self.node_features, self.node_targets = [], []
        noise_mask = []

        for p in paths:
            pos, vel, pressure, nt_raw, cells = _read_eagle_npz(p)
            vel = vel[:num_steps]; pressure = pressure[:num_steps]
            nt = eagle_node_type_to_class(nt_raw)               # [N] in {0..3}

            # ---- graph (t=0 stationary mesh) ----
            src, dst = VortexSheddingDataset.cell_to_adj(cells)
            graph = VortexSheddingDataset.create_graph(src, dst)
            graph = VortexSheddingDataset.add_edge_features(graph, pos[0])
            self.graphs.append(graph)

            nt_t = torch.tensor(nt, dtype=torch.uint8)
            self.node_type.append(self._one_hot4(nt_t))
            noise_mask.append(torch.eq(nt_t, 0))
            self.mesh_pos.append(torch.tensor(pos[0], dtype=torch.float32))
            self.cells.append(cells)
            self.rollout_mask.append(torch.eq(nt_t, 0))         # interior fluid

            # ---- node features / targets (same convention as base) ----
            feats = {"velocity": VortexSheddingDataset._drop_last(vel)}
            tgts = {"velocity": VortexSheddingDataset._push_forward_diff(vel),
                    "pressure": VortexSheddingDataset._push_forward(pressure)}
            if split == "train" and noise_std > 0:
                feats["velocity"], tgts["velocity"] = VortexSheddingDataset._add_noise(
                    feats["velocity"], tgts["velocity"], noise_std, noise_mask[-1])
            self.node_features.append(feats)
            self.node_targets.append(tgts)

        # ---- stats (compute on train, load otherwise) ----
        if split == "train":
            self.edge_stats = self._compute_edge_stats()
            self.node_stats = self._compute_node_stats()
        else:
            self.edge_stats = load_json(self._edge_file)
            self.node_stats = load_json(self._node_file)

        # ---- normalize ----
        for i in range(num_samples):
            self.graphs[i].edge_attr = VortexSheddingDataset.normalize_edge(
                self.graphs[i], self.edge_stats["edge_mean"], self.edge_stats["edge_std"])
            self.node_features[i]["velocity"] = VortexSheddingDataset.normalize_node(
                self.node_features[i]["velocity"],
                self.node_stats["velocity_mean"], self.node_stats["velocity_std"])
            self.node_targets[i]["velocity"] = VortexSheddingDataset.normalize_node(
                self.node_targets[i]["velocity"],
                self.node_stats["velocity_diff_mean"], self.node_stats["velocity_diff_std"])
            self.node_targets[i]["pressure"] = VortexSheddingDataset.normalize_node(
                self.node_targets[i]["pressure"],
                self.node_stats["pressure_mean"], self.node_stats["pressure_std"])

    @staticmethod
    def _one_hot4(nt: torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F
        return F.one_hot(nt.long(), num_classes=4)

    def _compute_edge_stats(self):
        s = {"edge_mean": 0, "edge_meansqr": 0}
        n = self.num_samples
        for g in self.graphs:
            s["edge_mean"] += torch.mean(g.edge_attr, 0) / n
            s["edge_meansqr"] += torch.mean(g.edge_attr ** 2, 0) / n
        s["edge_std"] = torch.sqrt(s["edge_meansqr"] - s["edge_mean"] ** 2)
        s.pop("edge_meansqr")
        save_json(s, self._edge_file)
        return s

    def _compute_node_stats(self):
        n = self.num_samples
        acc = {k: 0 for k in ["velocity_mean", "velocity_meansqr", "pressure_mean",
                              "pressure_meansqr", "velocity_diff_mean", "velocity_diff_meansqr"]}
        for i in range(n):
            v = self.node_features[i]["velocity"]
            dv = self.node_targets[i]["velocity"]
            pr = self.node_targets[i]["pressure"]
            acc["velocity_mean"] += torch.mean(v, (0, 1)) / n
            acc["velocity_meansqr"] += torch.mean(v ** 2, (0, 1)) / n
            acc["velocity_diff_mean"] += torch.mean(dv, (0, 1)) / n
            acc["velocity_diff_meansqr"] += torch.mean(dv ** 2, (0, 1)) / n
            acc["pressure_mean"] += torch.mean(pr, (0, 1)) / n
            acc["pressure_meansqr"] += torch.mean(pr ** 2, (0, 1)) / n
        s = {
            "velocity_mean": acc["velocity_mean"],
            "velocity_std": torch.sqrt(acc["velocity_meansqr"] - acc["velocity_mean"] ** 2),
            "velocity_diff_mean": acc["velocity_diff_mean"],
            "velocity_diff_std": torch.sqrt(acc["velocity_diff_meansqr"] - acc["velocity_diff_mean"] ** 2),
            "pressure_mean": acc["pressure_mean"],
            "pressure_std": torch.sqrt(acc["pressure_meansqr"] - acc["pressure_mean"] ** 2),
        }
        save_json(s, self._node_file)
        return s

    def get_cache(self, gidx: int) -> dict:
        if self.cache_dir is None:
            raise ValueError("cache_dir not set; run preprocess_partitions.py --source eagle first.")
        if gidx not in self._cache:
            path = os.path.join(self.cache_dir, f"partition_cache_{self.split}_{gidx}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"missing EAGLE partition cache: {path}\nRun: python "
                    f"preprocess_partitions.py --source eagle --split {self.split} "
                    f"--num_cases {self.num_samples}")
            self._cache[gidx] = torch.load(path, weights_only=False)
        return self._cache[gidx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)
        tidx = idx % (self.num_steps - 1)
        graph = self.graphs[gidx]
        graph.x = torch.cat(
            (self.node_features[gidx]["velocity"][tidx], self.node_type[gidx]), dim=-1)
        graph.y = torch.cat(
            (self.node_targets[gidx]["velocity"][tidx],
             self.node_targets[gidx]["pressure"][tidx]), dim=-1)
        graph.pos = self.mesh_pos[gidx].float()
        graph.gidx = gidx
        prev_t = tidx - 1 if tidx > 0 else 0
        graph.x_prev = self.node_features[gidx]["velocity"][prev_t].float()
        return graph
