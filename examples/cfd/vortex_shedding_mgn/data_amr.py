"""
data_amr.py — VortexSheddingDataset subclass exposing pos + cache (M4, D5/P1)
============================================================================
Decision gate D5 path P1: subclass the stock dataset and additionally expose
`graph.pos` and `graph.gidx` for ALL splits (the base class only stores
`mesh_pos` for non-train splits). The mesh is stationary, so we read each
case's mesh_pos once. Per-case partition caches (preprocess_partitions.py) are
loaded lazily by gidx.

Use with batch_size=1 in M4 (batched segment offsets are M5).
"""

from __future__ import annotations

import os

import torch

from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset


class VortexSheddingDatasetAMR(VortexSheddingDataset):
    def __init__(self, *args, cache_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure mesh_pos/cells are available for every split (stationary mesh).
        if len(self.mesh_pos) == 0:
            self._load_mesh_pos()
        self.cache_dir = cache_dir
        self._cache = {}  # gidx -> cache dict (lazy)

    def _load_mesh_pos(self):
        """Read mesh_pos/cells once (train split skips them in the base class)."""
        ds = self._load_tfrecord_dataset(self.data_dir, self.split)
        for i, data_np in enumerate(ds):
            if i >= self.num_samples:
                break
            self.mesh_pos.append(torch.tensor(data_np["mesh_pos"][0]))
            self.cells.append(data_np["cells"][0])

    def get_cache(self, gidx: int) -> dict:
        """Lazy-load the per-case partition cache by graph index."""
        if self.cache_dir is None:
            raise ValueError("cache_dir not set; run preprocess_partitions.py first.")
        if gidx not in self._cache:
            path = os.path.join(self.cache_dir, f"partition_cache_{self.split}_{gidx}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"missing partition cache: {path}\n"
                    f"Run: python preprocess_partitions.py --data_dir <...> "
                    f"--split {self.split} --num_cases {self.num_samples}")
            self._cache[gidx] = torch.load(path, weights_only=False)
        return self._cache[gidx]

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        graph = out[0] if isinstance(out, tuple) else out
        gidx = idx // (self.num_steps - 1)
        tidx = idx % (self.num_steps - 1)
        graph.pos = self.mesh_pos[gidx].float()
        graph.gidx = gidx
        # previous-frame normalized velocity for the virtual-step router (M6).
        # tidx==0 has no history -> reuse the current frame (virtual step = id).
        prev_t = tidx - 1 if tidx > 0 else 0
        graph.x_prev = self.node_features[gidx]["velocity"][prev_t].float()
        return graph


def make_amr_dataset(dataset: str = "vortex", **kwargs):
    """Factory selecting the AMR dataset by source (M7).

    dataset="vortex" -> VortexSheddingDatasetAMR (cylinder-flow TFRecord)
    dataset="eagle"  -> EagleDatasetAMR (EAGLE .npz)
    Both expose the same surface (node_stats/cells/rollout_mask/get_cache and
    __getitem__ -> graph(+pos,+gidx,+x_prev)), so callers only switch the source.
    """
    if dataset == "eagle":
        from eagle_dataset import EagleDatasetAMR
        return EagleDatasetAMR(**kwargs)
    return VortexSheddingDatasetAMR(**kwargs)
