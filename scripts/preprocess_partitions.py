"""
preprocess_partitions.py — offline partition/PE/area cache builder (M4)
=======================================================================
For each case of a split, run the geometry-only pipeline once and cache it:

    modal_decomp -> segmentation(L0,L1) -> RWSE(seg & node) -> l1_to_l0
    -> segment centroids -> node Voronoi area

The cache is geometry/partition only (mesh is stationary), so it is independent
of the time step; physical quantities (G/omega/M/S) are computed per-frame at
train time inside model.py.

Cache file (Design Doc 7.5):  partition_cache_{split}_{gidx}.pt  (a dict)

CLI:
    python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --split test --num_cases 1 --out_dir ./amr_cache
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch

from amr_m4gn.modal_decomp import laplacian_eigenmodes, compute_node_area
from amr_m4gn.segmentation import build_partition_tree, compute_obstacle_distance
from amr_m4gn.pe import rwse_segment, rwse_node
from amr_m4gn.amr_router import build_l1_to_l0


def _get_loader(source: str):
    """Return the single-case loader for the chosen dataset source. Both return
    the same dict {mesh_pos, cells, edge_index, node_type, num_nodes}."""
    if source == "eagle":
        from data.eagle import load_eagle_case
        return load_eagle_case
    from scripts.visualize_partition import load_single_case
    return load_single_case


def _segment_centroids(pos_t: torch.Tensor, assign: torch.Tensor, K: int) -> torch.Tensor:
    """Mean node position per segment -> [K, 2]."""
    d = pos_t.shape[1]
    s = torch.zeros(K, d, dtype=pos_t.dtype)
    s.index_add_(0, assign, pos_t)
    c = torch.zeros(K, dtype=pos_t.dtype)
    c.index_add_(0, assign, torch.ones(assign.shape[0], dtype=pos_t.dtype))
    return s / c.clamp(min=1.0).unsqueeze(1)


def build_cache(data_dir, split, case_idx, K0, K1, num_modes, tau,
                use_cotangent=True, boundary_type="neumann", steps=16,
                use_modal=True, source="tfrecord"):
    """Build the geometry/partition cache dict for one case.

    source="tfrecord" (cylinder-flow) or "eagle" (EAGLE) selects the single-case
    loader; both yield the same dict, so the rest is identical.

    use_modal=False (M6 ablation "w/o Modal Decomp"): zero the modal features so
    the SLIC refinement is guided by geometry (obstacle distance + position)
    only, i.e. a METIS+geometry partition without Laplacian-eigenmode guidance.
    """
    load_single_case = _get_loader(source)
    data = load_single_case(data_dir, split, case_idx, timestep=0)
    pos = data["mesh_pos"]             # np [N,2]
    cells = data["cells"]              # np [C,3]
    edge_index = data["edge_index"]    # tensor [2,E]
    node_type = data["node_type"]      # np [N]
    num_nodes = data["num_nodes"]

    f_md, eigvals = laplacian_eigenmodes(
        edge_index=edge_index,
        pos=pos,
        node_type=torch.tensor(node_type, dtype=torch.long),
        cells=cells if use_cotangent else None,
        num_modes=num_modes,
        use_cotangent=use_cotangent,
        boundary_type=boundary_type,
    )
    if not use_modal:
        f_md = np.zeros_like(np.asarray(f_md))

    f_obs = None
    for obs_id in [6, 1]:
        if np.any(node_type == obs_id):
            f_obs = compute_obstacle_distance(pos, node_type, obstacle_type_id=obs_id)
            break
    if f_obs is None:
        center = pos.mean(axis=0)
        f_obs = np.linalg.norm(pos - center, axis=1)

    levels, seg_adj = build_partition_tree(
        edge_index=edge_index, pos=pos, f_md=f_md,
        f_obs=torch.tensor(f_obs, dtype=torch.float32),
        K_list=(K0, K1), tau=tau, max_iter=10,
    )
    L0, L1 = levels[0], levels[1]
    K0a, K1a = int(L0.max()) + 1, int(L1.max()) + 1

    pos_t = torch.tensor(pos, dtype=torch.float32)
    area = compute_node_area(pos, cells, num_nodes)
    if not torch.is_tensor(area):
        area = torch.tensor(area, dtype=torch.float32)

    cache = {
        "levels": [L0.long(), L1.long()],
        "seg_adj": [seg_adj[0], seg_adj[1]],
        "rwse": {"L0": rwse_segment(seg_adj[0], K0a, steps),
                 "L1": rwse_segment(seg_adj[1], K1a, steps)},
        "node_pe": rwse_node(edge_index, num_nodes, steps),
        "l1_to_l0": build_l1_to_l0(levels),
        "centroid": {"L0": _segment_centroids(pos_t, L0.long(), K0a),
                     "L1": _segment_centroids(pos_t, L1.long(), K1a)},
        "area": area.float(),
        "pos": pos_t,
        "eigvals": eigvals,
        "meta": {"K0": K0a, "K1": K1a, "m": num_modes, "tau": tau,
                 "boundary_type": boundary_type,
                 "pos_min": pos.min(0).tolist(), "pos_max": pos.max(0).tolist()},
    }
    return cache


def _build_one(args_tuple):
    """Worker: build + save one case cache (module-level so it is picklable for
    multiprocessing on Windows/macOS)."""
    (c, data_dir, split, K0, K1, num_modes, tau, steps, use_modal, source,
     out_dir, suffix) = args_tuple
    cache = build_cache(data_dir, split, c, K0, K1, num_modes, tau,
                        steps=steps, use_modal=use_modal, source=source)
    path = os.path.join(out_dir, f"partition_cache_{split}_{c}{suffix}.pt")
    torch.save(cache, path)
    return c, path, cache["meta"]["K0"], cache["meta"]["K1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--source", type=str, default="tfrecord",
                   choices=["tfrecord", "eagle"],
                   help="dataset source: tfrecord (cylinder-flow) or eagle")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_cases", type=int, default=1)
    p.add_argument("--case_start", type=int, default=0)
    p.add_argument("--K0", type=int, default=64)
    p.add_argument("--K1", type=int, default=256)
    p.add_argument("--num_modes", type=int, default=6)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--out_dir", type=str, default="./amr_cache")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel processes for preprocessing (U6); 1 = serial")
    p.add_argument("--no_modal", action="store_true", default=False,
                   help="M6 ablation: build the partition WITHOUT modal-decomp "
                        "guidance (geometry-only SLIC); writes a *_nomodal.pt "
                        "cache so it never overwrites the modal cache.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_nomodal" if args.no_modal else ""
    cases = list(range(args.case_start, args.case_start + args.num_cases))
    jobs = [(c, args.data_dir, args.split, args.K0, args.K1, args.num_modes,
             args.tau, args.steps, not args.no_modal, args.source,
             args.out_dir, suffix) for c in cases]

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"[preprocess] {len(jobs)} cases ({args.source}{suffix}) "
              f"on {args.workers} workers ...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for c, path, K0a, K1a in ex.map(_build_one, jobs):
                print(f"  saved {path}  (K0={K0a}, K1={K1a})")
    else:
        for job in jobs:
            print(f"[preprocess] case {job[0]} ({args.split}{suffix}) ...")
            c, path, K0a, K1a = _build_one(job)
            print(f"  saved {path}  (K0={K0a}, K1={K1a})")
    print("done.")


if __name__ == "__main__":
    main()
