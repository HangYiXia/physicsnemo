"""
visualize_partition.py — Standalone Visualization for AMR-M4GN Preprocessing
=============================================================================
Loads ONE case from the vortex_shedding dataset, runs modal decomposition
and hybrid segmentation, then produces diagnostic visualizations:

1. Mesh connectivity plot
2. First 6 Laplacian eigenmodes (color-mapped on mesh)
3. Obstacle distance field
4. Level 0 (64 segments) partition
5. Level 1 (256 segments) partition
6. Segment adjacency graph overlay

Usage:
    cd E:\\phys\\physicsnemo\\examples\\cfd\\vortex_shedding_mgn
    python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow

Output:
    Saves PNG files to ./partition_vis/ directory.
"""

import os
import sys
import json
import atexit
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_m4gn.modal_decomp import laplacian_eigenmodes, compute_node_area
from amr_m4gn.segmentation import (
    build_partition_tree,
    compute_obstacle_distance,
    hybrid_segmentation,
)
from amr_m4gn.physics_ops import compute_ns_quantities
from amr_m4gn.amr_router import aggregate_per_segment, route


class _Tee:
    """Duplicate everything written to stdout into a log file as well.

    Lets the user save the full console output to a plain-text file without
    manual copy-paste (handy when running on a different machine). Each write
    is flushed immediately so a partial log survives even if the script crashes.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            if getattr(s, "closed", False):
                continue
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            if getattr(s, "closed", False):
                continue
            s.flush()


def load_single_case(data_dir, split="test", case_idx=0, timestep=0):
    """
    Load a single case from the TFRecord dataset.
    Returns mesh_pos, cells, node_type, edge_index, velocity (at `timestep`).

    `timestep` selects which frame of the trajectory to read the velocity from.
    t=0 is the (undeveloped) initial flow with no vortex street; use a later
    frame (e.g. 300) to see the developed Karman wake. Mesh geometry is static,
    so pos/cells/node_type are time-independent.
    """
    # Import tfrecord
    try:
        from tfrecord.torch.dataset import TFRecordDataset
    except ImportError:
        raise ImportError(
            "tfrecord package not found. Install with: pip install tfrecord"
        )
    
    # Load metadata
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.loads(f.read())
    
    # Load TFRecord
    tfrecord_path = os.path.join(data_dir, f"{split}.tfrecord")
    index_path = os.path.join(data_dir, f"{split}.tfindex")
    if not os.path.exists(index_path):
        index_path = None
    
    description = {k: "byte" for k in meta["field_names"]}
    
    def decode_record(rec_bytes):
        outvar = {}
        for k, v in meta["features"].items():
            dtype_map = {
                "float32": np.float32,
                "float64": np.float64,
                "int32": np.int32,
                "int64": np.int64,
            }
            dtype = dtype_map.get(v["dtype"], getattr(np, v["dtype"]))
            data = np.frombuffer(rec_bytes[k], dtype=dtype).copy()
            data = data.reshape(v["shape"])
            if v["type"] == "static":
                data = np.tile(data, (meta["trajectory_length"], 1, 1))
            outvar[k] = data
        return outvar
    
    dataset = TFRecordDataset(
        tfrecord_path, index_path, description, transform=decode_record
    )
    
    # Get the specified case
    for i, data_np in enumerate(dataset):
        if i == case_idx:
            break
    
    mesh_pos = data_np["mesh_pos"][0]      # [N, 2]
    cells = data_np["cells"][0]            # [num_cells, 3]
    node_type = data_np["node_type"][0]    # [N, 1]
    traj_len = data_np["velocity"].shape[0]
    t = int(np.clip(timestep, 0, traj_len - 1))
    if t != timestep:
        print(f"  [load] timestep {timestep} out of range [0,{traj_len-1}], "
              f"clamped to {t}.")
    velocity = data_np["velocity"][t]      # [N, 2] at timestep t
    
    # Build edge_index from cells
    num_cells = cells.shape[0]
    src = [cells[i][idx] for i in range(num_cells) for idx in [0, 1, 2]]
    dst = [cells[i][idx] for i in range(num_cells) for idx in [1, 2, 0]]
    edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
    
    # Make undirected
    from torch_geometric.utils import to_undirected
    edge_index = to_undirected(edges)
    
    return {
        "mesh_pos": mesh_pos,
        "cells": cells,
        "node_type": node_type.flatten(),
        "edge_index": edge_index,
        "velocity": velocity,
        "num_nodes": mesh_pos.shape[0],
    }


def plot_mesh(pos, cells, node_type, save_path):
    """Plot the mesh with node types color-coded."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Create triangulation
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    
    # Plot triangles
    ax.triplot(triang, 'k-', linewidth=0.15, alpha=0.3)
    
    # Color nodes by type
    unique_types = np.unique(node_type)
    colors = cm.tab10(np.linspace(0, 1, len(unique_types)))
    # DeepMind MeshGraphNet NodeType convention (common.py):
    #   0=NORMAL(fluid), 1=OBSTACLE, 4=INFLOW, 5=OUTFLOW, 6=WALL_BOUNDARY.
    # In cylinder_flow the cylinder surface and channel walls are both type 6.
    type_names = {0: "Fluid", 1: "Obstacle", 4: "Inflow", 5: "Outflow",
                  6: "Wall/Cylinder", 9: "Wall"}
    
    for idx, t in enumerate(unique_types):
        mask = node_type == t
        label = type_names.get(int(t), f"Type {int(t)}")
        ax.scatter(pos[mask, 0], pos[mask, 1], s=2, c=[colors[idx]],
                   label=f"{label} ({mask.sum()} nodes)", alpha=0.7)
    
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"Mesh: {pos.shape[0]} nodes, {cells.shape[0]} triangles")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_eigenmodes(pos, cells, f_md, eigvals, save_path):
    """Plot the first m Laplacian eigenmodes."""
    m = f_md.shape[1]
    ncols = 3
    nrows = (m + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    
    for idx in range(m):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        mode_vals = f_md[:, idx]
        vmax = max(abs(mode_vals.max()), abs(mode_vals.min()))
        
        tc = ax.tripcolor(triang, mode_vals, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(f"Mode {idx+1} (λ={eigvals[idx]:.4f})", fontsize=10)
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for idx in range(m, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)
    
    plt.suptitle("Laplacian Eigenmodes (f_md)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_obstacle_distance(pos, cells, f_obs, save_path):
    """Plot the obstacle distance field."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    tc = ax.tripcolor(triang, f_obs, cmap='viridis', shading='gouraud')
    ax.set_aspect('equal')
    ax.set_title("Obstacle Distance Field (f_obs)")
    fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_physics_fields(pos, cells, quantities, save_path):
    """Plot the four AMR physical indicators G / omega / M / S on the mesh.

    Diverging colormap (symmetric about 0) for omega & S; sequential for G & M.
    """
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)

    specs = [
        ("G",     "Velocity-gradient |grad U|",          "viridis", False),
        ("omega", "Vorticity  dv/dx - du/dy",            "RdBu_r",  True),
        ("M",     "Momentum  rho |U| area",              "viridis", False),
        ("S",     "Strain-rate  sqrt(2 S_ij S_ij)",      "viridis", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    for ax, (key, title, cmap, diverging) in zip(axes.flat, specs):
        field = np.asarray(quantities[key])
        if diverging:
            vmax = np.percentile(np.abs(field), 99) + 1e-12
            tc = ax.tripcolor(triang, field, cmap=cmap, shading='gouraud',
                              vmin=-vmax, vmax=vmax)
        else:
            vmax = np.percentile(field, 99) + 1e-12
            tc = ax.tripcolor(triang, field, cmap=cmap, shading='gouraud',
                              vmin=0, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("AMR Physical Indicators (M2)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_routing(pos, cells, kept_depth_per_node, kept_assign, T, save_path):
    """Visualize the AMR fold/keep decision (M3).

    Left  : nodes colored by depth (red = kept fine L1 token, blue = folded
            back into a coarse L0 token).
    Right : nodes colored by final token id (shows the variable-size tokens).
    """
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    tc0 = axes[0].tripcolor(triang, kept_depth_per_node.astype(float),
                            cmap="coolwarm", shading="gouraud", vmin=0, vmax=1)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"AMR routing: kept fine (red) vs folded coarse (blue) — T={T}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(tc0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1])

    tc1 = axes[1].tripcolor(triang, kept_assign.astype(float),
                            cmap="tab20", shading="flat")
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Final token id per node ({T} tokens)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(tc1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("AMR Token Router (M3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_partition(pos, cells, assign, level_name, num_segments, save_path,
                   seg_adjacency=None):
    """Plot mesh colored by segment assignment."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    
    # Use a colormap with enough distinct colors
    K = int(assign.max()) + 1
    # matplotlib >=3.7: cm.get_cmap is deprecated; use colormaps[...].resampled.
    _name = 'tab20' if K <= 20 else 'nipy_spectral'
    cmap = matplotlib.colormaps[_name].resampled(K)
    
    tc = ax.tripcolor(triang, assign.astype(float), cmap=cmap,
                      shading='flat', alpha=0.8)
    ax.triplot(triang, 'k-', linewidth=0.05, alpha=0.15)
    
    # Draw segment adjacency edges
    if seg_adjacency is not None and seg_adjacency.shape[1] > 0:
        # Compute segment centroids
        seg_centroids = np.zeros((K, 2))
        seg_counts = np.zeros(K)
        for i in range(pos.shape[0]):
            seg_centroids[assign[i]] += pos[i]
            seg_counts[assign[i]] += 1
        seg_counts = np.maximum(seg_counts, 1)
        seg_centroids /= seg_counts[:, None]
        
        # Draw edges between segment centroids
        src_seg = seg_adjacency[0].numpy()
        dst_seg = seg_adjacency[1].numpy()
        # Only draw one direction (undirected)
        seen = set()
        lines = []
        for s, d in zip(src_seg, dst_seg):
            edge_key = (min(s, d), max(s, d))
            if edge_key not in seen:
                seen.add(edge_key)
                lines.append([seg_centroids[s], seg_centroids[d]])
        
        if lines:
            lc = LineCollection(lines, colors='red', linewidths=0.8, alpha=0.5)
            ax.add_collection(lc)
    
    ax.set_aspect('equal')
    ax.set_title(f"{level_name}: {K} segments (target: {num_segments})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Add segment count stats
    seg_sizes = [np.sum(assign == k) for k in range(K)]
    stats_text = (f"Nodes/seg: min={min(seg_sizes)}, max={max(seg_sizes)}, "
                  f"mean={np.mean(seg_sizes):.1f}, std={np.std(seg_sizes):.1f}")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_velocity_field(pos, cells, velocity, save_path):
    """Plot velocity magnitude on the mesh."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    vel_mag = np.linalg.norm(velocity, axis=1)
    
    tc = ax.tripcolor(triang, vel_mag, cmap='jet', shading='gouraud')
    ax.set_aspect('equal')
    ax.set_title("Velocity Magnitude |u| (timestep 0)")
    fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AMR-M4GN preprocessing (modal decomp + partition)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="./raw_dataset/cylinder_flow/cylinder_flow",
        help="Path to the vortex shedding TFRecord directory"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to load (default: test)"
    )
    parser.add_argument(
        "--case_idx", type=int, default=0,
        help="Which case to visualize (default: 0)"
    )
    parser.add_argument(
        "--timestep", type=int, default=0,
        help="Trajectory frame for the velocity field (default: 0 = initial, "
             "undeveloped flow). Use a later frame (e.g. 300) to see the "
             "developed Karman vortex street in the physics fields."
    )
    parser.add_argument(
        "--num_modes", type=int, default=6,
        help="Number of Laplacian eigenmodes (default: 6)"
    )
    parser.add_argument(
        "--K0", type=int, default=64,
        help="Number of segments at Level 0 (default: 64)"
    )
    parser.add_argument(
        "--K1", type=int, default=256,
        help="Number of segments at Level 1 (default: 256)"
    )
    parser.add_argument(
        "--tau", type=float, default=1.0,
        help="SLIC compactness parameter (default: 1.0)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./partition_vis",
        help="Directory to save output images"
    )
    parser.add_argument(
        "--use_cotangent", action="store_true", default=True,
        help="Use cotangent (FEM) Laplacian instead of graph Laplacian"
    )
    parser.add_argument(
        "--boundary_type", type=str, default="neumann",
        choices=["neumann", "dirichlet"],
        help="Boundary condition type for Laplacian (default: neumann)"
    )
    parser.add_argument(
        "--plot_physics", action="store_true", default=False,
        help="Also compute & plot the four AMR physical indicators G/omega/M/S "
             "(M2). Uses PHYSICAL velocity from TFRecord (no denormalization)."
    )
    parser.add_argument(
        "--plot_routing", action="store_true", default=False,
        help="Also run the M3 AMR router (fold/keep) and plot the result "
             "(08_routing.png) + print the token-count T (decision gate D3). "
             "Implies physical-quantity computation."
    )
    parser.add_argument(
        "--route_pct", type=float, default=70.0,
        help="Demo threshold for routing: per-channel percentile of the L1 "
             "per-segment aggregated |phys|; segments above it are 'active' "
             "and kept fine (default 70). NOTE: real thresholds = decision "
             "gate D3 / train-time sampling; this is only a visualization aid."
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Path of the plain-text file to also save all console output to. "
             "Default: <output_dir>/run_log_case<idx>_t<timestep>.txt"
    )
    parser.add_argument(
        "--no_log", action="store_true", default=False,
        help="Disable saving console output to a text file."
    )
    args = parser.parse_args()

    # Save every output (images + cache + log) into a per-case subfolder so runs
    # with different --case_idx don't overwrite each other.
    args.output_dir = os.path.join(args.output_dir, f"case{args.case_idx}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Tee console output to a plain-text log file (unless disabled) ----
    if not args.no_log:
        log_path = args.log_file or os.path.join(
            args.output_dir,
            f"run_log_case{args.case_idx}_t{args.timestep}.txt",
        )
        _log_handle = open(log_path, "w", encoding="utf-8")

        def _close_log():
            # Restore the real stdout BEFORE closing the file, so the
            # interpreter's final flush doesn't hit the closed log handle.
            sys.stdout = sys.__stdout__
            _log_handle.flush()
            _log_handle.close()

        atexit.register(_close_log)
        sys.stdout = _Tee(sys.__stdout__, _log_handle)
        print(f"[log] Console output is also saved to: {os.path.abspath(log_path)}")

    print("=" * 70)
    print("AMR-M4GN Preprocessing Visualization")
    print("=" * 70)
    
    # ---- Load data ----
    print(f"\n[1/6] Loading case {args.case_idx} (timestep {args.timestep}) "
          f"from {args.data_dir} ({args.split})...")
    data = load_single_case(args.data_dir, args.split, args.case_idx, args.timestep)
    
    pos = data["mesh_pos"]
    cells = data["cells"]
    node_type = data["node_type"]
    edge_index = data["edge_index"]
    velocity = data["velocity"]
    num_nodes = data["num_nodes"]
    
    print(f"  Nodes: {num_nodes}, Cells: {cells.shape[0]}, "
          f"Edges: {edge_index.shape[1]}")
    print(f"  Node types: {dict(zip(*np.unique(node_type, return_counts=True)))}")
    # Print coordinate range (design doc decision gate D2 / uncertainty U1):
    # confirms whether different cases share a coordinate system / scale, which
    # determines if AMR thresholds can be absolute or must be per-case Top-r.
    print(f"  pos x-range: [{pos[:, 0].min():.4f}, {pos[:, 0].max():.4f}], "
          f"y-range: [{pos[:, 1].min():.4f}, {pos[:, 1].max():.4f}]")
    
    # ---- Plot mesh ----
    print(f"\n[2/6] Plotting mesh structure...")
    plot_mesh(pos, cells, node_type,
              os.path.join(args.output_dir, "01_mesh.png"))
    plot_velocity_field(pos, cells, velocity,
                       os.path.join(args.output_dir, "02_velocity.png"))
    
    # ---- Modal decomposition ----
    print(f"\n[3/6] Computing Laplacian eigenmodes (m={args.num_modes})...")
    f_md, eigvals = laplacian_eigenmodes(
        edge_index=edge_index,
        pos=pos,
        node_type=torch.tensor(node_type, dtype=torch.long),
        cells=cells if args.use_cotangent else None,
        num_modes=args.num_modes,
        use_cotangent=args.use_cotangent,
        boundary_type=args.boundary_type,
    )
    f_md_np = f_md.numpy()
    eigvals_np = eigvals.numpy()
    print(f"  Eigenvalues: {eigvals_np}")
    
    plot_eigenmodes(pos, cells, f_md_np, eigvals_np,
                    os.path.join(args.output_dir, "03_eigenmodes.png"))
    
    # ---- Obstacle distance ----
    print(f"\n[4/6] Computing obstacle distance field...")
    # Probe wall/obstacle node types in priority order:
    #   6 = WALL_BOUNDARY (cylinder + channel walls, the usual case),
    #   1 = OBSTACLE (some datasets). NOTE: 5 is OUTFLOW, 4 is INFLOW -> NOT walls.
    f_obs = None
    for obs_id in [6, 1]:
        if np.any(node_type == obs_id):
            f_obs = compute_obstacle_distance(pos, node_type, obstacle_type_id=obs_id)
            print(f"  Found wall/obstacle nodes with type={obs_id} "
                  f"(count={np.sum(node_type == obs_id)}). "
                  f"NOTE: cylinder & channel walls share type 6 (uncertainty U1).")
            break

    if f_obs is None:
        print("  No wall/obstacle nodes found. Using distance to mesh center as f_obs.")
        center = pos.mean(axis=0)
        f_obs = np.linalg.norm(pos - center, axis=1)
    
    plot_obstacle_distance(pos, cells, f_obs,
                           os.path.join(args.output_dir, "04_obstacle_dist.png"))

    # ---- AMR physical indicators (M2, optional) ----
    quantities = None
    if args.plot_physics or args.plot_routing:
        print(f"\n[4b] Computing AMR physical indicators (G/omega/M/S)...")
        # velocity from load_single_case is PHYSICAL (raw TFRecord) -> no denorm.
        area = compute_node_area(pos, cells, num_nodes)
        quantities = compute_ns_quantities(
            u=velocity[:, 0], v=velocity[:, 1],
            pos=pos, edge_index=edge_index, area=area,
        )
        # Print magnitudes for decision gate D1 (physical scale sanity check).
        for k in ["G", "omega", "M", "S"]:
            arr = quantities[k].numpy()
            print(f"  {k:5s}: min={arr.min():.3e}, max={arr.max():.3e}, "
                  f"|.|p99={np.percentile(np.abs(arr), 99):.3e}")
        if args.plot_physics:
            plot_physics_fields(
                pos, cells, {k: v.numpy() for k, v in quantities.items()},
                os.path.join(args.output_dir, "07_physics_fields.png"))

    # ---- Build partition tree ----
    print(f"\n[5/6] Building partition tree (K0={args.K0}, K1={args.K1}, tau={args.tau})...")
    partition_levels, segment_adjacency = build_partition_tree(
        edge_index=edge_index,
        pos=pos,
        f_md=f_md,
        f_obs=torch.tensor(f_obs, dtype=torch.float32),
        K_list=(args.K0, args.K1),
        tau=args.tau,
        max_iter=10,
    )
    
    L0_assign = partition_levels[0].numpy()
    L1_assign = partition_levels[1].numpy()
    L0_adj = segment_adjacency[0]
    L1_adj = segment_adjacency[1]
    
    K0_actual = int(L0_assign.max()) + 1
    K1_actual = int(L1_assign.max()) + 1
    print(f"  Level 0: {K0_actual} segments (target {args.K0})")
    print(f"  Level 1: {K1_actual} segments (target {args.K1})")
    print(f"  L0 adjacency edges: {L0_adj.shape[1] // 2}")
    print(f"  L1 adjacency edges: {L1_adj.shape[1] // 2}")
    
    # ---- Plot partitions ----
    print(f"\n[6/6] Plotting partitions...")
    plot_partition(pos, cells, L0_assign, "Level 0 (Coarse)", args.K0,
                   os.path.join(args.output_dir, "05_partition_L0.png"),
                   seg_adjacency=L0_adj)
    plot_partition(pos, cells, L1_assign, "Level 1 (Fine)", args.K1,
                   os.path.join(args.output_dir, "06_partition_L1.png"),
                   seg_adjacency=None)  # Too many edges to draw for L1
    
    # ---- AMR token routing (M3, optional) ----
    if args.plot_routing:
        print(f"\n[6b] Running AMR token router (fold/keep)...")
        L1 = partition_levels[1]
        # Demo thresholds: per-channel percentile of the L1 per-segment |phys|.
        # (Real thresholds are decision gate D3 / train-time sampling.)
        agg = aggregate_per_segment(quantities, L1, K1_actual)
        thresholds = {
            k: float(torch.quantile(agg[k], args.route_pct / 100.0))
            for k in ["G", "omega", "M", "S"]
        }
        kept_assign, kept_depth, T, _ = route(
            partition_levels, quantities, thresholds)
        n_fine = int((kept_depth == 1).sum())
        n_coarse = int((kept_depth == 0).sum())
        # Decision gate D3 statistics:
        print(f"  Demo thresholds (p{args.route_pct:g} of per-seg |phys|): "
              + ", ".join(f"{k}={v:.3e}" for k, v in thresholds.items()))
        print(f"  Tokens T = {T}  (range [{K0_actual}, {K1_actual}])")
        print(f"    kept fine (L1, active) : {n_fine}")
        print(f"    folded coarse (L0)     : {n_coarse}")
        print(f"    reduction vs all-fine  : "
              f"{100.0 * (K1_actual - T) / K1_actual:.1f}%")
        plot_routing(pos, cells, kept_depth.numpy()[kept_assign.numpy()],
                     kept_assign.numpy(), T,
                     os.path.join(args.output_dir, "08_routing.png"))

    # ---- Save partition data for later use ----
    cache_path = os.path.join(args.output_dir, "partition_cache.pt")
    torch.save({
        "partition_levels": partition_levels,
        "segment_adjacency": segment_adjacency,
        "f_md": f_md,
        "f_obs": torch.tensor(f_obs, dtype=torch.float32),
        "eigvals": eigvals,
        "pos": torch.tensor(pos, dtype=torch.float32),
        "cells": torch.tensor(cells, dtype=torch.long),
        "node_type": torch.tensor(node_type, dtype=torch.long),
        "edge_index": edge_index,
    }, cache_path)
    print(f"\n  Cached partition data to: {cache_path}")
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mesh:           {num_nodes} nodes, {cells.shape[0]} triangles")
    print(f"  Eigenmodes:     {args.num_modes} modes, λ_1..λ_m = "
          f"[{eigvals_np[0]:.4f}, ..., {eigvals_np[-1]:.4f}]")
    print(f"  Partition L0:   {K0_actual} segments")
    print(f"  Partition L1:   {K1_actual} segments")
    print(f"  AMR token range: [{K0_actual}, {K1_actual}]")
    print(f"  Output dir:     {os.path.abspath(args.output_dir)}")
    print(f"\nAll visualizations saved. Check the output directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
