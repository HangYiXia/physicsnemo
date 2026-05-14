"""
segmentation.py — Hybrid Mesh-Graph Segmentation (METIS + SLIC)
================================================================
Two-stage mesh partitioning strategy from M4GN (Lei et al., TMLR 2025):
  Stage 1: METIS graph partitioner for initial coarse segmentation
  Stage 2: SLIC superpixel-style refinement guided by modal decomposition features

Additionally provides recursive subdivision to build a 2-level partition tree
for AMR token routing.

References:
    - M4GN, Section 2.4: Hybrid Mesh-Graph Segmentation
    - Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for
      Partitioning Irregular Graphs", SIAM J. Sci. Comput., 1998.
    - Achanta et al., "SLIC Superpixels", IEEE TPAMI, 2012.

Usage:
    from amr_m4gn.segmentation import build_partition_tree
    levels = build_partition_tree(edge_index, pos, f_md, f_obs, K_list=[64, 256])
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix

# Try importing pymetis; fall back to spectral partitioning if not available
try:
    import pymetis
    HAS_PYMETIS = True
except ImportError:
    HAS_PYMETIS = False
    print("[segmentation.py] Warning: pymetis not found. "
          "Using fallback spectral partitioning (slower, lower quality). "
          "Install pymetis: pip install pymetis")


# ============================================================================
# Stage 1: Graph Partitioning (METIS or fallback)
# ============================================================================

def _edge_index_to_adjacency_list(edge_index, num_nodes):
    """Convert PyG edge_index [2, E] to list-of-lists adjacency format for pymetis."""
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        if s != d:  # skip self-loops
            adj[int(s)].append(int(d))
    
    # Remove duplicates (pymetis expects unique neighbors)
    adj = [list(set(neighbors)) for neighbors in adj]
    return adj


def metis_partition(edge_index, num_nodes, num_parts):
    """
    Partition a graph into num_parts segments using METIS.
    
    Parameters
    ----------
    edge_index : Tensor [2, E]
    num_nodes : int
    num_parts : int
    
    Returns
    -------
    assign : ndarray [N] with values in [0, num_parts)
    """
    if not HAS_PYMETIS:
        return _spectral_partition_fallback(edge_index, num_nodes, num_parts)
    
    adj = _edge_index_to_adjacency_list(edge_index, num_nodes)
    
    # pymetis.part_graph expects adjacency as list of lists
    try:
        _, membership = pymetis.part_graph(num_parts, adjacency=adj)
        return np.array(membership, dtype=np.int64)
    except Exception as e:
        print(f"[METIS] Error: {e}. Falling back to spectral partitioning.")
        return _spectral_partition_fallback(edge_index, num_nodes, num_parts)


def _spectral_partition_fallback(edge_index, num_nodes, num_parts):
    """
    Fallback: spectral clustering using scipy.
    Not as good as METIS but works without additional dependencies.
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import diags
    
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
    else:
        src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    
    mask = src != dst
    src, dst = src[mask], dst[mask]
    
    ones = np.ones(len(src))
    A = coo_matrix((ones, (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
    A = (A + A.T) / 2
    A.data[:] = np.where(A.data > 0, 1.0, 0.0)
    
    degree = np.array(A.sum(axis=1)).flatten()
    D = diags(degree)
    L = D - A
    
    # Compute first num_parts eigenvectors of normalized Laplacian
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    k = min(num_parts, num_nodes - 1)
    _, eigvecs = eigsh(L_norm, k=k, which='SM', tol=1e-4)
    
    # K-means on eigenvector embedding
    from scipy.cluster.vq import kmeans2
    _, labels = kmeans2(eigvecs, num_parts, minit='++', iter=50)
    
    return labels.astype(np.int64)


# ============================================================================
# Stage 2: SLIC-style Superpixel Refinement
# ============================================================================

def slic_refinement(
    pos,
    features,
    init_assign,
    num_segments,
    tau=1.0,
    max_iter=10,
    connectivity_constraint=True,
    edge_index=None,
):
    """
    SLIC-style iterative refinement of mesh segmentation.
    
    Updates segment assignments by minimizing a combined distance that
    considers both physics feature similarity and spatial proximity.
    
    Parameters
    ----------
    pos : ndarray [N, 2]
        Node spatial positions
    features : ndarray [N, F]
        Physics-aware features (e.g., modal decomposition + obstacle distance)
    init_assign : ndarray [N]
        Initial segment assignment from METIS
    num_segments : int
        Target number of segments
    tau : float
        Compactness parameter controlling spatial proximity weight.
        Higher tau = rounder, more spatially compact segments.
    max_iter : int
        Maximum SLIC iterations
    connectivity_constraint : bool
        If True, only reassign nodes to segments of adjacent nodes (local search)
    edge_index : Tensor [2, E] or None
        Required if connectivity_constraint=True
    
    Returns
    -------
    assign : ndarray [N] with values in [0, num_segments)
    """
    N = pos.shape[0]
    assign = init_assign.copy()
    
    # Compute average cluster size for local search radius
    S = np.sqrt(
        (pos[:, 0].max() - pos[:, 0].min()) *
        (pos[:, 1].max() - pos[:, 1].min()) / num_segments
    )
    
    # Build neighbor list for connectivity constraint
    if connectivity_constraint and edge_index is not None:
        if isinstance(edge_index, torch.Tensor):
            src_arr = edge_index[0].numpy()
            dst_arr = edge_index[1].numpy()
        else:
            src_arr, dst_arr = edge_index[0], edge_index[1]
        neighbors = [set() for _ in range(N)]
        for s, d in zip(src_arr, dst_arr):
            neighbors[int(s)].add(int(d))
    else:
        neighbors = None
    
    for iteration in range(max_iter):
        # Compute centroids
        centroids_pos = np.zeros((num_segments, 2))
        centroids_feat = np.zeros((num_segments, features.shape[1]))
        counts = np.zeros(num_segments)
        
        for i in range(N):
            seg = assign[i]
            centroids_pos[seg] += pos[i]
            centroids_feat[seg] += features[i]
            counts[seg] += 1
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        centroids_pos /= counts[:, None]
        centroids_feat /= counts[:, None]
        
        # Update assignments
        new_assign = assign.copy()
        changed = 0
        
        for i in range(N):
            # Determine candidate segments
            if neighbors is not None:
                # Local search: only consider segments of self and neighbors
                candidate_segs = {assign[i]}
                for nb in neighbors[i]:
                    candidate_segs.add(assign[nb])
                candidate_segs = list(candidate_segs)
            else:
                # Global search within radius S
                dists_to_all = np.linalg.norm(pos[i] - centroids_pos, axis=1)
                candidate_segs = np.where(dists_to_all <= 2.0 * S)[0].tolist()
                if not candidate_segs:
                    candidate_segs = [assign[i]]
            
            # Find best segment
            best_seg = assign[i]
            best_dist = float('inf')
            
            for seg in candidate_segs:
                d_feat = np.linalg.norm(features[i] - centroids_feat[seg])
                d_pos = np.linalg.norm(pos[i] - centroids_pos[seg])
                d_total = d_feat + tau * d_pos
                if d_total < best_dist:
                    best_dist = d_total
                    best_seg = seg
            
            if best_seg != assign[i]:
                new_assign[i] = best_seg
                changed += 1
        
        assign = new_assign
        
        # Check convergence
        if changed == 0:
            break
    
    # Remap segment IDs to consecutive [0, K')
    assign = _remap_consecutive(assign)
    
    return assign


def _remap_consecutive(assign):
    """Remap arbitrary integer labels to consecutive [0, K)."""
    unique_labels = np.unique(assign)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([mapping[a] for a in assign], dtype=np.int64)


# ============================================================================
# Combined: Hybrid Segmentation
# ============================================================================

def hybrid_segmentation(
    edge_index,
    pos,
    f_md,
    f_obs=None,
    num_segments=64,
    tau=1.0,
    max_iter=10,
):
    """
    Hybrid mesh-graph segmentation: METIS + SLIC refinement.
    
    Parameters
    ----------
    edge_index : Tensor [2, E]
        Graph edges (undirected)
    pos : Tensor or ndarray [N, 2]
        Node positions
    f_md : Tensor or ndarray [N, m]
        Modal decomposition features
    f_obs : Tensor or ndarray [N, 1] or None
        Obstacle distance features (distance to cylinder surface)
    num_segments : int
        Target number of segments (default 64)
    tau : float
        SLIC compactness parameter (default 1.0)
    max_iter : int
        SLIC iterations (default 10)
    
    Returns
    -------
    assign : ndarray [N] with values in [0, K'), K' <= num_segments
    """
    # Convert to numpy
    if isinstance(pos, torch.Tensor):
        pos_np = pos.numpy()
    else:
        pos_np = np.asarray(pos)
    
    if isinstance(f_md, torch.Tensor):
        f_md_np = f_md.numpy()
    else:
        f_md_np = np.asarray(f_md)
    
    num_nodes = pos_np.shape[0]
    
    # Build SLIC features: [f_md, f_obs] (or just f_md if no obstacles)
    if f_obs is not None:
        if isinstance(f_obs, torch.Tensor):
            f_obs_np = f_obs.numpy()
        else:
            f_obs_np = np.asarray(f_obs)
        if f_obs_np.ndim == 1:
            f_obs_np = f_obs_np[:, None]
        features = np.concatenate([f_md_np, f_obs_np], axis=1)
    else:
        features = f_md_np
    
    # Normalize features to [0, 1] range for balanced distance computation
    feat_min = features.min(axis=0, keepdims=True)
    feat_max = features.max(axis=0, keepdims=True)
    feat_range = feat_max - feat_min
    feat_range = np.where(feat_range < 1e-12, 1.0, feat_range)
    features_norm = (features - feat_min) / feat_range
    
    # Also normalize positions
    pos_min = pos_np.min(axis=0, keepdims=True)
    pos_max = pos_np.max(axis=0, keepdims=True)
    pos_range = pos_max - pos_min
    pos_range = np.where(pos_range < 1e-12, 1.0, pos_range)
    pos_norm = (pos_np - pos_min) / pos_range
    
    # Stage 1: METIS partition
    init_assign = metis_partition(edge_index, num_nodes, num_segments)
    
    # Stage 2: SLIC refinement
    assign = slic_refinement(
        pos=pos_norm,
        features=features_norm,
        init_assign=init_assign,
        num_segments=num_segments,
        tau=tau,
        max_iter=max_iter,
        connectivity_constraint=True,
        edge_index=edge_index,
    )
    
    return assign


# ============================================================================
# Multi-level Partition Tree
# ============================================================================

def build_partition_tree(
    edge_index,
    pos,
    f_md,
    f_obs=None,
    K_list=(64, 256),
    tau=1.0,
    max_iter=10,
):
    """
    Build a hierarchical partition tree with multiple levels.
    
    Level 0 (coarse): K_list[0] segments via hybrid_segmentation on full graph
    Level 1 (fine):   K_list[1] segments via further subdividing each L0 segment
    
    Parameters
    ----------
    edge_index : Tensor [2, E]
    pos : Tensor or ndarray [N, 2]
    f_md : Tensor or ndarray [N, m]
    f_obs : Tensor or ndarray [N, 1] or None
    K_list : tuple of int
        Number of segments at each level (e.g., (64, 256))
    tau : float
        SLIC compactness
    max_iter : int
        SLIC iterations per level
    
    Returns
    -------
    partition_levels : list of Tensor [N]
        Each tensor maps node -> segment_id at that level.
        partition_levels[0] has values in [0, K0),
        partition_levels[1] has values in [0, K1).
    segment_adjacency : list of Tensor [2, E_seg]
        Edge index at segment level (which segments are neighbors)
    """
    if isinstance(pos, torch.Tensor):
        pos_np = pos.numpy()
    else:
        pos_np = np.asarray(pos)
    
    if isinstance(f_md, torch.Tensor):
        f_md_np = f_md.numpy()
    else:
        f_md_np = np.asarray(f_md)
    
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index
    else:
        ei = torch.tensor(edge_index)
    
    num_nodes = pos_np.shape[0]
    
    # Level 0: coarse partition
    K0 = K_list[0]
    L0_assign = hybrid_segmentation(
        ei, pos_np, f_md_np, f_obs, num_segments=K0, tau=tau, max_iter=max_iter
    )
    
    # Level 1: fine partition by subdividing each L0 segment
    K1 = K_list[1]
    sub_K = max(K1 // K0, 2)  # subdivisions per L0 segment (default: 4)
    
    L1_assign = np.zeros(num_nodes, dtype=np.int64)
    offset = 0
    
    for seg_id in range(int(L0_assign.max()) + 1):
        mask = L0_assign == seg_id
        node_indices = np.where(mask)[0]
        
        if len(node_indices) < sub_K:
            # Too few nodes to subdivide; keep as one segment
            L1_assign[node_indices] = offset
            offset += 1
            continue
        
        # Extract subgraph
        sub_edge_index, sub_pos, sub_f_md, sub_f_obs = _extract_subgraph(
            ei, pos_np, f_md_np, f_obs, node_indices
        )
        
        # Partition the subgraph
        sub_num_nodes = len(node_indices)
        actual_sub_K = min(sub_K, sub_num_nodes)
        
        if sub_edge_index.shape[1] > 0 and sub_num_nodes > actual_sub_K:
            sub_assign = metis_partition(
                sub_edge_index, sub_num_nodes, actual_sub_K
            )
        else:
            sub_assign = np.zeros(sub_num_nodes, dtype=np.int64)
        
        # Map back to global indices
        for local_idx, global_idx in enumerate(node_indices):
            L1_assign[global_idx] = sub_assign[local_idx] + offset
        
        offset += int(sub_assign.max()) + 1
    
    # Build segment-level adjacency for each level
    L0_adj = _build_segment_adjacency(ei, L0_assign)
    L1_adj = _build_segment_adjacency(ei, L1_assign)
    
    partition_levels = [
        torch.tensor(L0_assign, dtype=torch.long),
        torch.tensor(L1_assign, dtype=torch.long),
    ]
    segment_adjacency = [L0_adj, L1_adj]
    
    return partition_levels, segment_adjacency


def _extract_subgraph(edge_index, pos, f_md, f_obs, node_indices):
    """Extract a subgraph given a set of node indices."""
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    # Create global -> local mapping
    node_set = set(node_indices.tolist())
    global_to_local = {g: l for l, g in enumerate(node_indices)}
    
    # Filter edges: both endpoints must be in node_indices
    sub_src, sub_dst = [], []
    for s, d in zip(src, dst):
        if int(s) in node_set and int(d) in node_set:
            sub_src.append(global_to_local[int(s)])
            sub_dst.append(global_to_local[int(d)])
    
    if len(sub_src) > 0:
        sub_edge_index = torch.tensor([sub_src, sub_dst], dtype=torch.long)
    else:
        sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    sub_pos = pos[node_indices]
    sub_f_md = f_md[node_indices]
    
    if f_obs is not None:
        if isinstance(f_obs, torch.Tensor):
            sub_f_obs = f_obs[node_indices]
        elif isinstance(f_obs, np.ndarray):
            sub_f_obs = f_obs[node_indices]
        else:
            sub_f_obs = None
    else:
        sub_f_obs = None
    
    return sub_edge_index, sub_pos, sub_f_md, sub_f_obs


def _build_segment_adjacency(edge_index, assign):
    """
    Build segment-level adjacency: two segments are adjacent if any
    edge in the original graph connects nodes from different segments.
    
    Returns
    -------
    seg_edge_index : Tensor [2, E_seg]
        Segment-level edge index (undirected, no self-loops)
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
    else:
        src, dst = edge_index[0], edge_index[1]
    
    seg_edges = set()
    for s, d in zip(src, dst):
        seg_s = assign[int(s)]
        seg_d = assign[int(d)]
        if seg_s != seg_d:
            edge = (min(seg_s, seg_d), max(seg_s, seg_d))
            seg_edges.add(edge)
    
    if len(seg_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    seg_edges = list(seg_edges)
    seg_src = [e[0] for e in seg_edges] + [e[1] for e in seg_edges]
    seg_dst = [e[1] for e in seg_edges] + [e[0] for e in seg_edges]
    
    return torch.tensor([seg_src, seg_dst], dtype=torch.long)


# ============================================================================
# Utility: Compute obstacle distance
# ============================================================================

def compute_obstacle_distance(pos, node_type, obstacle_type_id=5):
    """
    Compute signed distance from each node to the nearest obstacle node.
    
    For vortex_shedding: obstacle nodes are the cylinder surface
    (node_type == 5 in the raw dataset, before one-hot encoding).
    
    Parameters
    ----------
    pos : ndarray [N, 2]
        Node positions
    node_type : ndarray [N]
        Raw integer node type (0=fluid, 5=cylinder wall, etc.)
    obstacle_type_id : int
        Which node_type corresponds to the obstacle (default 5)
    
    Returns
    -------
    f_obs : ndarray [N]
        Distance to nearest obstacle node (always >= 0)
    """
    pos_np = np.asarray(pos)
    nt = np.asarray(node_type).flatten()
    
    obstacle_mask = nt == obstacle_type_id
    obstacle_pos = pos_np[obstacle_mask]
    
    if len(obstacle_pos) == 0:
        # No obstacle nodes found; try to find center and estimate
        # For vortex_shedding, cylinder is approximately at center
        print("[Warning] No obstacle nodes found. Using domain center as fallback.")
        center = pos_np.mean(axis=0)
        return np.linalg.norm(pos_np - center, axis=1)
    
    # Compute distance from each node to nearest obstacle node
    # For efficiency with large meshes, use batch computation
    f_obs = np.zeros(pos_np.shape[0], dtype=np.float64)
    
    # Batch processing to avoid memory issues
    batch_size = 500
    for start in range(0, pos_np.shape[0], batch_size):
        end = min(start + batch_size, pos_np.shape[0])
        chunk = pos_np[start:end]  # [B, 2]
        # Distance to all obstacle nodes
        diffs = chunk[:, None, :] - obstacle_pos[None, :, :]  # [B, M, 2]
        dists = np.linalg.norm(diffs, axis=2)  # [B, M]
        f_obs[start:end] = dists.min(axis=1)  # [B]
    
    return f_obs
