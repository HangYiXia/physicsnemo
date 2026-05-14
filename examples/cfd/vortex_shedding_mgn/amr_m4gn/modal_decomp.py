"""
modal_decomp.py — Laplacian Eigenfunctions for Fluid Mesh
==========================================================
Computes the first m eigenmodes of the mesh Laplacian operator.
These modes capture the geometry- and boundary-driven harmonic structures
of the flow domain, serving as physics-aware features for mesh segmentation.

References:
    - M4GN (Lei et al., TMLR 2025), Section 2.3: Laplacian Eigenfunctions
    - Grebenkov & Nguyen, "Geometrical structure of Laplacian eigenfunctions",
      SIAM Review, 2013.

Usage:
    from amr_m4gn.modal_decomp import laplacian_eigenmodes
    f_md = laplacian_eigenmodes(edge_index, pos, node_type, num_modes=6)
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh


def graph_laplacian(edge_index, num_nodes, pos=None):
    """
    Build the (unnormalized) graph Laplacian: L = D - A
    
    This is a simple approach suitable for any graph topology.
    For triangle meshes, use cotangent_laplacian() for better physical fidelity.
    
    Parameters
    ----------
    edge_index : Tensor [2, E]
        COO edge index (undirected, may contain duplicates)
    num_nodes : int
        Total number of nodes
    pos : Tensor [N, 2], optional
        Node positions (unused here, for API compatibility)
    
    Returns
    -------
    L : scipy.sparse.csr_matrix [N, N]
        Graph Laplacian matrix
    M : scipy.sparse.csr_matrix [N, N]
        Mass matrix (identity for graph Laplacian)
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    
    # Remove self-loops and duplicates
    mask = src != dst
    src, dst = src[mask], dst[mask]
    
    # Build adjacency matrix (symmetric)
    ones = np.ones(len(src), dtype=np.float64)
    A = coo_matrix((ones, (src, dst)), shape=(num_nodes, num_nodes))
    A = A.tocsr()
    # Ensure symmetry and remove duplicate entries
    A = (A + A.T) / 2
    A.data[:] = np.where(A.data > 0, 1.0, 0.0)
    
    # Degree matrix
    degree = np.array(A.sum(axis=1)).flatten()
    D = diags(degree)
    
    # Laplacian
    L = D - A
    L = L.tocsr()
    
    # Mass matrix = Identity (uniform node weight)
    M = eye(num_nodes, format='csr')
    
    return L, M


def cotangent_laplacian(pos, cells, num_nodes):
    """
    Build the cotangent (FEM) Laplacian for 2D triangle meshes.
    
    This provides a better discretization of -nabla^2 and yields
    eigenmodes that better represent continuous harmonic functions.
    
    Parameters
    ----------
    pos : ndarray [N, 2]
        Node positions
    cells : ndarray [num_cells, 3]
        Triangle connectivity (vertex indices)
    num_nodes : int
        Total number of nodes
    
    Returns
    -------
    L : scipy.sparse.csr_matrix [N, N]
        Cotangent stiffness matrix
    M : scipy.sparse.csr_matrix [N, N]
        Lumped mass matrix (diagonal, area-weighted)
    """
    num_cells = cells.shape[0]
    
    # Preallocate triplets
    rows, cols, vals_L = [], [], []
    area_per_node = np.zeros(num_nodes, dtype=np.float64)
    
    for tri_idx in range(num_cells):
        i, j, k = cells[tri_idx]
        pi, pj, pk = pos[i], pos[j], pos[k]
        
        # Edge vectors
        eij = pj - pi  # edge i->j
        eik = pk - pi  # edge i->k
        ejk = pk - pj  # edge j->k
        
        # Triangle area (2D cross product / 2)
        area = 0.5 * abs(eij[0] * eik[1] - eij[1] * eik[0])
        if area < 1e-16:
            continue
        
        # Cotangent weights for each edge
        # For edge opposite to vertex i: cot(angle at i)
        # angle at i: between edges ij and ik
        dot_i = np.dot(eij, eik)
        cot_i = dot_i / (2.0 * area)
        
        # angle at j: between edges ji and jk
        eji = -eij
        dot_j = np.dot(eji, ejk)
        cot_j = dot_j / (2.0 * area)
        
        # angle at k: between edges ki and kj
        eki = -eik
        ekj = -ejk
        dot_k = np.dot(eki, ekj)
        cot_k = dot_k / (2.0 * area)
        
        # Stiffness contributions (off-diagonal)
        # Edge jk (opposite vertex i): weight = cot_i / 2
        # Edge ik (opposite vertex j): weight = cot_j / 2
        # Edge ij (opposite vertex k): weight = cot_k / 2
        edges_weights = [
            (j, k, cot_i / 2.0),
            (i, k, cot_j / 2.0),
            (i, j, cot_k / 2.0),
        ]
        
        for (a, b, w) in edges_weights:
            # Off-diagonal: -w
            rows.extend([a, b])
            cols.extend([b, a])
            vals_L.extend([-w, -w])
            # Diagonal: +w for both a and b
            rows.extend([a, b])
            cols.extend([a, b])
            vals_L.extend([w, w])
        
        # Lumped mass matrix: area/3 per vertex
        area_per_node[i] += area / 3.0
        area_per_node[j] += area / 3.0
        area_per_node[k] += area / 3.0
    
    # Assemble sparse matrix
    L = coo_matrix(
        (np.array(vals_L), (np.array(rows), np.array(cols))),
        shape=(num_nodes, num_nodes)
    ).tocsr()
    
    # Mass matrix (lumped, diagonal)
    # Clamp to avoid zero mass for isolated nodes
    area_per_node = np.maximum(area_per_node, 1e-12)
    M = diags(area_per_node, format='csr')
    
    return L, M


def laplacian_eigenmodes(
    edge_index,
    pos,
    node_type=None,
    cells=None,
    num_modes=6,
    use_cotangent=True,
    boundary_type="neumann",
):
    """
    Compute the first m eigenmodes of the mesh Laplacian.
    
    For fluid domains, Laplacian eigenfunctions capture geometry-driven
    harmonic structures (analogous to vibration modes for solids).
    
    Parameters
    ----------
    edge_index : Tensor [2, E]
        Edge index in COO format
    pos : Tensor or ndarray [N, 2]
        Node positions
    node_type : Tensor [N] or None
        Integer node type (0=fluid, others=boundary).
        If boundary_type="dirichlet", boundary nodes get fixed eigenvectors.
    cells : ndarray [num_cells, 3] or None
        Triangle cells. If provided and use_cotangent=True, uses FEM Laplacian.
    num_modes : int
        Number of eigenmodes to compute (default: 6)
    use_cotangent : bool
        Whether to use cotangent Laplacian (requires cells). Default True.
    boundary_type : str
        "neumann" (free BC, default) or "dirichlet" (fix boundary nodes to 0)
    
    Returns
    -------
    f_md : Tensor [N, num_modes]
        Modal decomposition features per node
    eigvals : Tensor [num_modes]
        Corresponding eigenvalues (frequencies squared)
    """
    if isinstance(pos, torch.Tensor):
        pos_np = pos.numpy()
    else:
        pos_np = np.asarray(pos)
    
    if isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index
    else:
        edge_index_np = torch.tensor(edge_index)
    
    num_nodes = pos_np.shape[0]
    
    # Build Laplacian
    if use_cotangent and cells is not None:
        cells_np = np.asarray(cells)
        L, M = cotangent_laplacian(pos_np, cells_np, num_nodes)
    else:
        L, M = graph_laplacian(edge_index_np, num_nodes)
    
    # Handle boundary conditions
    if boundary_type == "dirichlet" and node_type is not None:
        if isinstance(node_type, torch.Tensor):
            nt = node_type.numpy().flatten()
        else:
            nt = np.asarray(node_type).flatten()
        
        # Boundary nodes: node_type != 0
        boundary_mask = nt != 0
        interior_mask = ~boundary_mask
        interior_idx = np.where(interior_mask)[0]
        
        if len(interior_idx) < num_modes + 1:
            print(f"Warning: too few interior nodes ({len(interior_idx)}) "
                  f"for {num_modes} modes. Using Neumann BC instead.")
            boundary_type = "neumann"
        else:
            # Restrict L and M to interior nodes
            L_int = L[np.ix_(interior_idx, interior_idx)]
            M_int = M[np.ix_(interior_idx, interior_idx)]
            
            # Solve eigenvalue problem on interior
            # Request num_modes+1 because the first might be trivial
            k_request = min(num_modes + 1, len(interior_idx) - 1)
            eigvals_int, eigvecs_int = eigsh(
                L_int, k=k_request, M=M_int, which='SM', sigma=0,
                tol=1e-6, maxiter=5000
            )
            
            # Skip the trivial (zero) mode
            start = 1 if abs(eigvals_int[0]) < 1e-8 else 0
            eigvals_sel = eigvals_int[start:start + num_modes]
            eigvecs_sel = eigvecs_int[:, start:start + num_modes]
            
            # Scatter back to full node set (boundary nodes get 0)
            eigvecs_full = np.zeros((num_nodes, num_modes), dtype=np.float64)
            eigvecs_full[interior_idx, :eigvecs_sel.shape[1]] = eigvecs_sel
            
            # Normalize each mode to unit norm
            for m in range(eigvecs_full.shape[1]):
                norm = np.linalg.norm(eigvecs_full[:, m])
                if norm > 1e-12:
                    eigvecs_full[:, m] /= norm
            
            f_md = torch.tensor(eigvecs_full, dtype=torch.float32)
            eigvals_out = torch.tensor(eigvals_sel, dtype=torch.float32)
            return f_md, eigvals_out
    
    # Neumann BC: solve on entire domain
    k_request = min(num_modes + 1, num_nodes - 1)
    try:
        eigvals_all, eigvecs_all = eigsh(
            L, k=k_request, M=M, which='SM', sigma=0,
            tol=1e-6, maxiter=5000
        )
    except Exception:
        # Fallback: use shift-invert without sigma
        eigvals_all, eigvecs_all = eigsh(
            L, k=k_request, M=M, which='SM',
            tol=1e-4, maxiter=10000
        )
    
    # Skip the trivial zero-eigenvalue mode (constant mode)
    start = 1 if abs(eigvals_all[0]) < 1e-8 else 0
    eigvals_sel = eigvals_all[start:start + num_modes]
    eigvecs_sel = eigvecs_all[:, start:start + num_modes]
    
    # Normalize each mode
    for m in range(eigvecs_sel.shape[1]):
        norm = np.linalg.norm(eigvecs_sel[:, m])
        if norm > 1e-12:
            eigvecs_sel[:, m] /= norm
    
    f_md = torch.tensor(eigvecs_sel, dtype=torch.float32)
    eigvals_out = torch.tensor(eigvals_sel, dtype=torch.float32)
    
    return f_md, eigvals_out


def compute_node_area(pos, cells, num_nodes):
    """
    Compute the Voronoi dual area per node from triangle cells.
    Uses lumped mass: area_i = sum(area_tri / 3) for all triangles containing i.
    
    Parameters
    ----------
    pos : ndarray [N, 2]
    cells : ndarray [num_cells, 3]
    num_nodes : int
    
    Returns
    -------
    area : Tensor [N]
    """
    pos_np = np.asarray(pos)
    cells_np = np.asarray(cells)
    area_per_node = np.zeros(num_nodes, dtype=np.float64)
    
    for tri_idx in range(cells_np.shape[0]):
        i, j, k = cells_np[tri_idx]
        pi, pj, pk = pos_np[i], pos_np[j], pos_np[k]
        eij = pj - pi
        eik = pk - pi
        area = 0.5 * abs(eij[0] * eik[1] - eij[1] * eik[0])
        area_per_node[i] += area / 3.0
        area_per_node[j] += area / 3.0
        area_per_node[k] += area / 3.0
    
    return torch.tensor(area_per_node, dtype=torch.float32)
