"""
physics_ops.py — Navier-Stokes Constraint-Aware Physical Quantities
====================================================================
Computes per-node physical indicators used by the AMR Router (M3) to decide
which mesh segments must stay fine (active) and which can be merged (calm):

    G  : velocity-gradient magnitude  = sqrt(||grad u||^2 + ||grad v||^2)
    omega : vorticity                 = dv/dx - du/dy
    S  : strain-rate magnitude        = sqrt(2 S_ij S_ij), S_ij = sym(grad U)
    M  : momentum indicator           = rho * |U| * area

Note on S: the AMR-Transformer paper defines the KH-shear as (du/dy - dv/dx),
which is mathematically identical to -omega and therefore redundant once the
router thresholds on magnitude. We instead use the strain-rate magnitude
sqrt(2 S_ij S_ij) (symmetric part of grad U), a standard, vorticity-INDEPENDENT
refinement indicator that is always >= 0. This makes the four indicators
genuinely independent (see Design Doc 4.6 note).

Gradients on the *unstructured triangle mesh* are estimated with a 1-ring
weighted least-squares fit (no structured-grid / quadtree assumption), so the
operators work directly on the vortex-shedding mesh.

References:
    - Design Doc 4.6 (N-S Constraint-Aware Physical Quantity Operators)
    - AMR-Transformer (Xu et al., CVPR 2025), Sec. 3.2, Eqs. 3-5, 11
      (velocity gradient, vorticity, momentum, KH shear, virtual step)

IMPORTANT (decision gate D1 / uncertainty U4):
    `u, v` may be PHYSICAL (raw, as read from TFRecord by visualize_partition)
    or NORMALIZED (graph.x[:, :2] at train time, mean-0/std-1). Physical
    quantities are only physically meaningful on PHYSICAL velocity. Pass
    `vel_mean` / `vel_std` (from node_stats.json) to denormalize first.
    See `compute_ns_quantities(..., vel_mean=, vel_std=)`.

Usage:
    from amr_m4gn.physics_ops import compute_ns_quantities
    q = compute_ns_quantities(u, v, pos, edge_index, area=area)
    omega = q["omega"]            # [N]
"""

import numpy as np
import torch


def _as_float_tensor(x):
    """Convert ndarray / tensor / list to a contiguous float32 torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def lstsq_gradient(field, pos, edge_index, eps=1e-8):
    """
    1-ring weighted least-squares gradient on an unstructured mesh.

    For each node i with neighbors j (from edges i->j), we fit a local linear
    model  f_j - f_i ~= grad_f_i . (x_j - x_i)  by solving the 2x2 normal
    equations  (sum dx dx^T) grad = (sum dx df)  per node.

    Parameters
    ----------
    field : Tensor/ndarray [N, C]
        Scalar fields stacked along the last dim (e.g. [u, v] -> C=2).
    pos : Tensor/ndarray [N, 2]
        Node coordinates.
    edge_index : Tensor/ndarray [2, E]
        COO edges (undirected => both directions present). Node `edge_index[0]`
        is treated as the center, `edge_index[1]` as its neighbor.
    eps : float
        Tikhonov regularization added to the 2x2 normal matrix to stay stable
        when a node's neighbors are (near-)collinear or too few.

    Returns
    -------
    grad : Tensor [N, C, 2]
        grad[:, c, 0] = d field_c / dx,  grad[:, c, 1] = d field_c / dy.
    """
    field = _as_float_tensor(field)
    if field.ndim == 1:
        field = field[:, None]
    pos = _as_float_tensor(pos)
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.long()
    else:
        ei = torch.as_tensor(np.asarray(edge_index), dtype=torch.long)

    N, C = field.shape
    ei = ei.to(field.device)
    i_idx, j_idx = ei[0], ei[1]

    dx = pos[j_idx] - pos[i_idx]            # [E, 2]
    df = field[j_idx] - field[i_idx]        # [E, C]

    # Per-edge contributions to the normal equations.
    outer_xx = dx[:, :, None] * dx[:, None, :]   # [E, 2, 2]
    outer_xf = dx[:, :, None] * df[:, None, :]   # [E, 2, C]

    A = torch.zeros(N, 2, 2, dtype=field.dtype, device=field.device)
    B = torch.zeros(N, 2, C, dtype=field.dtype, device=field.device)
    A.index_add_(0, i_idx, outer_xx)
    B.index_add_(0, i_idx, outer_xf)

    # Regularize and solve A @ grad = B  (grad: [N, 2, C]).
    A = A + eps * torch.eye(2, dtype=field.dtype, device=field.device)[None]
    grad = torch.linalg.solve(A, B)         # [N, 2, C]

    return grad.transpose(1, 2).contiguous()  # [N, C, 2]


def denormalize_velocity(u, v, vel_mean, vel_std):
    """Map normalized velocity back to physical: phys = norm * std + mean.

    vel_mean / vel_std : length-2 sequences [mean_u, mean_v] / [std_u, std_v]
    (read from node_stats.json at train time). Returns (u_phys, v_phys).
    """
    u = _as_float_tensor(u)
    v = _as_float_tensor(v)
    vm = _as_float_tensor(vel_mean).flatten()
    vs = _as_float_tensor(vel_std).flatten()
    u_phys = u * vs[0] + vm[0]
    v_phys = v * vs[1] + vm[1]
    return u_phys, v_phys


def virtual_step(uv_t, uv_prev):
    """
    Forward-Euler virtual velocity field (AMR-Transformer Eq. 11):
        uv' = uv_t + (uv_t - uv_prev)
    Used to pre-refine regions that are *about to* become active. If uv_prev is
    None, returns uv_t unchanged (no history available, e.g. first frame).
    """
    uv_t = _as_float_tensor(uv_t)
    if uv_prev is None:
        return uv_t
    uv_prev = _as_float_tensor(uv_prev)
    return uv_t + (uv_t - uv_prev)


def compute_ns_quantities(
    u, v, pos, edge_index,
    area=None, rho=1.0, eps=1e-8,
    vel_mean=None, vel_std=None,
):
    """
    Compute the four AMR physical indicators per node.

    Parameters
    ----------
    u, v : Tensor/ndarray [N]
        Velocity components (PHYSICAL, or NORMALIZED + vel_mean/vel_std).
    pos : Tensor/ndarray [N, 2]
    edge_index : Tensor/ndarray [2, E]
    area : Tensor/ndarray [N] or None
        Per-node area (e.g. modal_decomp.compute_node_area). Defaults to ones,
        so M reduces to |U| when area is unavailable.
    rho : float
        Fluid density (default 1.0).
    vel_mean, vel_std : optional length-2
        If given, denormalize (u, v) before computing quantities (D1 / U4).

    Returns
    -------
    dict with keys "G", "omega", "M", "S", each a Tensor [N].
    """
    u = _as_float_tensor(u).flatten()
    v = _as_float_tensor(v).flatten()
    if vel_mean is not None and vel_std is not None:
        u, v = denormalize_velocity(u, v, vel_mean, vel_std)

    field = torch.stack([u, v], dim=1)            # [N, 2]
    grad = lstsq_gradient(field, pos, edge_index, eps=eps)  # [N, 2, 2]

    du_dx, du_dy = grad[:, 0, 0], grad[:, 0, 1]
    dv_dx, dv_dy = grad[:, 1, 0], grad[:, 1, 1]

    G = torch.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2 + 1e-30)
    omega = dv_dx - du_dy
    # Strain-rate magnitude S = sqrt(2 S_ij S_ij), with S_ij the symmetric part
    # of grad(U). In 2D this expands to:
    #   2 S_ij S_ij = 2 du_dx^2 + 2 dv_dy^2 + (du_dy + dv_dx)^2
    # Unlike the paper's KH shear (du/dy - dv/dx, which equals -omega and is thus
    # redundant), the strain magnitude is independent of vorticity (it uses the
    # SYMMETRIC part du/dy + dv/dx) and is always >= 0, like G and M.
    S = torch.sqrt(2 * du_dx**2 + 2 * dv_dy**2 + (du_dy + dv_dx)**2 + 1e-30)

    if area is None:
        area_t = torch.ones_like(u)
    else:
        area_t = _as_float_tensor(area).flatten()
    M = rho * torch.sqrt(u**2 + v**2 + 1e-30) * area_t

    return {"G": G, "omega": omega, "M": M, "S": S}
