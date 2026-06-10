# AMR-M4GN: Adaptive Mesh Refinement + Multi-segment Hierarchical Graph Network
# for vortex shedding simulation

from .modal_decomp import laplacian_eigenmodes, graph_laplacian, cotangent_laplacian
from .segmentation import hybrid_segmentation, build_partition_tree
from .physics_ops import (
    compute_ns_quantities,
    lstsq_gradient,
    virtual_step,
    denormalize_velocity,
)
