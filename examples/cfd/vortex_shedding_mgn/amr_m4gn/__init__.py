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
from .amr_router import (
    aggregate_per_segment,
    sample_thresholds,
    build_l1_to_l0,
    route,
)
from .pe import rwse_segment, rwse_node
from .micro_gnn import MicroGNN
from .macro_transformer import (
    SegmentEncoder,
    MacroTransformer,
    dispatch,
    pack_segments,
    unpack_segments,
    run_macro_batched,
)
from .model import AMRM4GN
