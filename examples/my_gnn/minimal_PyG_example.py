import torch
from torch_geometric.data import Data
from physicsnemo.models.meshgraphnet import MeshGraphNet, HybridMeshGraphNet, MeshGraphKAN, BiStrideMeshGraphNet

# --- Minimal PyG Example (Base MeshGraphNet) ---

# Create a toy graph and random features
# In a real application, these would come from your mesh data
num_nodes = 100
num_edges = 300
edge_index = torch.randint(0, num_nodes, (2, num_edges))
graph = Data(edge_index=edge_index, num_nodes=num_nodes)
node_features = torch.randn(num_nodes, 4)  # [N, input_dim_nodes]
edge_features = torch.randn(num_edges, 3)  # [E, input_dim_edges]

# Instantiate the base model
model = MeshGraphNet(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    mlp_activation_fn="relu",
    aggregation="sum",
)

# Run a forward pass
node_outputs = model(node_features, edge_features, graph)  # [N, 2]
print("Base MeshGraphNet Output Shape:", node_outputs.shape)
# Output: Base MeshGraphNet Output Shape: torch.Size([100, 2])

# --- HybridMeshGraphNet Example ---
# The Hybrid model requires separate features for mesh and world edges
mesh_edge_features = torch.randn(edge_index.size(1), 3)
world_edge_features = torch.randn(edge_index.size(1), 3)

# Hybrid expects graph.num_edges == num_mesh_edges + num_world_edges
hybrid_edge_index = torch.cat([edge_index, edge_index], dim=1)
hybrid_graph = Data(edge_index=hybrid_edge_index, num_nodes=num_nodes)

model_hybrid = HybridMeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=2)
node_outputs_hybrid = model_hybrid(
    node_features, mesh_edge_features, world_edge_features, hybrid_graph
)
print("HybridMeshGraphNet Output Shape:", node_outputs_hybrid.shape)
# Output: HybridMeshGraphNet Output Shape: torch.Size([100, 2])

# --- MeshGraphKAN Example ---
model_kan = MeshGraphKAN(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    num_harmonics=5,  # KAN-specific parameter
)
node_outputs_kan = model_kan(node_features, edge_features, graph)
print("MeshGraphKAN Output Shape:", node_outputs_kan.shape)
# Output: MeshGraphKAN Output Shape: torch.Size([100, 2])

# --- BiStrideMeshGraphNet Example ---
# This model requires a pre-computed graph pyramid
# In a real-world scenario, you would create these based on your mesh hierarchy
num_nodes_lvl0 = num_nodes
num_nodes_lvl1 = num_nodes // 2

# Build ring edges so source indices cover all nodes at each level.
src0 = torch.arange(num_nodes_lvl0)
dst0 = (src0 + 1) % num_nodes_lvl0
lvl0_edge_index = torch.stack([src0, dst0], dim=0)

src1 = torch.arange(num_nodes_lvl1)
dst1 = (src1 + 1) % num_nodes_lvl1
lvl1_edge_index = torch.stack([src1, dst1], dim=0)

ms_edges = [lvl0_edge_index, lvl1_edge_index]
ms_ids = [torch.arange(num_nodes_lvl1)]
model_bistride = BiStrideMeshGraphNet(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    num_mesh_levels=1,
)
bistride_graph = Data(edge_index=lvl0_edge_index, num_nodes=num_nodes)
bistride_graph.pos = torch.randn(num_nodes, 3)
edge_features_bistride = torch.randn(lvl0_edge_index.size(1), 3)
node_outputs_bistride = model_bistride(
    node_features,
    edge_features_bistride,
    bistride_graph,
    ms_edges=ms_edges,
    ms_ids=ms_ids,
)
print("BiStrideMeshGraphNet Output Shape:", node_outputs_bistride.shape)
# Output: BiStrideMeshGraphNet Output Shape: torch.Size([100, 2])

