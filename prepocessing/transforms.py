"""Some utility functions """

from collections.abc import Iterable
import torch
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_scatter import scatter_mean
from tqdm import tqdm

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}


def pack_tensor_in_tuple(seq):
    """pack a tensor into a tuple of Tensor of length 1"""
    if isinstance(seq, torch.Tensor):
        return (seq,)
    elif isinstance(seq, Iterable):
        return (*seq,)
    else:
        return seq


def subtract_means(stacked_points, node2graph):
    """Center the stacked_points by subtracting the corresponding means. node2graph specifies wich points belong to the same graph.

    Args:
        stacked_points: tensor of atom points
        node2graph: tensor of indices specifying to which graph a node belongs

    Returns:
        centered_points: centered points
    """
    means = scatter_mean(
        stacked_points, node2graph, dim=0, dim_size=node2graph.max() + 1
    )
    centered_points = stacked_points - means.index_select(0, node2graph)
    return centered_points


def count_parameters(model):
    """Count the parameters of a model.

    Args:
        model: the torch.nn model

    Returns:
        num_parameters: the number of parameters
    """
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save a model checkpoint

    Args:
        state: the state of the model
        filename: Name of the file. Defaults to "checkpoint.pth.tar".
    """
    torch.save(state, filename)


def load_checkpoint(model, optimizer=None, filename="checkpoint.pth.tar"):
    """Loading the checkpoint

    Args:
        model: torch.nn model to load the parameters into
        optimizer: optimizer to load the parameters into
        filename: File name of the model and optimizer data. Defaults to "checkpoint.pth.tar".

    Returns:
        _type_: _description_
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def check_all_edgetypes(data_set):
    """Check all edge types in the dataset"""
    batch_size = 10000
    train_loader = DataLoader(data_set, batch_size, shuffle=False)
    edge_types = set()
    for ind, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        for edge_type in batch.edge_type.unique():
            edge_types.add(edge_type.item())
    return edge_types


def binarize(x):
    """If the element is greater than 0, it is replaced with 1.
    If the element is not greater than 0 (i.e., it is 0 or negative), it is replaced with 0.

    Args:
        x : matrix with values >=0

    Returns:
        binarized_matrix: a matrix that contains only 0 and 1. 1 where x had values >0, 0 where x had values =0
    """
    binarized_matrix = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    return binarized_matrix


def get_higher_order_adj_matrix(adj, order):
    """
    Find also higher order proximity: 1 means direct neighbor, 2 means neighbor of neighbor, etc.
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    Returns:
        Following attributes will be updated:
            - edge_index
            - edge_type
        Following attributes will be added to the data object:
            - bond_edge_index:  Original edge_index.
    """
    adj_mats = [
        torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
    ]

    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj)

    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

    return order_mat


def unique_edges(edge_index, edge_type):
    """Sorts the edge_index tensor and removes potential duplicates

    Args:
        edge_index: tensor of edge indices
        edge_type: tensor of edge types

    Returns:
        unique_edge_indices: unqiue edge indices
        unique_edge_types: unique edge types
    """
    # Combine edge_list and edge_type for unique operation
    edges_with_types = torch.cat([edge_index.t(), edge_type.reshape((-1, 1))], dim=1)

    # Use unique to remove duplicates. This also sorts the edges.
    unique_edges_with_types = torch.unique(edges_with_types, dim=0)

    # Split the unique edges and their types
    unique_edge_indices = unique_edges_with_types[:, :2].t()
    unique_edge_types = unique_edges_with_types[:, 2]

    return unique_edge_indices, unique_edge_types


def extend_edges_by_higher_order_neighborhood(edge_indices, edge_types):
    """Add edges based on neighborhood of higher order and also change the edge types to include higher order neighborhood information

    Args:
        edge_indices: the edge indices as specified by covalent bonds
        edge_types: edge types as were specified by rdkit when creating the data
    Returns:
        unique_edge_indices: unqiue edge indices
        unique_edge_types: unique edge types
    """
    num_types = len(BOND_TYPES)

    adj = to_dense_adj(edge_indices).squeeze(0)
    order = 3  # what is that?
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_indices, edge_attr=edge_types).squeeze(
        0
    )  # (N, N), matrix of bond types
    # we need to add types to ensure unique identifiability
    type_highorder = torch.where(
        adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order)
    )
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    new_edge_type = new_edge_type.int()
    unique_edge_indices, unique_edge_types = unique_edges(new_edge_index, new_edge_type)
    return unique_edge_indices, unique_edge_types


def extend_to_radius_graph(
    coords_1,
    coords_2,
    edge_index,
    edge_type,
    cutoff,
    node2graph,
    specified_type_number=3,
    unspecified_type_number=0,
):
    """Add further edges based on distance. Also include edge information

    Args:
        coords_1: coordinates of the first set of nodes
        coords_2: coordinates of the second set of nodes
        edge_index: source and destination of the edges
        edge_type: type of the edges
        cutoff: an edge will be set if the distance between the atoms is smaller than the cutoff
        node2graph: specifies which of the batched nodes belong to which graph
        unspecified_type_number: This is the edge type that is assigned to the edges constructed from radius. Defaults to 0.
        specified_type_number: The edge type that is assigned to the edges connecting coords_1 and coords_2. Defaults to 3.

    Returns:
        new_edge_index: the updated edge source/destination tensor
        new_edge_type: the type of the edges
    """
    assert edge_type.dim() == 1
    device = coords_1.device
    coords_2 = coords_2.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    node2graph = node2graph.to(device)

    number_nodes_1 = coords_1.size(0)
    number_nodes_2 = coords_2.size(0)
    total_nodes = number_nodes_1 + number_nodes_2

    # Combine the coordinates
    combined_coords = torch.cat([coords_1, coords_2], dim=0)

    # Extend the node2graph tensor
    extended_node2graph = torch.cat([node2graph, node2graph], dim=0).to(device)

    # Create new indices for connecting coords_1 with coords_2
    connect_indices_1 = torch.arange(number_nodes_1, device=device)
    connect_indices_2 = torch.arange(number_nodes_2, device=device) + number_nodes_1
    extended_edge_index = torch.stack([connect_indices_1, connect_indices_2], dim=0)

    # Edge types for the new connections
    extended_edge_type = torch.full(
        (number_nodes_1,), specified_type_number, dtype=torch.long, device=device
    )

    # Combine with existing edges
    combined_edge_index = torch.cat([edge_index, extended_edge_index], dim=1)
    combined_edge_type = torch.cat([edge_type, extended_edge_type], dim=0)

    # Create sparse adjacency matrix for existing and extended edges
    bgraph_adj = torch.sparse_coo_tensor(
        combined_edge_index,
        combined_edge_type,
        torch.Size([total_nodes, total_nodes]),
        device=device,
    )

    # Create radius graph for the combined coordinates
    rgraph_edge_index = radius_graph(
        combined_coords, r=cutoff, batch=extended_node2graph
    )
    rgraph_adj = torch.sparse_coo_tensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1), device=device).long()
        * unspecified_type_number,
        torch.Size([total_nodes, total_nodes]),
        device=device,
    )

    # Combine adjacency matrices
    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


def extend_to_fully_connected_graph(
    coords,
    edge_index,
    edge_type,
    node2graph,
    unspecified_type_number=0,
):
    """Add further edges to make the graph fully connected. Also include edge information.

    Args:
        coords: coordinates of the atoms
        edge_index: source and destination of the edges
        edge_type: type of the edges
        node2graph: specifies which of the batched nodes belong to which graph
        unspecified_type_number: This is the edge type that is assigned to the new edges. Defaults to 0.

    Returns:
        new_edge_index: the updated edge source/destination tensor
        new_edge_type: the type of the edges
    """
    assert edge_type.dim() == 1
    number_nodes = coords.size(0)

    # Create adjacency matrix for existing edges
    bgraph_adj = torch.sparse_coo_tensor(
        edge_index, edge_type, torch.Size([number_nodes, number_nodes])
    )

    # Create fully connected graph edge indices within each individual graph
    full_edge_index_list = []
    unique_graphs = node2graph.unique()
    for graph in unique_graphs:
        nodes_in_graph = (node2graph == graph).nonzero(as_tuple=True)[0]
        full_edge_index = torch.combinations(nodes_in_graph, r=2).T
        # Duplicate edges to include both directions
        full_edge_index = torch.cat([full_edge_index, full_edge_index.flip(0)], dim=1)
        full_edge_index_list.append(full_edge_index)

    full_edge_index = torch.cat(full_edge_index_list, dim=1)

    # Create adjacency matrix for fully connected graph with unspecified edge type
    full_adj = torch.sparse_coo_tensor(
        full_edge_index,
        torch.ones(full_edge_index.size(1)).long().to(coords.device)
        * unspecified_type_number,
        torch.Size([number_nodes, number_nodes]),
    )

    # Combine existing edges with fully connected graph edges
    composed_adj = (bgraph_adj + full_adj).coalesce()  # Sparse (N, N, T)

    # Extract new edge indices and types
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    return new_edge_index, new_edge_type
