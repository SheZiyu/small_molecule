import os
import zipfile
import numpy as np
import torch
from torch_scatter import scatter_add, scatter_mean


def construct_edges(A, n_node):
    # Flatten the adjacency matrix
    h_edge_fea = A.reshape(-1)  # [BPP]

    # Create indices for row and column
    h_row = (
        torch.arange(A.shape[1])
        .unsqueeze(-1)
        .expand([-1, A.shape[1]])
        .reshape(-1)
        .to(A.device)
    )
    h_col = (
        torch.arange(A.shape[1])
        .unsqueeze(0)
        .expand([A.shape[1], -1])
        .reshape(-1)
        .to(A.device)
    )

    # Expand row and column indices for batch dimension
    h_row = h_row.unsqueeze(0).expand([A.shape[0], -1])
    h_col = h_col.unsqueeze(0).expand([A.shape[0], -1])

    # Calculate offset for batch-wise indexing
    offset = (torch.arange(A.shape[0]) * n_node).unsqueeze(-1).to(A.device)

    # Apply offset to row and column indices
    h_row, h_col = (h_row + offset).reshape(-1), (h_col + offset).reshape(-1)

    # Create an edge mask where diagonal elements are set to 0
    h_edge_mask = torch.ones_like(h_row)
    base_diag_indices = (torch.arange(A.shape[1]) * (A.shape[1] + 1)).to(A.device)
    diag_indices_tensor = torch.tensor([]).to(A.device)
    for i in range(A.shape[0]):
        diag_indices = base_diag_indices + i * A.shape[1] * A.shape[-1]
        diag_indices_tensor = torch.cat(
            [diag_indices_tensor, diag_indices], dim=0
        ).long()
    h_edge_mask[diag_indices_tensor] = 0

    return h_row, h_col, h_edge_fea, h_edge_mask


def pairwise_distances(coords):
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
            distances[j, i] = distances[i, j]
    return distances


def calculate_displacement_xyz(frame1, frame2):
    dd = frame2 - frame1
    dx = dd[:, 0]
    dy = dd[:, 1]
    dz = dd[:, 2]
    return dd, dx, dy, dz


def calculate_rmsf(coordinates_array):
    # Calculate the average structure
    avg_structure = np.mean(coordinates_array, axis=0)

    # Initialize an array to store the squared fluctuations
    squared_fluctuations = np.zeros_like(avg_structure[:, 0])

    # Calculate displacement and squared fluctuations for each frame
    for coordinates in coordinates_array:
        displacement = coordinates - avg_structure
        squared_fluctuations += np.sum(displacement**2, axis=1)

    # Average the squared fluctuations over all frames
    avg_squared_fluctuations = squared_fluctuations / len(coordinates_array)

    # Take the square root to obtain RMSF
    rmsf = np.sqrt(avg_squared_fluctuations)

    return rmsf.reshape([-1, 1])


def radius_graph_custom(pos, r_min, r_max, batch) -> torch.Tensor:
    """Creates edges based on distances between points belonging to the same graph in the batch

    Args:
        pos: tensor of coordinates
        r_max: put no edge if distance is larger than r_max
        r_min: put no edge if distance is smaller than r_min
        batch : info to which graph a node belongs

    Returns:
        index: edges consisting of pairs of node indices
    """
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > r_min)).nonzero().T
    index_mask = index[0] != index[1]
    index = index[:, index_mask]
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index


def augment_edge(data):
    i, j = data.edge_index
    edge_vec = data.pos[j] - data.pos[i]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    data.edge_attr = edge_len
    return data


def augment_edge_extend(data, edge_index, pos):
    i, j = edge_index
    edge_vec = pos[j] - pos[i]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    data.edge_len = edge_len
    return data


def extract_pdb_from_zip(zip_folder, target_name, output_folder):
    """Extract PDB file from a specific ZIP file."""
    for zip_file_name in os.listdir(zip_folder):
        if zip_file_name.endswith(".zip"):
            if target_name in zip_file_name:
                zip_file_path = os.path.join(zip_folder, zip_file_name)
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    for file_name in zip_ref.namelist():
                        if file_name.endswith(".pdb"):
                            zip_ref.extract(file_name, output_folder)
                            return os.path.join(output_folder, file_name)
    return None


def write_combined_pdb(original_pdb, new_coordinates, output_file):
    """Write the combined PDB file with new coordinates."""
    print(f"Writing PDB file: {output_file}")
    print(f"Number of new coordinates: {len(new_coordinates)}")
    with open(original_pdb, "r") as original_file, open(
        output_file, "w"
    ) as combined_file:
        atom_idx = 0
        for line in original_file:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()

                # # Exclude hydrogen atoms
                # if atom_name.startswith('H'):
                #     combined_file.write(line)
                #     continue

                # Handle the new coordinates for non-hydrogen atoms
                if atom_idx < len(new_coordinates):
                    new_x, new_y, new_z = new_coordinates[atom_idx]
                    atom_idx += 1
                    new_line = (
                        f"{line[:30]}{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}{line[54:]}"
                    )
                    combined_file.write(new_line)
                else:
                    combined_file.write(line)

            elif line.startswith("ATOM") and resname == "UNL":
                combined_file.write(line)
            else:
                combined_file.write(line)

    print(f"Finished writing PDB file: {output_file}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA


def get_optimizer(model, name="adam", **kwargs):
    """Get optimizer by name and parameters from model"""
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(parameters, **kwargs)
    elif name == "adadelta":
        return torch.optim.Adadelta(parameters, **kwargs)
    elif name == "adagrad":
        return torch.optim.Adagrad(parameters, **kwargs)
    else:
        raise NotImplementedError("Optimizer not supported: %s" % name)


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
