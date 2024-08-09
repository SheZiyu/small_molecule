"""We have 10 times 10 k frames. We want to calculate dx and merge the data into a single file
"""

import os
import torch
import numpy as np
from prepocessing.from_noe import rdmol_to_edge


def calculate_increments(traj, stride):
    """Calculate the displacement

    Args:
        traj: the molecule trajectory
        stride: the increments between x_{i} and x_{i+stride} will be calculated
    Returns:
        increments the increments between x_{i+stride} and x_{i}
    """
    increments = traj[stride:] - traj[:-stride]
    return increments


def create_increments_and_merge(traj_dir, stride, output_dir):
    """Calculate the increments and merge the data into a single file

    Args:
        traj_dir: the directory containing the trajectory files
        stride: the increments between x_{i} and x_{i+stride} will be calculated
    Returns:
        trajectory_merged_tensor: the merged trajectory tensor
        increments_merged_tensor: the merged increments tensor
    """
    trajectory_merged = []
    increments_merged = []
    for ind in range(10):
        traj_path = os.path.join(traj_dir, f"trajectory_coords-{ind}.npy")
        traj = torch.tensor(np.load(traj_path))
        increments = calculate_increments(traj, stride)
        trajectory_merged.append(traj[:-stride])
        increments_merged.append(increments)
    trajectory_merged_tensor = torch.cat(trajectory_merged)
    increments_merged_tensor = torch.cat(increments_merged)
    # save the data
    torch.save(
        trajectory_merged_tensor,
        os.path.join(output_dir, "trajectory_merged_tensor.pt"),
    )
    # save increments:
    torch.save(
        increments_merged_tensor,
        os.path.join(output_dir, "increments_merged_tensor.pt"),
    )


if __name__ == "__main__":
    stride = 5
    traj_dir = "/storage/florian/ziyu_project/ala2/numpy_dir/"
    output_dir = "/storage/florian/ziyu_project/ala2/merged_data/"
    create_increments_and_merge(traj_dir, stride, output_dir)
