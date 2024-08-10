"""We have 10 times 10 k frames. We want to calculate dx and merge the data into a single file
"""

import os
import torch
import numpy as np
from prepocessing.from_noe import rdmol_to_edge
import mdtraj as md


def calculate_increments(traj, stride):
    increments = traj[stride:] - traj[:-stride]
    return increments


def calculate_increments_with_alignment(traj, stride):
    """Calculate the displacement

    Args:
        traj: the molecule trajectory
        stride: the increments between x_{i} and x_{i+stride} will be calculated
    Returns:
        increments the increments between x_{i+stride} and x_{i}
    """
    # alignmen
    increments = []
    for ind in range(len(traj) - stride):
        print(ind)
        two_element_traj = torch.cat(
            [traj[ind, None, :], traj[ind + stride, None, :]], dim=0
        )
        aligned_elements = align_traj(two_element_traj)
        increment = aligned_elements[1] - aligned_elements[0]
        increments.append(increment)
    increments_tensor = torch.tensor(increments)
    return increments_tensor


def align_traj(traj):
    md_traj = md.Trajectory(np.array(traj), topology=None)
    # Use the first frame as the reference for alignment
    reference_frame = md_traj[0]
    # Superpose all frames to the reference frame
    aligned_traj = md_traj.superpose(reference_frame).xyz
    return aligned_traj



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
        traj = align_traj(traj)
        traj = torch.tensor(traj)
        increments = calculate_increments_with_alignment(traj, stride)

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
