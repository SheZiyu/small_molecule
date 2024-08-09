"""Turn trajectory from dcd to numpy

Returns:
    _type_: _description_
"""

import os
import numpy as np
import MDAnalysis as mda


def dcd_to_numpy(pdb_path, dcd_path, output_path=None):
    """
    Convert a DCD trajectory to a NumPy array.

    Parameters:
    - pdb_path (str): Path to the PDB file (topology).
    - dcd_path (str): Path to the DCD file (trajectory).
    - output_path (str, optional): Path to save the NumPy array. If None, the array is not saved.

    Returns:
    - np.ndarray: A NumPy array containing the trajectory coordinates with shape (n_frames, n_atoms, 3).
    """
    # Load the PDB and DCD files into an MDAnalysis Universe
    u = mda.Universe(pdb_path, dcd_path)
    # Extract the trajectory coordinates into a NumPy array
    trajectory_coords = np.array([ts.positions for ts in u.trajectory])
    # Save the NumPy array to a file if an output path is provided
    if output_path:
        np.save(output_path, trajectory_coords)
        print(f"Trajectory coordinates saved to {output_path}")
    return trajectory_coords


# Example usage
if __name__ == "__main__":
    # Define the base path
    base_path = "/storage/florian/ziyu_project/ala2"
    pdb_file = os.path.join(base_path, "ala2.pdb")
    for i in range(0, 10):
        dcd_file = os.path.join(base_path, f"trajectory-{i}.dcd")
        output_file = os.path.join(base_path, "numpy_dir", f"trajectory_coords-{i}.npy")
        # Convert DCD to NumPy and save the result
        trajectory_array = dcd_to_numpy(pdb_file, dcd_file, output_file)
        # Print the shape of the resulting array
        print("Trajectory array shape:", trajectory_array.shape)
