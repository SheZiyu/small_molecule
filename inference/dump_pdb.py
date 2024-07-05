import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings('ignore')

# Specify the file path to the .npy file
file_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_solver1/1a10A/all_positions.npy'
# Load the numpy array from the .npy file
xyz = np.load(file_path)
print(f"Loaded positions array shape: {xyz.shape}")

# Load the PDB file to get the topology
pdb_file = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_solver1/1a10A/gt_1a10A.pdb'
u = mda.Universe(pdb_file)
print(f"Loaded PDB file: {pdb_file}")

# Extract unit cell dimensions from the PDB file if available
unitcell = u.trajectory.ts.dimensions if u.trajectory.ts.dimensions is not None else np.array([1, 1, 1, 90, 90, 90], dtype=np.float32)
print(f"Using unit cell dimensions: {unitcell}")

# Create a new Universe with the topology
new_universe = mda.Merge(u.atoms)
new_universe.add_TopologyAttr('tempfactors')
print("Created new universe with the loaded topology.")

# Create a trajectory with MemoryReader
memory_reader = MemoryReader(xyz)
print("Created MemoryReader with the loaded positions.")

# Assign the MemoryReader to the new_universe
new_universe.trajectory = memory_reader
print("Assigned MemoryReader to the new universe.")

# Explicitly set the unit cell dimensions for each frame
for ts in new_universe.trajectory:
    ts.dimensions = unitcell

# Save the new trajectory in different formats
output_dir = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_solver1/1a10A'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

with mda.Writer(os.path.join(output_dir, 'output_trajectory.pdb'), new_universe.atoms.n_atoms) as PDB_writer:
    PDB_writer.write(new_universe.atoms)

with mda.Writer(os.path.join(output_dir, 'output_trajectory.dcd'), new_universe.atoms.n_atoms, unitcell=unitcell) as DCD_writer:
    for ts in new_universe.trajectory:
        DCD_writer.write(new_universe.atoms)

with mda.Writer(os.path.join(output_dir, 'output_trajectory.xtc'), new_universe.atoms.n_atoms, unitcell=unitcell) as XTC_writer:
    for ts in new_universe.trajectory:
        XTC_writer.write(new_universe.atoms)
