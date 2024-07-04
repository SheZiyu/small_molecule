import numpy as np

import MDAnalysis as mda

# Function to load PDB and XTC files
def load_data(pdb_file, xtc_file):
    traj = mda.Universe(pdb_file, xtc_file)
    return traj

# Function A
def print_residue_names(traj):
    for residue in traj.residues:
        print(residue.resname)

## Function B
def extract_all_coordinates(traj):
    coordinates = np.empty((len(traj.trajectory), len(traj.atoms), 3))
    for i, ts in enumerate(traj.trajectory):
        coordinates[i] = ts.positions
    return coordinates

# Function C
def extract_ca_coordinates(traj):
    ca_indices = traj.select_atoms('name CA and not (name H*)').indices
    ca_coordinates = np.empty((len(traj.trajectory), len(ca_indices), 3))
    for i, ts in enumerate(traj.trajectory):
        ca_coordinates[i] = ts.positions[ca_indices]
    return ca_coordinates

# Function D
def print_atoms_after_hydrogen_removal(traj):
    for residue in traj.residues:
        atoms = [atom.name for atom in residue.atoms if atom.name != 'H']
        print('Residue {}: {}'.format(residue.resname, atoms))

if __name__ == '__main__':
    # Example usage:
    # Load data
    pdb_file = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/task2/1ah7_A_analysis/1ah7_A.pdb'
    xtc_file = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/task2/1ah7_A_analysis/1ah7_A_R3.xtc'
    traj = load_data(pdb_file, xtc_file)

    # Call the functions as needed
    print_residue_names(traj)
    all_coordinates = extract_all_coordinates(traj)
    ca_coordinates = extract_ca_coordinates(traj)
    print_atoms_after_hydrogen_removal(traj)
