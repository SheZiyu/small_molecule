'''
plot dx distribution to test the impact of scaling
'''
import pickle
import sys
import os
import glob
import zipfile
import shutil

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import networkx as nx
import pickle
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import radius_graph
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, to_networkx
from torch_geometric.transforms import gcn_norm, Compose

import MDAnalysis as mda

from prepocessing.preprocessing import parse_toml_file

# Set random seed for reproducibility
# torch.manual_seed(42)

# def radius_graph_custom(pos, r_max, r_min, batch) -> torch.Tensor:
#     """Creates edges based on distances between points belonging to the same graph in the batch
#
#     Args:
#         pos: tensor of coordinates
#         r_max: put no edge if distance is larger than r_max
#         r_min: put no edge if distance is smaller than r_min
#         batch : info to which graph a node belongs
#
#     Returns:
#         index: edges consisting of pairs of node indices
#     """
#     r = torch.cdist(pos, pos)
#     index = ((r < r_max) & (r > r_min)).nonzero().T
#     index_mask = index[0] != index[1]
#     index = index[:, index_mask]
#     index = index[:, batch[index[0]] == batch[index[1]]]
#     return index

def calculate_displacement_xyz(frame1, frame2):
    dd = frame2 - frame1
    dx = dd[:, 0]
    dy = dd[:, 1]
    dz = dd[:, 2]
    return dd, dx, dy, dz

class DataAnalysis:
    def __init__(self, directory_path, num_frames_to_process, selection='resname UNL', scaling=True, scale=2.0, align=True):
        self.directory_path = directory_path
        self.num_frames_to_process = num_frames_to_process
        self.selection = selection
        self.scaling = scaling
        self.scale = scale
        self.coordinates_arrays = []
        self.align = align
        # self.rmsf_values = []
        self.one_hot_trajs = []
        self.atoms_nums = []
        self.h5_filename = os.path.join(
            self.directory_path,
            '{}.h5'.format(self.selection.replace(' ', '_').lower())
        )

    def preprocess_coordinate_onehot(self):
        # Iterate over each system in the directory
        for zip_file_name in os.listdir(self.directory_path):
            if zip_file_name.endswith('.zip'):
                system_path = os.path.join(self.directory_path, zip_file_name)
                print(system_path)
                # Create a temporary directory with the same name as the zip file (without extension)
                temp_dir = os.path.splitext(zip_file_name)[0]
                temp_dir_path = os.path.join(self.directory_path, temp_dir)
                os.makedirs(temp_dir_path, exist_ok=True)

                # Extract contents of the zip file to the temporary directory
                with zipfile.ZipFile(system_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)

                # # Load true RMSF
                # rmsf_path = self._find_rmsf_tsv_files(temp_dir_path)[0]
                # rmsf_true = self._load_tsv_file(rmsf_path)
                #
                # # Append the true RMSF values to the list for the current replicate
                # self.rmsf_values.extend(np.split(rmsf_true, rmsf_true.shape[1], axis=1))

                # Check if the directory contains trajectory and topology files
                # print(glob.glob(temp_dir_path))
                xtc_files = [file for file in os.listdir(temp_dir_path) if file.endswith('.xtc')]
                if not xtc_files:
                    print('No XTC files found in {}'.format(temp_dir_path))
                pdb_file = [file for file in os.listdir(temp_dir_path) if file.endswith('.pdb')]
                if not pdb_file:
                    print('No PDB file found in {}'.format(temp_dir_path))

                # Load trajectory and topology using MDAnalysis
                for xtc_file in xtc_files:
                    trajectory_path = os.path.join(temp_dir_path, xtc_file)
                    topology_path = os.path.join(temp_dir_path, pdb_file[0])
                    try:
                        # ref = mda.Universe(topology_path)
                        u = mda.Universe(topology_path, trajectory_path)
                        # average = align.AverageStructure(u, u, select=self.selection, ref_frame=0).run()
                        # ref = average.results.universe
                        # aligner = align.AlignTraj(u, ref, select=self.selection, in_memory=True).run()
                        # rmsd_analysis = align.AlignTraj(u, ref, select=self.selection, match_atoms=True, in_memory=False)
                        # rmsd_analysis.run()
                        one_hot_encoded_atoms = self._one_hot_encode_atoms(u, selection=self.selection)
                        self.one_hot_trajs.append(one_hot_encoded_atoms)
                        self.atoms_nums.append(len(u.select_atoms(self.selection)))
                    except Exception as e:
                        print('Error loading trajectory and topology: {}'.format(e))

                    # if self.align:
                    #     # Initial reference frame is the first frame
                    #     ref_frame = u.select_atoms(self.selection)
                    #     # Iterate through each frame starting from the second frame
                    #     for ts in u.trajectory[1: self.num_frames_to_process]:
                    #         mobile = u.select_atoms(self.selection)
                    #         # Align the current frame to the previous frame (reference frame)
                    #         align.alignto(mobile, ref_frame, select=self.selection)
                    #         ref_frame = mobile  # Update the reference frame for the next iteration

                    # Initialize an empty array to store coordinates
                    coordinates_array = np.empty((len(u.trajectory[0: self.num_frames_to_process]),
                                                  u.select_atoms(self.selection).n_atoms, 3))
                    # Iterate through the trajectory frames and store coordinates
                    for i, ts in enumerate(u.trajectory[0: self.num_frames_to_process]):
                        coordinates_array[i] = u.select_atoms(self.selection).positions
                    if self.scaling:
                        coordinates_array *= self.scale
                    self.coordinates_arrays.append(coordinates_array)

                # Clean up: Remove the temporary extracted directory
                shutil.rmtree(temp_dir_path)

        # Save the NumPy array to a file
        with (h5py.File(self.h5_filename, 'w') as hf):
            # Create a group to store the datasets
            group = hf.create_group('h5')
            for i, (coordinates_array, one_hot_traj, atoms_num) in \
                    enumerate(zip(self.coordinates_arrays,
                                  self.one_hot_trajs,
                                  self.atoms_nums
                                  )):
                # Create dataset
                group.create_dataset('coordinates_array_{}'.format(i), data=coordinates_array)
                group.create_dataset('one_hot_traj_{}'.format(i), data=one_hot_traj)
                group.create_dataset('atoms_num_{}'.format(i), data=atoms_num)

        # return self.coordinates_arrays, self.rmsf_values, self.one_hot_trajs

    def _find_rmsf_tsv_files(self, directory):
        # Helper function to find RMSF TSV files in a directory
        search_pattern = '{}/*RMSF*.tsv'.format(directory)
        file_paths = glob.glob(search_pattern)
        return file_paths

    def _load_tsv_file(self, tsv_file_path):
        # Helper function to load data from a TSV file
        data_frame = pd.read_csv(tsv_file_path, sep='\t')
        data_array = data_frame.values
        return data_array[:, 1:]

    def _one_hot_encode_residues(self, u, selection):
        # Helper function to one-hot encode residues
        selected_residues = u.select_atoms(selection).residues  # Select residues based on the given selection
        #
        # # Get the unique residue names
        # unique_residue_names = np.unique(selected_residues.resnames)
        unique_residue_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        # Create a dictionary mapping each residue name to its one-hot encoded index
        residue_to_index = {resname: i for i, resname in enumerate(unique_residue_names)}

        # Initialize an array to store the one-hot encoded representations
        one_hot_encoded = np.zeros((len(selected_residues), len(unique_residue_names)))

        # Iterate through the selected residues and set the corresponding one-hot encoded values
        for i, residue in enumerate(selected_residues):
            residue_name = residue.resname
            one_hot_encoded[i, residue_to_index[residue_name]] = 1

        return one_hot_encoded

    def _one_hot_encode_atoms(self, u, selection):
        # Helper function to one-hot encode atom types
        selected_atoms = u.select_atoms(selection)  # Select atoms based on the given selection

        # Define unique atom types that are expected
        unique_atom_types = ['C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'F', 'Mg', 'Na', 'K', 'Ca', 'Fe', 'Zn',
                             'Cu']

        # Create a dictionary mapping each atom type to its one-hot encoded index
        atom_to_index = {atom_type: i for i, atom_type in enumerate(unique_atom_types)}

        # Initialize an array to store the one-hot encoded representations
        one_hot_encoded = np.zeros((len(selected_atoms), len(unique_atom_types)))

        # Iterate through the selected atoms and set the corresponding one-hot encoded values
        for i, atom in enumerate(selected_atoms):
            atom_type = atom.element
            if atom_type in atom_to_index:
                one_hot_encoded[i, atom_to_index[atom_type]] = 1
        print(one_hot_encoded)
        return one_hot_encoded

    def load_data_from_hdf5(self, file=None, group_name='h5'):
        '''
        Load datasets from an HDF5 file.

        Parameters:
        - file (string):
        - group_name: The name of the group containing the datasets.

        Returns:
        - coordinates_arrays: A list containing loaded coordinates arrays.
        - one_hot_trajs: A list containing loaded one-hot trajectory arrays.
        - atom_names: A list containing the number of atoms for each trajectory.
        '''
        # Initialize lists to store loaded datasets
        coordinates_arrays = []
        one_hot_trajs = []
        atoms_nums = []

        if file is None:
            file = self.h5_filename

        # Open the HDF5 file for reading
        with h5py.File(file, 'r') as hf:
            # Access the group containing the datasets
            group = hf[group_name]

            # Iterate over the datasets in the group
            for i in range(
                    len(group.keys()) // 3):  # Assuming pairs of coordinates and one_hot_trajs and atoms_nums datasets
                coordinates_array_key = 'coordinates_array_{}'.format(i)
                one_hot_traj_key = 'one_hot_traj_{}'.format(i)
                atoms_num_key = 'atoms_num_{}'.format(i)

                # Load coordinates array
                coordinates_array = np.array(group[coordinates_array_key])

                # Load one-hot trajectory array
                one_hot_traj = np.array(group[one_hot_traj_key])

                # Load numbers of atoms array
                atoms_num = np.array(group[atoms_num_key])

                # Append loaded arrays to the corresponding lists
                coordinates_arrays.append(coordinates_array)
                one_hot_trajs.append(one_hot_traj)
                atoms_nums.append(atoms_num)

        return coordinates_arrays, one_hot_trajs, atoms_nums

# class TrajectoriesDataset(Dataset):
#     def __init__(
#             self,
#             is_preprocess,
#             directory_path,
#             num_frames_to_process,
#             cutoff,
#             scale=1.0,
#             augment=False,
#             dataset=[],
#             file=None,
#             store_data=True,
#             store_location='/data2/ziyu_project/store_location/store.h5'
#     ):
#         super(TrajectoriesDataset, self).__init__()
#         self.augment = augment
#         self.dataset = dataset
#
#         if is_preprocess:
#             DataAnalysis(directory_path, num_frames_to_process).preprocess_coordinate_onehot()
#         coordinates_arrays, one_hot_trajs, atoms_nums = DataAnalysis(directory_path, num_frames_to_process).load_data_from_hdf5(file=file)
#         self.atoms_nums = np.unique(atoms_nums)
#         count = 0
#         with h5py.File(store_location, 'w') as h5f:
#             for traj_idx, (coordinates_array, one_hot, atoms_num) in enumerate(zip(coordinates_arrays, one_hot_trajs, atoms_nums)):
#                 one_hot = (torch.tensor(one_hot, dtype=torch.float)).unsqueeze(0).expand([coordinates_array.shape[0], -1, -1])
#                 # rmsf_value = (torch.tensor(rmsf_value.astype(np.float_), dtype=torch.float)).unsqueeze(0).expand([coordinates_array.shape[0], -1, -1])
#
#                 # Instantiate dataset as a list of PyG Data objects
#                 for frame_idx, (coordinates_i, one_hot_i) in tqdm(enumerate(zip(coordinates_array, one_hot)), disable=True,
#                                                                   desc='Sampling {}/{}'.format(traj_idx, atoms_num), total=len(coordinates_array)):
#                     coordinates_i = torch.tensor(coordinates_i, dtype=torch.float)
#                     if frame_idx < len(coordinates_array) - 1:
#                         dd, dx, dy, dz = calculate_displacement_xyz(coordinates_array[frame_idx], coordinates_array[frame_idx + 1])
#                     else:
#                         dd, dx, dy, dz = np.zeros_like(dd), np.zeros_like(dx), np.zeros_like(dy), np.zeros_like(dz)
#                     dd = torch.tensor(dd, dtype=torch.float)
#                     i, j = radius_graph(coordinates_i, r=cutoff*scale)
#                     frame_idx = torch.tensor(frame_idx).repeat([coordinates_i.shape[0], 1])
#                     data = Data(
#                         pos=coordinates_i,
#                         x=one_hot_i,
#                         edge_index=torch.stack([i, j]),
#                         traj_idx=traj_idx,
#                         frame_idx=frame_idx,
#                         atoms_num=atoms_num,
#                         dd=dd
#                     )
#                     data = self.augment_edge(data)
#                     obj = pickle.dumps(data)
#                     h5f.create_dataset(f'object_{count}', data=np.void(obj))
#                     count = count + 1
#                     # self.dataset.append(data)
#         print('stop')
#
#     def len(self):
#         return len(self.dataset)
#
#     def get(self, idx):
#         if self.augment:
#             return self.random_rotate(self.dataset[idx])
#         else:
#             return self.dataset[idx]
#
#     def random_rotate(self, data):
#         R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device)
#         data.pos @= R
#         return data
#
#     def augment_edge(self, data):
#         # Extract edge indices i, j from the data
#         i, j = data.edge_index
#
#         # Compute edge vectors (edge_vec) and edge lengths (edge_len)
#         edge_vec = data.pos[j] - data.pos[i]
#         edge_len = edge_vec.norm(dim=-1, keepdim=True)
#
#         # Concatenate edge vectors and edge lengths into edge_encoding
#         # data.edge_encoding = torch.hstack([edge_vec, edge_len])
#         data.edge_encoding = edge_len
#         return data

def construct_edges(A, n_node):
    # Flatten the adjacency matrix
    h_edge_fea = A.reshape(-1) # [BPP]

    # Create indices for row and column
    h_row = torch.arange(A.shape[1]).unsqueeze(-1).expand([-1, A.shape[1]]).reshape(-1).to(A.device)
    h_col = torch.arange(A.shape[1]).unsqueeze(0).expand([A.shape[1], -1]).reshape(-1).to(A.device)

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
        diag_indices_tensor = torch.cat([diag_indices_tensor, diag_indices], dim=0).long()
    h_edge_mask[diag_indices_tensor] = 0

    return h_row, h_col, h_edge_fea, h_edge_mask

# Plot dx distribution
def plt_dx_distribution(num_frames_to_process, traj1, traj2):
    # Generate coordinates over time
    time = np.arange(0, num_frames_to_process - 1)

    # dx trajectory
    dx_trajectory1 = np.zeros([num_frames_to_process - 1, traj1[0].dx.shape[0], traj1[0].dx.shape[1]])
    for i in range(num_frames_to_process - 1):
        dx_trajectory1[i] = traj1[i].dx

    dx_trajectory2 = np.zeros([num_frames_to_process - 1, traj2[0].dx.shape[0], traj2[0].dx.shape[1]])
    for i in range(num_frames_to_process - 1):
        dx_trajectory2[i] = traj2[i].dx

    # Create a colormap based on time index
    colors = cm.viridis(np.linspace(0, 1, len(time)))

    # Create 3D scatter plots for the dx trajectory with a colormap
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(traj1[0].dx.shape[0]):
        x1 = dx_trajectory1[:, i, 0]
        y1 = dx_trajectory1[:, i, 1]
        z1 = dx_trajectory1[:, i, 2]
        ax1.scatter(x1, y1, z1, c=colors, cmap='viridis', s=10, marker='o', label='Atom {}'.format(i + 1))
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('3D dx Trajectory with Time Index - Scale=1.0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(traj2[0].dx.shape[0]):
        x2 = dx_trajectory2[:, i, 0]
        y2 = dx_trajectory2[:, i, 1]
        z2 = dx_trajectory2[:, i, 2]
        ax2.scatter(x2, y2, z2, c=colors, cmap='viridis', s=10, marker='^', label='Atom {}'.format(i + 1))
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title('3D dx Trajectory with Time Index - Scale=2.0')
    ax2.legend()

    # Add colorbars to show the time index
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax1)
    cbar.set_label('Time Index')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
    cbar.set_label('Time Index')

    # Show the plot
    plt.show()

# Plot point distribution
def plt_point_distribution(num_frames_to_process, traj1, traj2, idx=0):
    # Generate coordinates over time
    time = np.arange(0, num_frames_to_process - 1)

    # point trajectory
    x1 = np.zeros(num_frames_to_process - 1)
    y1 = np.zeros(num_frames_to_process - 1)
    z1 = np.zeros(num_frames_to_process - 1)

    x2 = np.zeros(num_frames_to_process - 1)
    y2 = np.zeros(num_frames_to_process - 1)
    z2 = np.zeros(num_frames_to_process - 1)

    for i in range(num_frames_to_process - 1):
        x1[i] = traj1[i].dx[idx][0]
        y1[i] = traj1[i].dx[idx][1]
        z1[i] = traj1[i].dx[idx][2]
        x2[i] = traj2[i].dx[idx][0]
        y2[i] = traj2[i].dx[idx][1]
        z2[i] = traj2[i].dx[idx][2]

    # Create a colormap based on time index
    colors = cm.viridis(np.linspace(0, 1, len(time)))

    # Create a 3D scatter plot for the point trajectory with a colormap
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x1, y1, z1, c=colors, cmap='viridis', label='Trajectory')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('3D Point Trajectory with Time Index - Scale=1.0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x2, y2, z2, c=colors, cmap='viridis', label='Trajectory')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title('3D Point Trajectory with Time Index - Scale=2.0')
    ax2.legend()

    # Add a colorbar to show the time index
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax1)
    cbar.set_label('Time Index')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
    cbar.set_label('Time Index')

    # Show the plot
    plt.show()

def generate_loaders(dataset, parameters):
    pin_memory = parameters['pin_memory']
    num_workers = parameters['num_workers']
    batch_size = parameters['batch_size']
    train_size = parameters['train_size']

    # Train-validation split
    train_set, valid_set = train_test_split(dataset, train_size=train_size, random_state=42)
    print('Number of training graphs: {}'.format(len(train_set)))
    print('Number of validation graphs: {}'.format(len(valid_set)))

    # Move to data loaders
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory, 'follow_batch': ['pos']}
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **kwargs)
    return batch_size, train_loader, valid_loader

def data_augmentation(data):
    data.x += torch.randn_like(data.x) * 0.1  # Add random noise to node features
    return data

def generate_dataset(dataset, batch_size, num_workers, test_size=0.2):
    # Calculate sizes for train, validation, and test sets
    # total_size = len(dataset)
    # test_size = int(test_split * total_size)
    # val_size = int(validation_split * total_size)
    # train_size = total_size - test_size - val_size
    #
    # # Define indices for slicing
    # # train_indices = range(train_size)
    # # val_indices = range(train_size, train_size + val_size)
    # # test_indices = range(train_size + val_size, total_size)
    #
    # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],  generator=torch.Generator().manual_seed(42))
    train_set, val_set = train_test_split(dataset, test_size=test_size, random_state=42)

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
    return train_loader, val_loader

def generate_train_dataset(dataset, batch_size, num_workers):
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
    return train_loader

def generate_val_dataset(dataset, batch_size, num_workers):
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
    return val_loader

def generate_test_dataset(dataset, batch_size, num_workers):
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
    return test_loader

# directory_path = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/task2'
# cutoff = 1  # Angstrom
# scale = 1e-22
#
# TrajsDataset = TrajectoriesDataset(
#     directory_path,
#     cutoff=cutoff,
#     scale=scale
# )
# print('TrajsDataset[0].size: {}'.format(TrajsDataset[-1].size))
# node_dim = TrajsDataset[-1].x.shape[-1]
# print('init_node_dim: {}'.format(node_dim))
# config = parse_toml_file('../config.toml')
# data_dir = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/'
# dataset_location = os.path.join(data_dir, 'dataset.pickle')
# pickle_object(TrajsDataset, dataset_location)
# dataset = unpickle_object(dataset_location)
# batch_size, train_loader, val_loader = generate_loaders(dataset, config)
# device0 = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# for batch in train_loader:
#     batch = augment_batch(batch)
#     batch_idx = batch.batch
#     edge_dim = batch.edge_encoding.shape[-1]
#     vector_dim = batch.pos.shape[-1]
#     break

class TrajectoriesDataset_Efficient(Dataset):
    def __init__(
            self,
            cutoff,
            scale=1.0,
            augment=False,
            dataset=[],
            original_h5_file=None,

    ):
        super(TrajectoriesDataset_Efficient, self).__init__()
        self.cutoff = cutoff
        self.scale = scale
        self.original_h5_file = original_h5_file
        self.augment = augment
        self.dataset = dataset
        self.indices_traj_frames = []
        self.h5_file = h5py.File(self.original_h5_file, 'r')

        with h5py.File(self.original_h5_file, 'r') as hf:
            # Access the group containing the datasets
            group = hf['h5']
            self.number_trajs = int(len(group.keys()) // 3)
            self.number_frames = group['coordinates_array_0'].shape[0]

        for i in range(self.number_trajs):
            for j in range(self.number_frames - 1):
                self.indices_traj_frames.append([i, j])

    def len(self):
        return len(self.indices_traj_frames)

    def get(self, idx):
        # system_idx = 0
        # frame_idx = 999
        system_idx, frame_idx = self.indices_traj_frames[idx]

        # with h5py.File(self.original_h5_file, 'r') as hf:
        # Access the group containing the datasets
        group = self.h5_file['h5']

        # Iterate over the datasets in the group
        coordinates_array_key = 'coordinates_array_{}'.format(system_idx)
        one_hot_traj_key = 'one_hot_traj_{}'.format(system_idx)
        atoms_num_key = 'atoms_num_{}'.format(system_idx)

        # Load coordinates array
        coordinates_array = np.array(group[coordinates_array_key][frame_idx])
        coordinates_array_plus1 = np.array(group[coordinates_array_key][frame_idx + 1])
        coordinates_array = torch.tensor(coordinates_array, dtype=torch.float)
        coordinates_array_plus1 = torch.tensor(coordinates_array_plus1, dtype=torch.float)

        # Load one-hot trajectory array
        one_hot_traj = np.array(group[one_hot_traj_key])
        one_hot_traj = torch.tensor(one_hot_traj, dtype=torch.float)  # .unsqueeze(0).expand([coordinates_array.shape[0], -1, -1])

        # Load numbers of atoms array
        atoms_num = np.array(group[atoms_num_key])

        i, j = radius_graph(coordinates_array,
                            self.cutoff*self.scale,
                            batch=torch.zeros(coordinates_array.shape[0]))
        edge_index = torch.stack([i, j])

        i, j = radius_graph(coordinates_array_plus1,
                            self.cutoff * self.scale,
                            batch=torch.ones(coordinates_array_plus1.shape[0]))
        edge_index_plus1 = torch.stack([i, j])

        frame_idx = torch.tensor(frame_idx).repeat([coordinates_array.shape[0], 1])

        frame_idx_plus1 = frame_idx + 1

        data = Data(
            pos=coordinates_array,
            x=one_hot_traj,
            edge_index=edge_index,
            traj_idx=system_idx,
            frame_idx=frame_idx,
            atoms_num=atoms_num)
        data = self.augment_edge(data)

        data_plus1 = Data(
            pos=coordinates_array_plus1,
            x=one_hot_traj,
            edge_index=edge_index_plus1,
            traj_idx=system_idx,
            frame_idx=frame_idx_plus1,
            atoms_num=atoms_num)
        data_plus1 = self.augment_edge(data_plus1)

        if self.augment:
            return self.random_rotate(data), self.random_rotate(data_plus1)
        else:
            return data, data_plus1

    def random_rotate(self, data):
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device)
        data.pos @= R
        return data

    def augment_edge(self, data):
        # Extract edge indices i, j from the data
        i, j = data.edge_index

        # Compute edge vectors (edge_vec) and edge lengths (edge_len)
        edge_vec = data.pos[j] - data.pos[i]
        edge_len = edge_vec.norm(dim=-1, keepdim=True)

        # Concatenate edge vectors and edge lengths into edge_encoding
        # data.edge_encoding = torch.hstack([edge_vec, edge_len])
        data.edge_attr = edge_len
        return data

if __name__ == '__main__':
    config = parse_toml_file('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/config.toml')
    # directory_path = config['directory_path']
    directory_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys'
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    device = config['device']
    num_frames_to_process = config['num_frames_to_process']
    DataAnalysis(directory_path, num_frames_to_process, align=True).preprocess_coordinate_onehot()
    TrajsDataset = TrajectoriesDataset_Efficient(cutoff,
                                                 scale,
                                                 original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys/resname_unl.h5')
    A, B = TrajsDataset.get(0)
    AA, BB = TrajsDataset[0]
    loader = DataLoader(TrajsDataset, batch_size=2, num_workers=8, shuffle=True, pin_memory=False)
    for (a, b) in loader:
    #     print('a')
    #     # del(a)
        print(a)
        print(b)
        # print(aa)
        print(a.atoms_num)
        print(type(a.atoms_num))
        print(a.frame_idx)
        print(b.frame_idx)
        break



