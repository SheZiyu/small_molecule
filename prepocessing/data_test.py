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

from rdkit import Chem
import MDAnalysis as mda

from prepocessing.from_noe import rdmol_to_edge
from prepocessing.transforms import extend_to_radius_graph
from utils.auxiliary import pairwise_distances, calculate_displacement_xyz, calculate_rmsf
from prepocessing.preprocessing import parse_toml_file


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
        self.names = []
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
                        self.names.append(temp_dir)
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
            for i, (coordinates_array, one_hot_traj, atoms_num, name) in \
                    enumerate(zip(self.coordinates_arrays,
                                  self.one_hot_trajs,
                                  self.atoms_nums,
                                  self.names
                                  )):
                # Create dataset
                group.create_dataset('coordinates_array_{}'.format(i), data=coordinates_array)
                group.create_dataset('one_hot_traj_{}'.format(i), data=one_hot_traj)
                group.create_dataset('atoms_num_{}'.format(i), data=atoms_num)
                group.create_dataset('name_{}'.format(i), data=name)

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
        # print(one_hot_encoded)
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
        names = []

        if file is None:
            file = self.h5_filename

        # Open the HDF5 file for reading
        with h5py.File(file, 'r') as hf:
            # Access the group containing the datasets
            group = hf[group_name]

            # Iterate over the datasets in the group
            for i in range(
                    len(group.keys()) // 4):  # Assuming pairs of coordinates and one_hot_trajs and atoms_nums datasets
                coordinates_array_key = 'coordinates_array_{}'.format(i)
                one_hot_traj_key = 'one_hot_traj_{}'.format(i)
                atoms_num_key = 'atoms_num_{}'.format(i)
                name_key = 'name_{}'.format(i)

                # Load coordinates array
                coordinates_array = np.array(group[coordinates_array_key])

                # Load one-hot trajectory array
                one_hot_traj = np.array(group[one_hot_traj_key])

                # Load numbers of atoms array
                atoms_num = np.array(group[atoms_num_key])
                name = np.array(group[name_key])

                # Append loaded arrays to the corresponding lists
                coordinates_arrays.append(coordinates_array)
                one_hot_trajs.append(one_hot_traj)
                atoms_nums.append(atoms_num)
                names.append(name)

        return coordinates_arrays, one_hot_trajs, atoms_nums, names


class TrajectoriesDataset_Efficient(Dataset):
    def __init__(
            self,
            cutoff,
            scale=1.0,
            augment=False,
            dataset=[],
            original_h5_file=None,
            smiles = "CC(C(=O)O)N" # change in future for multi-systems
    ):
        super(TrajectoriesDataset_Efficient, self).__init__()
        self.cutoff = cutoff
        self.scale = scale
        self.original_h5_file = original_h5_file
        self.augment = augment
        self.dataset = dataset
        self.indices_traj_frames = []
        self.h5_file = h5py.File(self.original_h5_file, 'r')
        self.smiles = smiles

        with h5py.File(self.original_h5_file, 'r') as hf:
            # Access the group containing the datasets
            group = hf['h5']
            self.number_trajs = int(len(group.keys()) // 4)
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
        name_key = 'name_{}'.format(system_idx)

        # Load coordinates array
        rmsf = calculate_rmsf(np.array(group[coordinates_array_key]))
        coordinates_array = np.array(group[coordinates_array_key][frame_idx])
        coordinates_array_plus1 = np.array(group[coordinates_array_key][frame_idx + 1])
        coordinates_array = torch.tensor(coordinates_array, dtype=torch.float)
        coordinates_array_plus1 = torch.tensor(coordinates_array_plus1, dtype=torch.float)

        # Load one-hot trajectory array
        one_hot_traj = np.array(group[one_hot_traj_key])
        one_hot_traj = torch.tensor(one_hot_traj, dtype=torch.float)  # .unsqueeze(0).expand([coordinates_array.shape[0], -1, -1])

        # Load numbers of atoms array
        atoms_num = np.array(group[atoms_num_key])
        
        # Covalent bond 
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol)
        covalent_edge_index, covalent_edge_type = rdmol_to_edge(mol)
        
        # Create virtual bond based on distance 
        # i, j = radius_graph(coordinates_array,
        #                     self.cutoff*self.scale,
        #                     batch=torch.zeros(coordinates_array.shape[0]))
        # edge_index = torch.stack([i, j])
        # edge_type = torch.zeros(edge_index.size(-1), dtype=torch.long)
        # i, j = radius_graph(coordinates_array_plus1,
        #                     self.cutoff * self.scale,
        #                     batch=torch.ones(coordinates_array_plus1.shape[0]))
        # edge_index_plus1 = torch.stack([i, j])
        # edge_type_plus1 = torch.zeros(edge_index_plus1.size(-1), dtype=torch.long)
        
        edge_index, edge_type = extend_to_radius_graph(
            coordinates_array,
            covalent_edge_index, 
            covalent_edge_type,
            self.cutoff * self.scale,
            torch.zeros(coordinates_array.shape[0]))
        edge_index_plus1, edge_type_plus1 = extend_to_radius_graph(
            coordinates_array_plus1,
            covalent_edge_index, 
            covalent_edge_type,
            self.cutoff * self.scale,
            torch.ones(coordinates_array_plus1.shape[0]))
     
        frame_idx = torch.tensor(frame_idx).repeat([coordinates_array.shape[0], 1])
        frame_idx_plus1 = frame_idx + 1

        data = Data(
            pos=coordinates_array,
            x=one_hot_traj,
            edge_index=edge_index,
            edge_type=edge_type,
            traj_idx=system_idx,
            frame_idx=frame_idx,
            atoms_num=atoms_num,
            name=np.array(group[name_key]),
            rmsf=rmsf
        )
        data = self.augment_edge(data)

        data_plus1 = Data(
            pos=coordinates_array_plus1,
            x=one_hot_traj,
            edge_index=edge_index_plus1,
            edge_type=edge_type_plus1,
            traj_idx=system_idx,
            frame_idx=frame_idx_plus1,
            atoms_num=atoms_num,
            name=np.array(group[name_key]),
            rmsf = rmsf
        )
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


if __name__ == '__main__':
    config = parse_toml_file('config.toml')
    # directory_path = config['directory_path']
    directory_path = 'data/sys_test'
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    device = config['device']
    num_frames_to_process = config['num_frames_to_process']
    # DataAnalysis(directory_path, num_frames_to_process, align=True).preprocess_coordinate_onehot()
    TrajsDataset = TrajectoriesDataset_Efficient(cutoff,
                                                 scale,
                                                 original_h5_file='data/sys_test/resname_unl.h5')
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
        print(len(a.frame_idx))
        print(b.frame_idx)
        print(a.rmsf)
        print(len(a.rmsf[0]))
        print(b.rmsf)
        print(a.batch)
        print(b.batch)
        break



