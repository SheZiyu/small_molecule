'''
extract coordinates, atom types, atomic numbers from trajectory
plot noise distribution using matplotlib
'''
import argparse
import os
import copy
import glob
import pickle
import time
from tqdm.notebook import tqdm
import toml
import zipfile
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

# import mdtraj as md
import MDAnalysis as mda
import MDAnalysis.analysis.align as align

from small_sys_gnn.data.load import *

def parse_toml_file(filename):
    with open(filename, 'r') as f:
        return toml.load(f)

def find_files(directory, extension):
    path = os.path.join(directory, '*.{}'.format(extension))
    return glob.glob(path)

def pickle_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print('Object pickled successfully to {}'.format(filename))

def unpickle_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def find_rmsf_tsv_files(directory):
    '''
    Find files with the keyword "RMSF" and the extension ".tsv" in the specified directory.

    Parameters:
    - directory: Path to the directory where files will be searched.

    Returns:
    - file_paths: List of file paths matching the criteria.
    '''
    search_pattern = '{}/*RMSF*.tsv'.format(directory)
    file_paths = glob.glob(search_pattern)
    return file_paths

def load_tsv_file(tsv_file_path):
    '''
    Load data from a TSV file into a pandas DataFrame.

    Parameters:
    - tsv_file_path: Path to the TSV file.

    Returns:
    - data_array: NumPy array containing the data.
    '''
    data_frame = pd.read_csv(tsv_file_path, sep='\t')
    data_array = data_frame.values
    return data_array[:, 1:]

def one_hot_encode_atoms(u, selection):
    # Select atoms based on the given selection
    selected_atoms = u.select_atoms(selection)

    # Get the unique atom names
    unique_atom_names = np.unique(selected_atoms.names)

    # Create a dictionary mapping each atom name to its one-hot encoded index
    atom_to_index = {atom: i for i, atom in enumerate(unique_atom_names)}

    # Initialize an array to store the one-hot encoded representations
    one_hot_encoded = np.zeros((len(selected_atoms), len(unique_atom_names)))

    # Iterate through the selected atoms and set the corresponding one-hot encoded values
    for i, atom in enumerate(selected_atoms):
        atom_name = atom.name
        one_hot_encoded[i, atom_to_index[atom_name]] = 1
    return one_hot_encoded

def load_coordinate_onehot_name(directory_path, selection='name CA and not (name H*)'):
    '''
    Load coordinates and RMSF and one-hot encodings for each protein system based on C-alpha atoms.

    Parameters:
    - directory_path: Path to the directory containing zip files for each protein system.
    - selection: Name of selection.

    Returns:
    # - rmsf_values: List of RMSF values for each protein system.
    - one_hot_trajs: List of one-hot encodings for each protein system.
    '''

    # rmsf_values = []
    one_hot_trajs = []
    # Iterate over each protein system in the directory
    for zip_file_name in os.listdir(directory_path):
        if zip_file_name.endswith('.zip'):
            system_path = os.path.join(directory_path, zip_file_name)

            # Create a temporary directory with the same name as the zip file (without extension)
            temp_dir = os.path.splitext(zip_file_name)[0]
            temp_dir_path = os.path.join(directory_path, temp_dir)
            os.makedirs(temp_dir_path, exist_ok=True)

            # Extract contents of the zip file to the temporary directory
            with zipfile.ZipFile(system_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir_path)

            # # Load true RMSF
            # rmsf_path = find_rmsf_tsv_files(temp_dir_path)[0]
            # rmsf_true = load_tsv_file(rmsf_path)
            #
            # # Append the true RMSF values to the list for the current replicate
            # rmsf_values.extend(np.split(rmsf_true, rmsf_true.shape[1], axis=1))

            # Check if the directory contains trajectory and topology files
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
                    u = mda.Universe(topology_path, trajectory_path)
                    one_hot_encoded_atoms = one_hot_encode_atoms(u, selection=selection)
                    one_hot_trajs.append(one_hot_encoded_atoms)
                except Exception as e:
                    print('Error loading trajectory and topology: {}'.format(e))

                # Initialize an empty array to store coordinates
                coordinates_array = np.empty((len(u.trajectory), u.select_atoms(selection).n_atoms, 3))

                # Iterate through the trajectory frames and store coordinates
                for i, ts in enumerate(u.trajectory):
                    coordinates_array[i] = u.select_atoms(selection).positions

                # Save the NumPy array to a file with a trajectory-specific name
                output_filename = os.path.join(
                    directory_path,
                    'coordinates_{}_{}.npy'.format(selection.replace(' ', '_').lower(), xtc_file[:-4])
                )
                np.save(output_filename, coordinates_array)

            # Clean up: Remove the temporary extracted directory
            shutil.rmtree(temp_dir_path)
    # return rmsf_values, one_hot_trajs
    return one_hot_trajs

if __name__ == '__main__':
    start = time.time()
    directory_path = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/task2'
    one_hot_trajs = load_coordinate_onehot_name(directory_path)
    # print(rmsf_values[0].shape)
    print(one_hot_trajs[0].shape)
    print('Time used in loading data: {}'.format(time.time() - start))







