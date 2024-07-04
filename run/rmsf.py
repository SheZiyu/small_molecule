import os
import time
import subprocess

import numpy as np
# import mpld3
import seaborn as sns
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch_geometric.data.batch import *

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

from torch_geometric.nn.pool import radius_graph

# from small_sys_gnn.model.solver1_gnn_lightning import *
# from small_sys_gnn.data.data_test import *
from prepocessing.preprocessing import parse_toml_file
from prepocessing.data_test import TrajectoriesDataset_Efficient, generate_test_dataset, calculate_rmsf
from model.solver1_gnn_lightning import LitModel

def process_folder(folder, folder_path):
    pdb_files = [file for file in os.listdir(folder_path) if file == 'gt_1a10A.pdb']
    xtc_files = [file for file in os.listdir(folder_path) if file == 'output_trajectory.xtc']

    if len(pdb_files) != 1 or len(xtc_files) != 1:
        print(f"Folder {folder_path} does not contain exactly one PDB and one XTC file.")
        return

    pdb_file = os.path.join(folder_path, pdb_files[0])
    xtc_file = os.path.join(folder_path, xtc_files[0])

    # Load trajectory data from XTC file using MDAnalysis
    u = mda.Universe(pdb_file, xtc_file)
    average = align.AverageStructure(u, u, select='resname UNL', ref_frame=0).run()
    ref = average.results.universe
    aligner = align.AlignTraj(u, ref, select='resname UNL', in_memory=True).run()
    # c_alphas = u.select_atoms('name CA')
    # R = rms.RMSF(c_alphas).run()
    # rmsf = R.results.rmsf
    coordinates_array = [u.select_atoms('resname UNL').positions for ts in u.trajectory]
    # Calculate RMSF
    rmsf = calculate_rmsf(coordinates_array)
    return rmsf

def process_directory(directory_path):
    rmsf_results = {}

    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if os.path.isdir(folder_path):
            rmsf = process_folder(folder, folder_path)
            rmsf_results[folder] = rmsf
    return rmsf_results

def plot_scatter(pred_rmsf, real_rmsf, output_dir, name):
    plt.figure(figsize=(8, 6))
    assert pred_rmsf.shape[0] == real_rmsf.shape[0]
    num_atoms = pred_rmsf.shape[0]
    plt.scatter(range(num_atoms), pred_rmsf, c='blue', label='Predicted RMSF')
    plt.scatter(range(num_atoms), real_rmsf, c='orange', label='Real RMSF')
    plt.xlabel('Number of Atoms')
    plt.ylabel('RMSF ($\AA$)')
    plt.title('Predicted and Real RMSF of {}'.format(name))
    plt.legend()
    # plt.grid(True)
    plt.savefig(os.path.join(output_dir, name))
    # plt.show()

if __name__ == '__main__':
    print(os.getcwd())
    config = parse_toml_file('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/config.toml')
    directory_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_test'
    data_dir = config['data_dir']
    dataset_location = os.path.join(data_dir, 'dataset.pickle')
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_CUDA_ALLOC_SYNC"] = "1"
    device0 = torch.device("cuda:0")
    print(torch.cuda.current_device())
    torch.cuda.set_device(device0)
    print(torch.cuda.current_device())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_splits = config['num_splits']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    patience = config['patience']
    num_frames_to_process = 10000
    TrajsDataset_test = TrajectoriesDataset_Efficient(cutoff=cutoff,
                                                      scale=scale,
                                                      original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_test/resname_unl.h5')
    print(TrajsDataset_test)

    test_loader = generate_test_dataset(TrajsDataset_test, 1, num_workers)
    Model = LitModel.load_from_checkpoint(
        '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_8_1-v12.ckpt',
        config=config)
    model = Model.model.to(device0)
    dpm = Model.dpm.to(device0)
    # torch.manual_seed(42)
    Model.to(device0).eval()

    input_dir = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_hybrid_0.0001_8_1'
    os.makedirs(input_dir, exist_ok=True)
    rmsf_results = process_directory(input_dir)
    output_dir = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_hybrid_0.0001_8_1_rmsf'
    os.makedirs(output_dir, exist_ok=True)

    names_list = []

    # Test the model on the test set
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if idx % num_frames_to_process == 0:
                data = data[0].to(device0)
                name = data.name
                name = np.str_(name[0])[2:-1]
                names_list.append(name)
                pred_rmsf = rmsf_results.get(name) # Get predicted RMSF or None if not found
                if pred_rmsf is not None:
                    real_rmsf = data.rmsf[0] / 2.0
                    plot_scatter(pred_rmsf, real_rmsf, output_dir, name)




