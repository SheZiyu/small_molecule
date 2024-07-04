import os
import glob
import time
import subprocess
import zipfile

import numpy as np
# import mpld3
import seaborn as sns
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch_geometric.data.batch import *
from torch_geometric.nn.pool import radius_graph

# from small_sys_gnn.model.solver1_gnn_lightning import *
# from small_sys_gnn.data.data_test import *
from prepocessing.preprocessing import parse_toml_file
from prepocessing.data_test import TrajectoriesDataset_Efficient, generate_test_dataset
from model.solver1_gnn_lightning import LitModel

def augment_edge(data):
    # Extract edge indices i, j from the data
    i, j = data.edge_index

    # Compute edge vectors (edge_vec) and edge lengths (edge_len)
    edge_vec = data.pos[j] - data.pos[i]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)

    # Concatenate edge vectors and edge lengths into edge_encoding
    # data.edge_encoding = torch.hstack([edge_vec, edge_len])
    data.edge_attr = edge_len
    return data

def extract_pdb_from_zip(zip_folder, target_name, output_folder):
    """Extract PDB file from a specific ZIP file."""
    for zip_file_name in os.listdir(zip_folder):
        if zip_file_name.endswith('.zip'):
            if target_name in zip_file_name:
                zip_file_path = os.path.join(zip_folder, zip_file_name)
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    for file_name in zip_ref.namelist():
                        if file_name.endswith('.pdb'):
                            zip_ref.extract(file_name, output_folder)
                            return os.path.join(output_folder, file_name)
    return None

def write_combined_pdb(original_pdb, new_coordinates, output_file):
    """Write the combined PDB file with new coordinates."""
    print(f"Writing PDB file: {output_file}")
    print(f"Number of new coordinates: {len(new_coordinates)}")
    with open(original_pdb, 'r') as original_file, open(output_file, 'w') as combined_file:
        atom_idx = 0
        for line in original_file:
            if line.startswith('ATOM'):
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
                    new_line = f"{line[:30]}{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}{line[54:]}"
                    combined_file.write(new_line)
                else:
                    combined_file.write(line)

            elif line.startswith('ATOM') and resname == "UNL":
                combined_file.write(line)
            else:
                combined_file.write(line)

    print(f"Finished writing PDB file: {output_file}")

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
    num_frames_to_process = 100 # config['num_frames_to_process']
    is_noise = True
    # DataAnalysis(directory_path, num_frames_to_process).preprocess_coordinate_onehot()
    # TrajsDataset = TrajectoriesDataset(
    #     False,
    #     directory_path,
    #     num_frames_to_process,
    #     cutoff=cutoff,
    #     scale=scale,
    #     file=None#'/data2/ziyu_project/trajs/not_(name_h*)_and_name_ca_ehgn.h5'
    # )
    # test_loader = generate_test_dataset(TrajsDataset, 1, num_workers)
    # print(sys.getsizeof(test_loader))
    TrajsDataset_test = TrajectoriesDataset_Efficient(cutoff=cutoff,
                                                      scale=scale,
                                                      original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_test/resname_unl.h5')
    print(TrajsDataset_test)

    test_loader = generate_test_dataset(TrajsDataset_test, 1, num_workers)
    Model = LitModel.load_from_checkpoint(
        '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_8_1-v28.ckpt', config=config)
    model = Model.model.to(device0)
    dpm = Model.dpm.to(device0)
    # torch.manual_seed(42)
    Model.to(device0).eval()

    output_dir = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_gnn_hybrid_0.0001_8_1'
    os.makedirs(output_dir, exist_ok=True)

    # Test the model on the test set
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 1):

            if idx == 1:
                name = data[0].name
                name = np.str_(name[0])[2:-1]
                folder = os.path.join(output_dir, name)
                os.makedirs(folder, exist_ok=True)
                pdb = extract_pdb_from_zip(directory_path, name, folder)
                gt_file = os.path.join(folder,
                                       'gt_{}.pdb'.format(
                                           name
                                       ))
                write_combined_pdb(pdb, data[0].pos.detach().cpu().clone().numpy() / 2.0, gt_file)
                if is_noise:
                    data[0].pos = torch.tensor(data[0].pos.to(device0)) + torch.randn_like(data[0].pos.to(device0))
                else:
                    data[0].pos = torch.tensor(data[0].pos.to(device0))

                for stop_idx in range(num_frames_to_process):
                    i, j = radius_graph(data[0].pos,
                                        r_min=torch.tensor(1.5).to(device0),
                                        r_max=torch.tensor(4.5).to(device0),
                                        batch=torch.zeros(data[0].pos.shape[0]).to(device0))
                    data[0].edge_index = torch.stack([i, j]).to(device0)
                    data[0] = data[0].to(device0)
                    data[0] = augment_edge(data[0])

                    dx_T = torch.randn_like(data[0].pos)
                    pos = dpm.adaptive_reverse_denoise(dx_T, model,
                                                       dpm.solver1, dpm.solver2, dpm.solver3,
                                                       order1=50, order2=25, order3=0, M=100,
                                                       edge_index=data[0].edge_index,
                                                       edge_attr=data[0].edge_attr,
                                                       h=data[0].x,
                                                       cond=data[0].pos)
                    data[0].pos = pos
                    # print('0: ', data[0].pos)

                    denoised_file = os.path.join(folder,
                                                 'denoised_{}_frame_{}_from_{}_to_{}.pdb'.format(
                                                     name,
                                                     idx,
                                                     stop_idx,
                                                     stop_idx + 1
                                                 ))
                    write_combined_pdb(pdb, pos.detach().cpu().clone().numpy() / 2.0, denoised_file)
                    # print('1: ', data[0].pos)

                # Execute the GROMACS commands every 1000 frames
                # Convert denoised PDB files to GRO files
                for pdb_file in glob.glob(os.path.join(folder, 'denoised*.pdb')):
                    gro_file = pdb_file.replace('.pdb', '.gro')
                    subprocess.run(
                        ['/home/ziyu/Downloads/gromacs-install/bin/gmx', 'editconf', '-f', pdb_file, '-o', gro_file])

                # Concatenate denoised GRO files into a single trajectory GRO file
                cat_command = 'cat {} > {}'.format(os.path.join(folder, 'denoised*.gro'),
                                                   os.path.join(folder, 'trajectory.gro'))
                subprocess.run(cat_command, shell=True)

                # Generate trajectory XTC file from the concatenated GRO file
                trjconv_command = '/home/ziyu/Downloads/gromacs-install/bin/gmx trjconv -f {} -s {} -o {}'.format(
                    os.path.join(folder, 'trajectory.gro'),
                    os.path.join(folder, 'denoised_{}_frame_1_from_0_to_1.pdb'.format(name)),
                    os.path.join(folder, 'trajectory.xtc')
                )
                # subprocess.run(trjconv_command, shell=True)

                # Automate the input of '3' when prompted to select the functional group
                proc = subprocess.Popen(trjconv_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
                stdout, stderr = proc.communicate(input='0\n')

                # Check if there were any errors
                if stderr:
                    print("Error:", stderr)


# for pdb_file in /home/ziyu/PycharmProjects/pythonProject/Ex/output_images_egnn_hybrid_0.001_1/4xqmA/denoised*.pdb; do     gro_file="${pdb_file%.pdb}.gro";     gmx editconf -f "$pdb_file" -o "$gro_file"; done
# cat /home/ziyu/PycharmProjects/pythonProject/Ex/output_images_egnn_hybrid_0.001_1/4xqmA/denoise*.gro > /home/ziyu/PycharmProjects/pythonProject/Ex/output_images_egnn_hybrid_0.001_1/4xqmA/trajectory.gro
# gmx trjconv -f /home/ziyu/PycharmProjects/pythonProject/Ex/output_images_egnn_hybrid_0.001_1/4xqmA/trajectory.gro -s /home/ziyu/PycharmProjects/pythonProject/Ex/output_images_egnn_hybrid_0.001_1/4xqmA/denoised_4xqmA_from_frame_[0]_to_[1]_1.pdb -o /home/ziyu/PycharmProjects/pythonProject/Ex/output_images_egnn_hybrid_0.001_1/4xqmA/trajectory.xtc
# gmx make_ndx -f /home/ziyu/PycharmProjects/pythonProject/Ex/test/4xqmA/4xqm_A_prod_R1.tpr -o index.ndx
#
# [  40   47   50   52   57   58   62   64   70   74   76   77   78   81
#    83   84   85   86   87   90   93   94   97  102  103  105  106  110
#   111  113  116  120  127  129  130  136  137  138  145  147  148  151
#   154  155  156  157  161  170  171  184  188  194  195  196  198  199
#   200  203  205  209  211  212  213  217  228  229  232  235  239  242
#   243  246  249  258  260  262  264  267  271  274  275  282  284  288
#   301  306  313  314  316  324  331  341  344  345  348  350  354  357
#   360  366  380  386  392  393  399  410  422  463  484  506  511  525
#   561  567  574  598  613  688  839 1039]

