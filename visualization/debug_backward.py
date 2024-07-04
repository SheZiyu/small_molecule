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

from small_sys_gnn.model.solver1_gnn_lightning import *
from small_sys_gnn.data.data_test import *

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
    num_frames_to_process = 1000 # config['num_frames_to_process']
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
    # Model = LitModel(config)
    Model = LitModel.load_from_checkpoint(
        '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_8_1-v34.ckpt', config=config)
    print(Model.model.time_embedding.B)
    # torch.manual_seed(42)
    model = Model.model.to(device0).eval()
    dpm = Model.dpm.to(device0)

    output_dir = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_debug_backward'
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

                numpy_arrays = []

                for stop_idx in range(num_frames_to_process):
                    if stop_idx == 0:
                        x_T = torch.randn_like(data[0].pos.to(device0))
                        i, j = radius_graph(data[0].pos.to(device0),
                                            torch.tensor(4.5).to(device0),
                                            batch=torch.zeros(data[0].pos.shape[0]).to(device0))
                        data[0].edge_index = torch.stack([i, j]).to(device0)
                        data[0] = data[0].to(device0)
                        data[0] = augment_edge(data[0])

                        pos_list = dpm.adaptive_reverse_denoise(x_T, model,
                                                                dpm.solver1, dpm.solver2, dpm.solver3,
                                                                order1=500, order2=0, order3=0, M=500,
                                                                edge_index=data[0].edge_index,
                                                                edge_attr=data[0].edge_attr,
                                                                h=data[0].x,
                                                                cond=data[0].pos,
                                                                return_all=True)

                        pos_list = pos_list.detach().cpu().numpy() / 2.0
                        print(pos_list)

                        # After the loop, you can also save all arrays as a single numpy array if needed
                        np.save(os.path.join(folder, 'all_positions.npy'), np.array(pos_list))