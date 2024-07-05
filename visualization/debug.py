import os
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
# from torch_geometric.data.batch import *
from torch_geometric.nn.pool import radius_graph

# from small_sys_gnn.model.solver1_gnn_lightning import *
# from small_sys_gnn.data.data_test import *
from utils.auxiliary import augment_edge, extract_pdb_from_zip, write_combined_pdb
from prepocessing.preprocessing import parse_toml_file
from prepocessing.data_test import TrajectoriesDataset_Efficient, generate_test_dataset
from model.solver1_gnn_lightning import LitModel

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
        '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_8_1-v11.ckpt', config=config)
    print(Model.model.time_embedding.B)
    model = Model.model.to(device0)
    print(Model.model.time_embedding.B)
    dpm = Model.dpm.to(device0)
    # torch.manual_seed(42)
    Model.to(device0).eval()


    output_dir = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output_images_debug'
    os.makedirs(output_dir, exist_ok=True)

    # Test the model on the test set
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 1):

            if idx == 1:
                name = data[0].name
                name = np.str_(name[0])[2:-1]

                # # Step 1: Determine unique graph indices within the batch
                # unique_graph_indices = torch.unique(batch_idx)
                # # Step 2: Determine noise levels based on the number of unique graph indices
                # num_unique_graphs = len(unique_graph_indices)
                # lambdas = torch.empty(num_unique_graphs, 1, device=data[0].pos.device).uniform_(dpm.lambda_max,
                #                                                                                 dpm.lambda_min)

                folder = os.path.join(output_dir, name)
                os.makedirs(folder, exist_ok=True)
                pdb = extract_pdb_from_zip(directory_path, name, folder)
                deformed_file = os.path.join(folder,
                                             'deformed_{}.pdb'.format(
                                                 name
                                             ))
                write_combined_pdb(pdb, data[0].pos.detach().cpu().clone().numpy() / 2.0, deformed_file)

                for idt, t in enumerate(torch.linspace(1e-3, 1, 10)):
                    data[0] = data[0].to(device0)
                    batch_idx = data[0].batch.to(device0)
                    t = torch.tensor([[t]])
                    alpha_t = dpm.alpha_t(t)
                    sigma_t = dpm.sigma_t(t)
                    lambda_t = dpm.lambda_t(t)
                    print(t, alpha_t, sigma_t, lambda_t)

                    pos, _ = model(t=t.to(device0),
                                   edge_index=data[0].edge_index.to(device0),
                                   edge_attr=data[0].edge_attr.to(device0),
                                   x=data[0].pos.to(device0),
                                   h=data[0].x.to(device0),
                                   cond=data[0].pos.to(device0))
                    print(data[0].pos)
                    data[0].pos = pos
                    print(data[0].pos)

                    i, j = radius_graph(data[0].pos,
                                        r_min=torch.tensor(1.5).to(device0),
                                        r_max=torch.tensor(4.5).to(device0),
                                        batch=torch.zeros(data[0].pos.shape[0]).to(device0))
                    data[0].edge_index = torch.stack([i, j])
                    data[0] = augment_edge(data[0])
                    pos = pos.detach().cpu().clone().numpy() / 2.0
                    deformed_file = os.path.join(folder,
                                                 'deformed_{}_frame_{}_from_{}_to_{}_{:.2f}.pdb'.format(
                                                     name,
                                                     np.unique(data[0].detach().cpu().clone().frame_idx),
                                                     idt,
                                                     idt+1,
                                                     t.detach().cpu().clone().numpy()[0][0]
                                                 ))
                    write_combined_pdb(pdb, pos, deformed_file)

