import os
import numpy as np
import torch
from rdkit import Chem
from prepocessing.from_noe import rdmol_to_edge
from prepocessing.transforms import extend_to_radius_graph
from prepocessing.preprocessing import parse_toml_file
from prepocessing.data_test_extend import (
    TrajectoriesDataset_Efficient,
    generate_test_dataset,
)
from utils.auxiliary import augment_edge, extract_pdb_from_zip, write_combined_pdb


if __name__ == "__main__":
    print(os.getcwd())
    config = parse_toml_file("config_gnn.toml")
    directory_path = "data/sys_test"
    cutoff = config["cutoff"]
    scale = config["scale"]
    node_dim = config["node_dim"]
    edge_dim = config["edge_dim"]
    vector_dim = config["vector_dim"]
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_CUDA_ALLOC_SYNC"] = "1"
    device0 = torch.device("cuda:0")
    print(torch.cuda.current_device())
    torch.cuda.set_device(device0)
    print(torch.cuda.current_device())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_splits = config["num_splits"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    num_frames_to_process = 10  # config['num_frames_to_process']

    TrajsDataset_test = TrajectoriesDataset_Efficient(
        cutoff=cutoff, scale=scale, original_h5_file="data/sys_test/resname_unl.h5"
    )
    print(TrajsDataset_test)

    test_loader = generate_test_dataset(TrajsDataset_test, 1, num_workers)
    Model = LitModel.load_from_checkpoint(
        "/home/ziyu/repos/small_molecule/output/solver1_gnn_test_beta_8_1.ckpt",
        config=config,
    )
    model = Model.model.to(device0)
    print(model.time_embedding.B)
    # exit()
    dpm = Model.dpm.to(device0)
    # torch.manual_seed(42)
    Model.to(device0).eval()

    output_dir = "/home/ziyu/repos/small_molecule/output_images_gnn_solver1"
    os.makedirs(output_dir, exist_ok=True)

    # Test the model on the test set
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 1):
            # Initialize an empty list to store numpy arrays

            if idx == 1:
                # Covalent bond
                smiles = "CC(C(=O)O)N"  # change in future for multi-systems
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                covalent_edge_index, covalent_edge_type = rdmol_to_edge(mol)

                name = data[0].name
                name = np.str_(name[0])[2:-1]
                folder = os.path.join(output_dir, name)
                os.makedirs(folder, exist_ok=True)
                pdb = extract_pdb_from_zip(directory_path, name, folder)
                gt_file = os.path.join(folder, "gt_{}.pdb".format(name))
                write_combined_pdb(
                    pdb, data[0].pos.detach().cpu().clone().numpy() / 2.0, gt_file
                )

                numpy_arrays = []
                numpy_arrays.append(data[0].pos / 2.0)
                print(data[0].pos)

                for stop_idx in range(num_frames_to_process):
                    x_T = torch.randn_like(data[0].pos.to(device0))
                    data[0].edge_index, data[0].edge_type = extend_to_radius_graph(
                        data[0].pos.to(device0),
                        covalent_edge_index.to(device0),
                        covalent_edge_type.to(device0),
                        torch.tensor(4.5).to(device0),
                        torch.zeros(data[0].pos.shape[0]).to(device0),
                    )
                    data[0] = data[0].to(device0)
                    data[0] = augment_edge(data[0])

                    pos = dpm.adaptive_reverse_denoise(
                        x_T,
                        model,
                        dpm.solver1,
                        dpm.solver2,
                        dpm.solver3,
                        order1=500,
                        order2=0,
                        order3=0,
                        M=500,
                        edge_index=data[0].edge_index,
                        edge_type=data[0].edge_type,
                        edge_attr=data[0].edge_attr,
                        h=data[0].x,
                        cond=data[0].pos,
                    )

                    print(data[0].pos)
                    data[0].pos = pos
                    print(data[0].pos)

                    # Convert the tensor to numpy array and save it to the list
                    numpy_array = data[0].pos.detach().cpu().numpy() / 2.0
                    numpy_arrays.append(numpy_array)

                    # Save the numpy array to a file
                    np.save(os.path.join(folder, f"pos_{stop_idx}.npy"), numpy_array)

                # After the loop, you can also save all arrays as a single numpy array if needed
                np.save(
                    os.path.join(folder, "all_positions.npy"), np.array(numpy_arrays)
                )
