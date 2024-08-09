"""Create data loaders
"""

from torch_geometric.loader import DataLoader
from prepocessing.data_extend import TrajectoriesDataset_Efficient
from torch.utils.data import Dataset
import lightning as L
import torch
from torch_geometric.data import Data
from prepocessing.from_noe import rdmol_to_edge
from rdkit import Chem


class LitData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.scale = config["scale"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

    def train_dataloader(self):
        TrajsDataset = TrajectoriesDataset_Efficient(
            scale=self.scale,
            original_h5_file="/storage/florian/ziyu_project/resname_unl.h5",
        )
        # ind=10
        # (TrajsDataset[ind+1].pos- TrajsDataset[ind].pos)- TrajsDataset[ind].dd
        return DataLoader(
            TrajsDataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # ProteinAnalysis(val_path, num_frames_to_process).preprocess_coordinate_onehot()
        traj_dataset_val = TrajectoriesDataset_Efficient(
            scale=self.scale,
            original_h5_file="/storage/florian/ziyu_project/resname_unl.h5",
        )
        return DataLoader(
            traj_dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        # ProteinAnalysis(test_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset_test = TrajectoriesDataset_Efficient(
            scale=self.scale,
            original_h5_file="/storage/florian/ziyu_project/resname_unl.h5",
        )
        return DataLoader(
            TrajsDataset_test,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class TrajectoryData(Dataset):
    def __init__(self, pdb_path, trajectory_path, increments_path, selected_indices):
        super().__init__()
        self.trajectory_data = torch.load(trajectory_path)[selected_indices]
        self.increment_data = torch.load(increments_path)[selected_indices]
        mol = Chem.MolFromPDBFile(pdb_path)
        self.edge_index, self.edge_type = rdmol_to_edge(mol)

    def __getitem__(self, idx):
        data = Data(
            pos=self.trajectory_data[idx],
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            increments=self.increment_data[idx],
        )
        return data

    def __len__(self):
        return len(self.trajectory_data)


if __name__ == "__main__":
    pdb_path = "/storage/florian/ziyu_project/ala2/ala2.pdb"
    trajectory_path = (
        "/storage/florian/ziyu_project/ala2/merged_data/trajectory_merged_tensor.pt"
    )
    increments_path = (
        "/storage/florian/ziyu_project/ala2/merged_data/increments_merged_tensor.pt"
    )
    selected_indices = torch.arange(len(torch.load(trajectory_path)))
    data_set = TrajectoryData(
        pdb_path, trajectory_path, increments_path, selected_indices
    )
    print("done")
