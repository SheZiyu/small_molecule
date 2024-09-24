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
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as PGDataLoader
from omegaconf import OmegaConf


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


def atom_positions_from_mol(rdkit_mol):
    if rdkit_mol.GetNumConformers() > 0:
        conformer = rdkit_mol.GetConformer(0)  # Get the first conformer (index 0)
        # Step 2: Extract coordinates
        atom_coords = []
        for atom in rdkit_mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            atom_coords.append((pos.x, pos.y, pos.z))
            atom_coords = torch.tensor(atom_coords)
    else:
        print("The molecule does not have any conformers.")


class TrajectoryData(Dataset):
    def __init__(
        self, pdb_path, trajectory_path, increments_path, selected_indices=None
    ):
        super().__init__()
        if selected_indices is None:
            selected_indices = torch.arange(len(torch.load(trajectory_path)))
        self.trajectory_data = torch.load(trajectory_path)[selected_indices]
        self.increment_data = torch.load(increments_path)[selected_indices]
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        self.edge_index, self.edge_type, self.atom_type = rdmol_to_edge(mol)

    def __getitem__(self, idx):
        data = Data(
            pos=self.trajectory_data[idx],
            atom_type=self.atom_type,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            increments=self.increment_data[idx],
        )
        return data

    def __len__(self):
        return len(self.trajectory_data)


def create_dataloaders(config_data: DictConfig, config_data_loader: DictConfig):
    training_set = TrajectoryData(
        config_data.pdb_path,
        config_data.trajectory_path,
        config_data.increments_path,
    )
    train_loader_kwargs = {
        "batch_size": config_data_loader.batch_size,
        "num_workers": config_data_loader.num_workers,
        "pin_memory": config_data_loader.pin_memory,
    }
    train_dataloader = PGDataLoader(
        dataset=training_set, shuffle=True, **train_loader_kwargs
    )
    return train_dataloader


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
    # read yaml file:
    config = OmegaConf.load(
        "/home/florian/Repos/small_molecule/config/diffusion_egnn2.yaml"
    )
    train_dataloader = create_dataloaders(config)
    print("done")
