import os
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ['NCCL_P2P_DISABLE'] = "1"
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch_ema import ExponentialMovingAverage

import math
import copy
from pathlib import Path
from random import randint, random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import datetime
import subprocess

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import torch.optim as optim
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel

from model.base import dynamicsGNN

torch.set_num_threads(64)
torch.manual_seed(42)

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

class DDPM(nn.Module):
    """
    Implementation of the Denoising Diffusion Probabilistic Model (DDPM).

    Parameters:
    - schedule (str): The scheduling method for the noise levels, e.g., 'linear'.
    - t_min (float): The minimum time step for the diffusion process, used to control the start of the noise schedule.
    """

    def __init__(self, gamma_0=-10.0, gamma_1=10.0):
        super().__init__()

        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1

    def gamma(self, t):
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * t

    def forward_noise(self, x, t):
        """
        Applies forward noise to the input tensor x at time t, simulating the diffusion process.

        Parameters:
        - x (Tensor): The input tensor.
        - t (Tensor): The time step tensor.

        Returns:
        - Tuple[Tensor, Tensor]: The noised tensor and the generated noise tensor.
        """
        gamma_t = self.gamma(t)
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        eps = torch.randn_like(x)  # Generates random noise.
        x_t = alpha_t * x + sigma_t * eps  # Applies the diffusion process.
        return x_t, eps

class LitData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cutoff = config['cutoff']
        self.scale = config['scale']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

    def train_dataloader(self):
        # ProteinAnalysis(directory_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset = TrajectoriesDataset_Efficient(cutoff=self.cutoff,
                                                     scale=self.scale,
                                                     original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_gnn/resname_unl.h5')
        return DataLoader(TrajsDataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        # ProteinAnalysis(val_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset_val = TrajectoriesDataset_Efficient(cutoff=self.cutoff,
                                                         scale=self.scale,
                                                         augment=False,
                                                         original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_gnn/resname_unl.h5')
        return DataLoader(TrajsDataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_steps = 50

        #  Initialize your model, optimizer, scheduler, criterion, etc
        self.model = dynamicsGNN
        self.dpm = DDPM()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = lr_scheduler.LinearLR(self.optimizer,
                                               start_factor=1.0 / 1000,
                                               total_iters=1000 - 1)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)

        self.criterion = nn.MSELoss()
        # self.criterion = nn.KLDivLoss(reduction='batchmean')

        self.save_hyperparameters()

    def diffusion_loss(self, batch, t):
        with torch.enable_grad():
            t = t.clone().detach().requires_grad_(True)
            gamma_t = self.dpm.gamma(t)
            grad_gamma_t = torch.autograd.grad(gamma_t.sum(), t, create_graph=True)[0]
        gamma_t = gamma_t.detach()
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        eps = subtract_means(torch.randn_like(batch.dd), batch.batch)
        noise_dd = alpha_t * batch.dd + sigma_t * eps  # Applies the diffusion process.
        # noise_pred = self.model(
        #     t=t,
        #     bnd_index=batch.edge_index,
        #     bond_encoding=batch.edge_attr,
        #     noised_dx=noise_dd,
        #     node_encoding=batch.x,
        # )
        # noise_pred = self.model(
        #     t=gamma_t,
        #     bnd_index=batch.edge_index,
        #     bnd_type=batch.edge_type,
        #     bond_encoding=batch.edge_attr,
        #     noised_dx=noise_dd,
        #     node_encoding=batch.x
        # )
        original_nodes = batch.pos
        pred_perturbed_nodes = self.model(
            t=gamma_t,
            h=batch.x,
            original_nodes=original_nodes,
            perturbed_nodes=original_nodes+noise_dd,
            edge_index=batch.edge_index,
            node2graph=batch.batch,
            edge_type=batch.edge_type
        )[-1]

        loss = (
                0.5
                * grad_gamma_t
                # * self.criterion(pred_perturbed_nodes - batch.pos, eps)
                * self.criterion(subtract_means(pred_perturbed_nodes - original_nodes, batch.batch), eps)
        )

        return loss
        # loss = (
        #         0.5
        #         * grad_gamma_t
        #         * self.criterion(noise_pred, eps)
        # )
        #
        # return loss

    def training_step(self, batch):
        # Step 1: Determine unique graph indices within the batch
        unique_graph_indices = torch.unique(batch.batch)
        # Step 2: Determine noise levels based on the number of unique graph indices
        num_unique_graphs = len(unique_graph_indices)
        t = torch.rand(num_unique_graphs, 1)
        # Step 3: Assign t values to graphs based on batch index
        t_assigned = t.to(self.device)[batch.batch]
        loss = self.diffusion_loss(batch, t_assigned)
        loss = torch.mean(loss)
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizer.param_groups[0]['lr'])

        return loss

    def validation_step(self, batch):
        # Step 1: Determine unique graph indices within the batch
        unique_graph_indices = torch.unique(batch.batch)
        # Step 2: Determine noise levels based on the number of unique graph indices
        num_unique_graphs = len(unique_graph_indices)
        t = torch.rand(num_unique_graphs, 1)
        # t = 0.0 + torch.rand(num_unique_graphs, 1) * 0.5
        # Step 3: Assign t values to graphs based on batch index
        t_assigned = t.to(self.device)[batch.batch]

        with self.ema.average_parameters():
            loss = self.diffusion_loss(batch, t_assigned)
        loss = torch.mean(loss)
        # Log the validation loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return {"optimizer": self.optimizer,
                "lr_scheduler": self.scheduler}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

    @torch.inference_mode()
    def sample(self, batch, return_all=False):
        time_steps = torch.linspace(
            1.0, 0.0, steps=self.num_steps + 1, device=self.device
        )
        dx_T_list = []
        dx_T = subtract_means(torch.randn_like(batch.pos), batch.batch)
        dx_T_list.append(dx_T)

        for i in range(self.num_steps):
            t = torch.broadcast_to(time_steps[i], (batch.pos.shape[0], 1))
            s = torch.broadcast_to(time_steps[i + 1], (batch.pos.shape[0], 1))
            gamma_t = self.dpm.gamma(t)
            gamma_s = self.dpm.gamma(s)
            alpha_sq_t = torch.sigmoid(-gamma_t)
            alpha_sq_s = torch.sigmoid(-gamma_s)
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
            c = -torch.expm1(gamma_s - gamma_t)

            # noise_pred = self.model(
            #     t=gamma_t,
            #     bnd_index=batch.edge_index,
            #     bnd_type=batch.edge_type,
            #     bond_encoding=batch.edge_attr,
            #     noised_dx=dx_T,
            #     node_encoding=batch.x
            # )
            noise_pred = self.model(
                t=gamma_t,
                h=batch.x,
                original_nodes=batch.pos,
                perturbed_nodes=batch.pos+dx_T,
                edge_index=batch.edge_index,
                node2graph=batch.batch,
                edge_type=batch.edge_type
            )[-1]
            noise_pred = subtract_means(noise_pred-batch.pos, batch.batch)

            dx_T_mean = torch.sqrt(alpha_sq_s / alpha_sq_t) * (
                    dx_T - (sigma_t * c) * noise_pred
            )
            dx_T_std = torch.sqrt((1.0 - alpha_sq_s) * c)
            noise = subtract_means(torch.randn_like(dx_T), batch.batch)
            dx_T = dx_T_mean + dx_T_std * noise
            dx_T_list.append(dx_T)

        gamma_0 = self.dpm.gamma(dx_T.new_zeros(dx_T.shape[0], 1))
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        sigma_0 = torch.sqrt(torch.sigmoid(gamma_0))
        noise = subtract_means(torch.randn_like(dx_T), batch.batch)
        dx_T = (dx_T + sigma_0 * noise) / alpha_0
        dx_T_list.append(dx_T)
        if return_all:
            return dx_T_list
        else:
            return dx_T

if __name__ == "__main__":
    print(os.getcwd())
    torch.set_float32_matmul_precision('medium')
    config = parse_toml_file('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/config_gnn.toml')
    directory_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_gnn'
    val_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_gnn'
    data_dir = config['data_dir']
    dataset_location = os.path.join(data_dir, 'dataset.pickle')
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    num_splits = config['num_splits']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    patience = config['patience']
    num_frames_to_process = config['num_frames_to_process']

    # # ProteinAnalysis(directory_path, num_frames_to_process).preprocess_coordinate_onehot()
    # TrajsDataset = TrajectoriesDataset_Efficient(cutoff=cutoff,
    #                                              scale=scale,
    #                                              original_h5_file='/data2/ziyu_project/trajs/not_(name_h*)_and_name_ca.h5')
    # print(TrajsDataset)
    # train_loader = generate_train_dataset(TrajsDataset, batch_size, num_workers)
    # print(sys.getsizeof(train_loader))
    #
    # # ProteinAnalysis(val_path, num_frames_to_process).preprocess_coordinate_onehot()
    # TrajsDataset_val = TrajectoriesDataset_Efficient(cutoff=cutoff,
    #                                                  scale=scale,
    #                                                  original_h5_file='/data2/ziyu_project/trajs_real_val/not_(name_h*)_and_name_ca.h5')
    # print(TrajsDataset_val)
    # val_loader = generate_val_dataset(TrajsDataset_val, batch_size, num_workers)
    # print(sys.getsizeof(val_loader))

    datamodule = LitData(config)
    model = LitModel(config)
    # model = LitModel.load_from_checkpoint('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_20_0.ckpt', config=config)
    # print(model.model.time_embedding.B)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        mode='min',
        verbose=True
    )

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        # dirpath='/home/ziyu/PycharmProjects/pythonProject/Ex/output',
        filename='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_20_0',
        monitor='val_loss',
        mode='min'
    )

    # from lightning.pytorch.loggers import CSVLogger
    # logger = CSVLogger("logs", name="my_exp_name")

    # Initialize Trainer with early stopping callback and model checkpoint callback
    trainer = L.Trainer(
        devices=[0,1,2,3],
        accelerator="cuda",
        max_epochs=config['num_epochs'],
        callbacks=[early_stop_callback, checkpoint_callback],
        strategy="ddp_find_unused_parameters_false",
        # strategy=DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=14400)),
        # precision="bf16",
        # precision=16,
        log_every_n_steps=4,
        # logger=TensorBoardLogger(save_dir='/home/ziyu/PycharmProjects/pythonProject/Ex/output', name='gnn')
    )

    # Train the model
    trainer.fit(model=model,
                datamodule=datamodule)












