import lightning as L
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

from small_sys_gnn.model.base import *

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

    def __init__(self, schedule='linear', t_min=1e-3, beta_0=0.1, beta_1=8):
        super().__init__()
        self.schedule = schedule
        self.t_min = t_min

        # Define the noise schedule parameters based on the chosen schedule.
        if schedule == 'linear':
             # Parameters defining the linear schedule for noise addition.
            self.beta_0, self.beta_1 = beta_0, beta_1
            self.t_max = 1  # Maximum time step, representing the end of the diffusion process.

            # Alpha (α): Variance retention coefficient at each time step.
            # Controls the proportion of the original signal retained during diffusion.
            self.alpha_t = lambda t: torch.exp(-(beta_0 * t / 2 + t.pow(2) / 4 * (beta_1 - beta_0)))

            # Lambda (λ) and Time (t) Transformation Functions:
            # Used for reparameterizing the diffusion schedule for efficiency or performance.
            self.t_lambda = lambda lmd: 2 * torch.log(torch.exp(-2 * lmd) + 1) / (torch.sqrt(
                beta_0 ** 2 + (beta_1 - beta_0) * 2 * torch.log(torch.exp(-2 * lmd) + 1)) + beta_0)

        # Sigma (σ): Standard deviation of the noise added at each time step.
        # Indicates the noise level introduced to the data based on α(t).
        self.sigma_t = lambda t: torch.sqrt(1 - self.alpha_t(t) ** 2)

        # Lambda (λ) Function: Represents a transformed time variable to parameterize the diffusion process.
        self.lambda_t = lambda t: torch.log(self.alpha_t(t) / self.sigma_t(t))

        # Minimum and maximum λ values, setting the bounds for the diffusion schedule.
        self.lambda_min = self.lambda_t(torch.tensor(self.t_min)).item()
        self.lambda_max = self.lambda_t(torch.tensor(self.t_max)).item()

    def ve_noise(self, x, t, degree=0.01):
        """
        Applies forward noise to the input tensor x at time t, simulating the diffusion process.

        Parameters:
        - x (Tensor): The input tensor.
        - t (Tensor): The time step tensor.

        Returns:
        - Tuple[Tensor, Tensor]: The noised tensor and the generated noise tensor.
        """
        eps = degree * torch.randn_like(x)  # Generates random noise.
        x_t = x + self.sigma_t(t) * eps  # Applies the diffusion process.
        return x_t, eps

    def forward_noise(self, x, t):
        """
        Applies forward noise to the input tensor x at time t, simulating the diffusion process.

        Parameters:
        - x (Tensor): The input tensor.
        - t (Tensor): The time step tensor.

        Returns:
        - Tuple[Tensor, Tensor]: The noised tensor and the generated noise tensor.
        """
        eps = torch.randn_like(x)  # Generates random noise.
        x_t = self.alpha_t(t) * x + self.sigma_t(t) * eps  # Applies the diffusion process.
        return x_t, eps

    def reverse_t(self, x_T, M=30):
        """
        Performs the reverse denoising process, reconstructing the data from its noised state.

        Parameters:
        - x_T (Tensor): The tensor at the final time step T.
        - noise_net (callable): The noise network function, predicting noise to be subtracted.
        - solver (callable): The solver function for the reverse process, optimizing denoising steps.
        - M (int): The number of steps in the reverse process. Default is 20.

        Returns:
        - Tensor: The stack of denoised tensors at each step.
        """
        # Compute λ values evenly spaced between the max and min, dictating the denoising schedule.
        lambdas = torch.linspace(self.lambda_max, self.lambda_min, M + 1, device=x_T.device)#.view(-1, 1, 1)
        # print(lambdas)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        # print(t)
        return t

    def reverse_ode(self, x_T, noise_net, M=30, return_all=False, **kwargs):
        """
        Performs the reverse denoising process, reconstructing the data from its noised state.

        Parameters:
        - x_T (Tensor): The tensor at the final time step T.
        - noise_net (callable): The noise network function, predicting noise to be subtracted.
        - solver (callable): The solver function for the reverse process, optimizing denoising steps.
        - M (int): The number of steps in the reverse process. Default is 20.

        Returns:
        - Tensor: The stack of denoised tensors at each step.
        """
        # Compute λ values evenly spaced between the max and min, dictating the denoising schedule.
        lambdas = torch.linspace(self.lambda_max, self.lambda_min, M + 1, device=x_T.device)#.view(-1, 1, 1)
        # print(lambdas)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        # print(t)
        x = x_T
        xs = []  # Stores the denoised states at each step.
        for i in range(len(t)):
            x = noise_net(x=x, t=t[i][None], **kwargs)[0] # Apply the reverse denoising solver.
            # print(x)
            xs.append(x.clone())
        if return_all:
            return torch.stack(xs)
        else:
            return x

    def reverse_denoise(self, x_T, noise_net, solver, order=1, M=30, return_all=False, **kwargs):
        """
        Performs the reverse denoising process, reconstructing the data from its noised state.

        Parameters:
        - x_T (Tensor): The tensor at the final time step T.
        - noise_net (callable): The noise network function, predicting noise to be subtracted.
        - solver (callable): The solver function for the reverse process, optimizing denoising steps.
        - M (int): The number of steps in the reverse process. Default is 20.

        Returns:
        - Tensor: The stack of denoised tensors at each step.
        """
        # Compute λ values evenly spaced between the max and min, dictating the denoising schedule.
        lambdas = torch.linspace(self.lambda_max, self.lambda_min, M + 1, device=x_T.device)#.view(-1, 1, 1)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        x = x_T
        xs = []  # Stores the denoised states at each step.
        for i in range(1, len(t)//order):
            x = solver(x, t[i - 1], t[i], noise_net, **kwargs) # Apply the reverse denoising solver.
            xs.append(x.clone())
        if return_all:
            return torch.stack(xs)
        else:
            return x

    def adaptive_reverse_denoise(self, x_T, noise_net,
                                 solver1, solver2, solver3,
                                 order1, order2, order3,
                                 M=30, return_all=False, **kwargs):
        """
        Performs the reverse denoising process, reconstructing the data from its noised state.

        Parameters:
        - x_T (Tensor): The tensor at the final time step T.
        - noise_net (callable): The noise network function, predicting noise to be subtracted.
        - solver (callable): The solver function for the reverse process, optimizing denoising steps.
        - M (int): The number of steps in the reverse process. Default is 20.

        Returns:
        - Tensor: The stack of denoised tensors at each step.
        """
        # Compute λ values evenly spaced between the max and min, dictating the denoising schedule.
        lambdas = torch.linspace(self.lambda_max, self.lambda_min, M + 1, device=x_T.device)#.view(-1, 1, 1)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        x = x_T
        xs = []
        xs.append(x.clone()) # Stores the denoised states at each step.

        assert order3 * 3 + order2 * 2 + order1 * 1 == len(t) - 1

        if order3 > 0:
            for i in range(1, order3+1):
                x = solver3(x, t[i - 1], t[i], noise_net, **kwargs) # Apply the reverse denoising solver.
                xs.append(x.clone())
        if order2 > 0:
            for j in range(1, order2+1):
                x = solver2(x, t[j - 1], t[j], noise_net, **kwargs)  # Apply the reverse denoising solver.
                xs.append(x.clone())
        if order1 > 0:
            for k in range(1, order1+1):
                x = solver1(x, t[k - 1], t[k], noise_net, **kwargs)  # Apply the reverse denoising solver.
                xs.append(x.clone())
        if return_all:
            return torch.stack(xs)
        else:
            return x

    def solver1(self, x, t_im1, t_i, noise_net, **kwargs):
        """
        A solver function for the reverse denoising process, computing the next state.

        Parameters:
        - x (Tensor): The current tensor.
        - t_im1 (Tensor): The previous time step.
        - t_i (Tensor): The current time step.
        - noise_net (callable): The noise network function, applied at the previous time step.

        Returns:
        - Tensor: The updated tensor after applying the solver, moving one step closer to denoised data.
        """
        h_i = self.lambda_t(t_i) - self.lambda_t(t_im1)
        # Difference in λ values, dictating the step size.
        # Update the tensor based on α, σ, and the predicted noise, effectively denoising it.
        return self.alpha_t(t_i) / self.alpha_t(t_im1) * x - \
            self.sigma_t(t_i) * torch.expm1(h_i) * noise_net(x=x, t=t_im1[None], **kwargs)[0]

    def solver2(self, x, t_im1, t_i, noise_net, r1=0.5, **kwargs):
        """
        A solver function for the reverse denoising process, computing the next state.

        Parameters:
        - x (Tensor): The current tensor.
        - t_im1 (Tensor): The previous time step.
        - t_i (Tensor): The current time step.
        - noise_net (callable): The noise network function, applied at the previous time step.

        Returns:
        - Tensor: The updated tensor after applying the solver, moving one step closer to denoised data.
        """
        lambda_ti, lambda_tim1 = self.lambda_t(t_i), self.lambda_t(t_im1)
        h_i = lambda_ti - lambda_tim1
        lambda_s1 = lambda_tim1 + r1 * h_i
        s1 = self.t_lambda(lambda_s1)
        alpha_s1, alpha_tim1, alpha_ti = (self.alpha_t(s1),
                                          self.alpha_t(t_im1),
                                          self.alpha_t(t_i))
        sigma_s1, sigma_tim1, sigma_ti = (self.sigma_t(s1),
                                          self.sigma_t(t_im1),
                                          self.sigma_t(t_i))

        phi_11, phi_1 = torch.expm1(r1 * h_i), torch.expm1(h_i)

        model_im1 = noise_net(x=x, t=t_im1[None], **kwargs)[0]

        x_s1 = (alpha_s1 / alpha_tim1 * x
                - (sigma_s1 * phi_11) * model_im1)
        model_s1 = noise_net(x=x_s1, t=s1[None], **kwargs)[0]

        x_ti = (alpha_ti / alpha_tim1 * x
                - (sigma_ti * phi_1) * model_im1
                - (0.5 / r1) * (sigma_ti * phi_1) * (model_s1 - model_im1))
        return x_ti

    def solver3(self, x, t_im1, t_i, noise_net, r1=1./3., r2=2./3., **kwargs):
        """
        A solver function for the reverse denoising process, computing the next state.

        Parameters:
        - x (Tensor): The current tensor.
        - t_im1 (Tensor): The previous time step.
        - t_i (Tensor): The current time step.
        - noise_net (callable): The noise network function, applied at the previous time step.

        Returns:
        - Tensor: The updated tensor after applying the solver, moving one step closer to denoised data.
        """
        lambda_ti, lambda_tim1 = self.lambda_t(t_i), self.lambda_t(t_im1)
        h_i = lambda_ti - lambda_tim1
        lambda_s1, lambda_s2 = lambda_tim1 + r1 * h_i, lambda_tim1 + r2 * h_i
        s1, s2 = self.t_lambda(lambda_s1), self.t_lambda(lambda_s2)
        alpha_s1, alpha_s2, alpha_tim1, alpha_ti = (self.alpha_t(s1),
                                                    self.alpha_t(s2),
                                                    self.alpha_t(t_im1),
                                                    self.alpha_t(t_i))
        sigma_s1, sigma_s2, sigma_tim1, sigma_ti = (self.sigma_t(s1),
                                                    self.sigma_t(s2),
                                                    self.sigma_t(t_im1),
                                                    self.sigma_t(t_i))

        phi_11 = torch.expm1(r1 * h_i)
        phi_12 = torch.expm1(r2 * h_i)
        phi_1 = torch.expm1(h_i)
        phi_22 = torch.expm1(r2 * h_i) / (r2 * h_i) - 1.
        phi_2 = phi_1 / h_i - 1.
        phi_3 = phi_2 / h_i - 0.5

        model_im1 = noise_net(x=x, t=t_im1[None], **kwargs)[0]

        x_s1 = (alpha_s1 / alpha_tim1 * x
                - (sigma_s1 * phi_11) * model_im1)
        model_s1 = noise_net(x=x_s1, t=s1[None], **kwargs)[0]

        x_s2 = (alpha_s2 / alpha_tim1 * x
                - (sigma_s2 * phi_12) * model_im1
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_im1))
        model_s2 = noise_net(x=x_s2, t=s2[None], **kwargs)[0]

        x_ti = (alpha_ti / alpha_tim1 * x
                - (sigma_ti * phi_1) * model_im1
                - (1. / r2) * (sigma_ti * phi_2) * (model_s2 - model_im1))
        return x_ti

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
                                                     original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys/resname_unl.h5')
        return DataLoader(TrajsDataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        # ProteinAnalysis(val_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset_val = TrajectoriesDataset_Efficient(cutoff=self.cutoff,
                                                         scale=self.scale,
                                                         original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys/resname_unl.h5')
        return DataLoader(TrajsDataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        # ProteinAnalysis(test_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset_test = TrajectoriesDataset_Efficient(cutoff=self.cutoff,
                                                          scale=self.scale,
                                                          original_h5_file='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys_test/resname_unl.h5')
        return DataLoader(TrajsDataset_test, batch_size=1, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True)

# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #  Initialize your model, optimizer, scheduler, criterion, etc
        self.model = DynamicsGNN(config['node_dim'], config['edge_dim'], config['vector_dim'])
        self.dpm = DDPM()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)  # Configure scheduler here
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config['decay'])

        self.criterion = nn.MSELoss()
        # self.criterion = nn.KLDivLoss(reduction='batchmean')

        self.save_hyperparameters()

    def training_step(self, batch):
        # pairwise frames
        source_graphs, target_graphs = batch

        # add noise to get morphism transition kernels
        with torch.no_grad():
            unique_graph_indices = torch.unique(source_graphs.batch)
            num_unique_graphs = len(unique_graph_indices)
            lambdas = torch.empty(num_unique_graphs, 1).uniform_(self.dpm.lambda_max, self.dpm.lambda_min)
            t = self.dpm.t_lambda(lambdas)

            t_assigned = torch.zeros_like(source_graphs.batch, dtype=torch.float32)
            for i, graph_idx in enumerate(unique_graph_indices):
                graph_indices = (source_graphs.batch == graph_idx).nonzero(as_tuple=True)[0]
                t_assigned[graph_indices] = t[i]
            t_assigned = t_assigned[:, None]

            # noised_pos, eps = self.dpm.forward_noise(source_graphs.pos, t_assigned)
            noised_pos, eps = self.dpm.forward_noise(source_graphs.pos, t_assigned)
        # print(t_assigned)
        # transformed_graphs, h = self.model(
        #     t=t_assigned,
        #     edge_index=source_graphs.edge_index,
        #     edge_attr=source_graphs.edge_attr,
        #     x=noised_pos,
        #     h=source_graphs.x,
        #     cond=source_graphs.pos
        # )

        pred_eps, h = self.model(
            t=t_assigned,
            edge_index=source_graphs.edge_index,
            edge_attr=source_graphs.edge_attr,
            x=noised_pos,
            h=source_graphs.x,
            cond=source_graphs.pos
        )

        # loss from one frame to another frame
        # loss = self.criterion(transformed_graphs, target_graphs.pos)
        loss =self.criterion(pred_eps, eps)
        # loss = F.kl_div(torch.log(transformed_graphs + 1e-10), target_graphs.pos, reduction='batchmean', log_target=False)
        self.log('train_loss', loss)
        self.log('learning_rate', self.optimizer.param_groups[0]['lr'])

        return loss

    def validation_step(self, batch):
        source_graphs, target_graphs = batch
        unique_graph_indices = torch.unique(source_graphs.batch)
        num_unique_graphs = len(unique_graph_indices)
        lambdas = torch.empty(num_unique_graphs, 1).uniform_(self.dpm.lambda_max, self.dpm.lambda_min)
        t = self.dpm.t_lambda(lambdas)

        t_assigned = torch.zeros_like(source_graphs.batch, dtype=torch.float32)
        for i, graph_idx in enumerate(unique_graph_indices):
            graph_indices = (source_graphs.batch == graph_idx).nonzero(as_tuple=True)[0]
            t_assigned[graph_indices] = t[i]
        t_assigned = t_assigned[:, None]
        # print(t_assigned)
        # noised_pos, eps = self.dpm.forward_noise(source_graphs.pos, t_assigned)
        noised_pos, eps = self.dpm.forward_noise(source_graphs.pos, t_assigned)
        # transformed_graphs, h = self.model(
        #     t=t_assigned,
        #     edge_index=source_graphs.edge_index,
        #     edge_attr=source_graphs.edge_attr,
        #     x=noised_pos,
        #     h=source_graphs.x,
        #     cond=source_graphs.pos
        # )

        pred_eps, h = self.model(
            t=t_assigned,
            edge_index=source_graphs.edge_index,
            edge_attr=source_graphs.edge_attr,
            x=noised_pos,
            h=source_graphs.x,
            cond=source_graphs.pos
        )
        # Compute loss
        with self.ema.average_parameters():
            # loss = self.criterion(transformed_graphs, target_graphs.pos)
            loss = self.criterion(pred_eps, eps)
            # loss = F.kl_div(torch.log(transformed_graphs + 1e-10), target_graphs.pos, reduction='batchmean', log_target=False)
        # Log the validation loss
        self.log('val_loss', loss)
        return loss

    def transform_frame(self, batch, is_noise=False):
        source_graphs = batch[0]
        # Ensure the model is in evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Assign t value to the source frame
            lambdas = torch.empty(1, 1).uniform_(self.dpm.lambda_max, self.dpm.lambda_min)
            t = self.dpm.t_lambda(lambdas)
            t_assigned = torch.full((source_graphs.pos.shape[0], 1), t.item(), device=source_graphs.pos.device)
            noised_pos, eps = self.dpm.forward_noise(source_graphs.pos, t_assigned)
            if is_noise:
                # Forward pass through the model
                transformed_frame, h = self.model(
                    t=t_assigned,
                    edge_index=source_graphs.edge_index,
                    edge_attr=source_graphs.edge_attr,
                    x=noised_pos,
                    h=source_graphs.x,
                    cond=source_graphs.pos
                )
            else:
                # Forward pass through the model
                transformed_frame, h = self.model(
                    t=t_assigned,
                    edge_index=source_graphs.edge_index,
                    edge_attr=source_graphs.edge_attr,
                    x=source_graphs.pos,
                    h=source_graphs.x,
                    cond=source_graphs.pos
                )

        return transformed_frame

    def test_sequence(self, batch, num_frames_to_process, folder, name, ref_pdb):
        for step in range(num_frames_to_process):
            transformed_frame = self.transform_frame(batch)
            print(f"Step {step}, Transformed Frame Pos: {transformed_frame}")
            # Save the generated frame as a PDB file
            denoised_file = os.path.join(
                folder,
                'denoised_{}_from_frame_{}_to_{}.pdb'.format(name, step, step + 1)
            )
            write_combined_pdb(ref_pdb, transformed_frame.detach().cpu().numpy() / 2.0, denoised_file)

            # Here you might want to save or process the transformed_frame
            batch[0].pos = transformed_frame

        return batch[0].pos

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

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

if __name__ == "__main__":
    print(os.getcwd())
    config = parse_toml_file('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/config.toml')
    directory_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys'
    val_path = '/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/small_sys/sys'
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
    os.environ['NCCL_P2P_DISABLE'] = '1'

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
    # model = LitModel(config)
    model = LitModel.load_from_checkpoint('/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_8_1-v33.ckpt', config=config)

    print(model.model.time_embedding.B)
    torch.manual_seed(42)

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
        filename='/home/ziyu/PycharmProjects/pythonProject/small_sys_gnn/output/solver1_gnn_test_beta_8_1',
        monitor='val_loss',
        mode='min'
    )

    # from lightning.pytorch.loggers import CSVLogger
    # logger = CSVLogger("logs", name="my_exp_name")

    # Initialize Trainer with early stopping callback and model checkpoint callback
    trainer = L.Trainer(
        devices=2,
        accelerator="cuda",
        max_epochs=config['num_epochs'],
        callbacks=[early_stop_callback, checkpoint_callback],
        strategy="ddp_find_unused_parameters_true",
        # strategy=DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=14400)),
        # precision="bf16",
        log_every_n_steps=50,
        # logger=TensorBoardLogger(save_dir='/home/ziyu/PycharmProjects/pythonProject/Ex/output', name='gnn')
    )

    # Train the model
    trainer.fit(model=model,
                datamodule=datamodule)












