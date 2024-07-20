from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch_ema import ExponentialMovingAverage

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from prepocessing.preprocessing import parse_toml_file
from prepocessing.data_extend import TrajectoriesDataset_Efficient
from model.egnn_flow_matching import DynamicsEGNN


class DDPM(nn.Module):
    """
    Implementation of the Denoising Diffusion Probabilistic Model (DDPM).

    Parameters:
    - schedule (str): The scheduling method for the noise levels, e.g., 'linear'.
    - t_min (float): The minimum time step for the diffusion process, used to control the start of the noise schedule.
    """

    def __init__(self, schedule="linear", t_min=1e-3, beta_0=0.1, beta_1=20):
        super().__init__()
        self.schedule = schedule
        self.t_min = t_min

        # Define the noise schedule parameters based on the chosen schedule.
        if schedule == "linear":
            # Parameters defining the linear schedule for noise addition.
            self.beta_0, self.beta_1 = beta_0, beta_1
            self.t_max = (
                1  # Maximum time step, representing the end of the diffusion process.
            )

            # Alpha (α): Variance retention coefficient at each time step.
            # Controls the proportion of the original signal retained during diffusion.
            self.alpha_t = lambda t: torch.exp(
                -(beta_0 * t / 2 + t.pow(2) / 4 * (beta_1 - beta_0))
            )

            # Lambda (λ) and Time (t) Transformation Functions:
            # Used for reparameterizing the diffusion schedule for efficiency or performance.
            self.t_lambda = (
                lambda lmd: 2
                * torch.log(torch.exp(-2 * lmd) + 1)
                / (
                    torch.sqrt(
                        beta_0**2
                        + (beta_1 - beta_0) * 2 * torch.log(torch.exp(-2 * lmd) + 1)
                    )
                    + beta_0
                )
            )

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
        x_t = (
            self.alpha_t(t) * x + self.sigma_t(t) * eps
        )  # Applies the diffusion process.
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
        lambdas = torch.linspace(
            self.lambda_max, self.lambda_min, M + 1, device=x_T.device
        )  # .view(-1, 1, 1)
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
        lambdas = torch.linspace(
            self.lambda_max, self.lambda_min, M + 1, device=x_T.device
        )  # .view(-1, 1, 1)
        # print(lambdas)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        # print(t)
        x = x_T
        xs = []  # Stores the denoised states at each step.
        for i in range(len(t)):
            x = noise_net(x=x, t=t[i][None], **kwargs)[
                0
            ]  # Apply the reverse denoising solver.
            # print(x)
            xs.append(x.clone())
        if return_all:
            return torch.stack(xs)
        else:
            return x

    def reverse_denoise(
        self, x_T, noise_net, solver, order=1, M=30, return_all=False, **kwargs
    ):
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
        lambdas = torch.linspace(
            self.lambda_max, self.lambda_min, M + 1, device=x_T.device
        )  # .view(-1, 1, 1)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        x = x_T
        xs = []  # Stores the denoised states at each step.
        for i in range(1, len(t) // order):
            x = solver(
                x, t[i - 1], t[i], noise_net, **kwargs
            )  # Apply the reverse denoising solver.
            xs.append(x.clone())
        if return_all:
            return torch.stack(xs)
        else:
            return x

    def adaptive_reverse_denoise(
        self,
        x_T,
        noise_net,
        solver1,
        solver2,
        solver3,
        order1,
        order2,
        order3,
        M=30,
        return_all=False,
        **kwargs
    ):
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
        lambdas = torch.linspace(
            self.lambda_max, self.lambda_min, M + 1, device=x_T.device
        )  # .view(-1, 1, 1)
        t = self.t_lambda(lambdas)  # Convert λ values back to time steps for denoising.
        x = x_T
        xs = []
        xs.append(x.clone())  # Stores the denoised states at each step.

        assert order3 * 3 + order2 * 2 + order1 * 1 == len(t) - 1

        if order3 > 0:
            for i in range(1, order3 + 1):
                x = solver3(
                    x, t[i - 1], t[i], noise_net, **kwargs
                )  # Apply the reverse denoising solver.
                xs.append(x.clone())
        if order2 > 0:
            for j in range(1, order2 + 1):
                x = solver2(
                    x, t[j - 1], t[j], noise_net, **kwargs
                )  # Apply the reverse denoising solver.
                xs.append(x.clone())
        if order1 > 0:
            for k in range(1, order1 + 1):
                x = solver1(
                    x, t[k - 1], t[k], noise_net, **kwargs
                )  # Apply the reverse denoising solver.
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
        return (
            self.alpha_t(t_i) / self.alpha_t(t_im1) * x
            - self.sigma_t(t_i)
            * torch.expm1(h_i)
            * noise_net(x=x, t=t_im1[None, None].repeat([x.shape[0], 1]), **kwargs)[0]
        )

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
        alpha_s1, alpha_tim1, alpha_ti = (
            self.alpha_t(s1),
            self.alpha_t(t_im1),
            self.alpha_t(t_i),
        )
        sigma_s1, sigma_tim1, sigma_ti = (
            self.sigma_t(s1),
            self.sigma_t(t_im1),
            self.sigma_t(t_i),
        )

        phi_11, phi_1 = torch.expm1(r1 * h_i), torch.expm1(h_i)

        model_im1 = noise_net(x=x, t=t_im1[None], **kwargs)[0]

        x_s1 = alpha_s1 / alpha_tim1 * x - (sigma_s1 * phi_11) * model_im1
        model_s1 = noise_net(x=x_s1, t=s1[None], **kwargs)[0]

        x_ti = (
            alpha_ti / alpha_tim1 * x
            - (sigma_ti * phi_1) * model_im1
            - (0.5 / r1) * (sigma_ti * phi_1) * (model_s1 - model_im1)
        )
        return x_ti

    def solver3(self, x, t_im1, t_i, noise_net, r1=1.0 / 3.0, r2=2.0 / 3.0, **kwargs):
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
        alpha_s1, alpha_s2, alpha_tim1, alpha_ti = (
            self.alpha_t(s1),
            self.alpha_t(s2),
            self.alpha_t(t_im1),
            self.alpha_t(t_i),
        )
        sigma_s1, sigma_s2, sigma_tim1, sigma_ti = (
            self.sigma_t(s1),
            self.sigma_t(s2),
            self.sigma_t(t_im1),
            self.sigma_t(t_i),
        )

        phi_11 = torch.expm1(r1 * h_i)
        phi_12 = torch.expm1(r2 * h_i)
        phi_1 = torch.expm1(h_i)
        phi_22 = torch.expm1(r2 * h_i) / (r2 * h_i) - 1.0
        phi_2 = phi_1 / h_i - 1.0
        phi_3 = phi_2 / h_i - 0.5

        model_im1 = noise_net(x=x, t=t_im1[None], **kwargs)[0]

        x_s1 = alpha_s1 / alpha_tim1 * x - (sigma_s1 * phi_11) * model_im1
        model_s1 = noise_net(x=x_s1, t=s1[None], **kwargs)[0]

        x_s2 = (
            alpha_s2 / alpha_tim1 * x
            - (sigma_s2 * phi_12) * model_im1
            - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_im1)
        )
        model_s2 = noise_net(x=x_s2, t=s2[None], **kwargs)[0]

        x_ti = (
            alpha_ti / alpha_tim1 * x
            - (sigma_ti * phi_1) * model_im1
            - (1.0 / r2) * (sigma_ti * phi_2) * (model_s2 - model_im1)
        )
        return x_ti


class LitData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cutoff = config["cutoff"]
        self.scale = config["scale"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

    def train_dataloader(self):
        # ProteinAnalysis(directory_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset = TrajectoriesDataset_Efficient(
            cutoff=self.cutoff,
            scale=self.scale,
            original_h5_file="/storage/florian/ziyu_project/resname_unl.h5",
        )
        return DataLoader(
            TrajsDataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # ProteinAnalysis(val_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset_val = TrajectoriesDataset_Efficient(
            cutoff=self.cutoff,
            scale=self.scale,
            original_h5_file="/storage/florian/ziyu_project/resname_unl.h5",
        )
        return DataLoader(
            TrajsDataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        # ProteinAnalysis(test_path, num_frames_to_process).preprocess_coordinate_onehot()
        TrajsDataset_test = TrajectoriesDataset_Efficient(
            cutoff=self.cutoff,
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


# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #  Initialize your model, optimizer, scheduler, criterion, etc
        self.model = DynamicsEGNN(config["node_dim"], 4)
        self.dpm = DDPM()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.1
        )  # Configure scheduler here
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=config["decay"]
        )

        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, batch):
        with torch.no_grad():
            unique_graph_indices = torch.unique(batch.batch)
            num_unique_graphs = len(unique_graph_indices)
            lambdas = torch.empty(num_unique_graphs, 1).uniform_(
                self.dpm.lambda_max, self.dpm.lambda_min
            )
            t = self.dpm.t_lambda(lambdas)
            t_assigned = t.to(self.device)[batch.batch]
            noised_dd, eps = self.dpm.forward_noise(batch.dd, t_assigned)
        pred_perturbed_nodes, h = self.model(
            t=t_assigned,
            h=batch.x,
            original_nodes=batch.pos,
            perturbed_nodes=batch.pos + noised_dd,
            edge_index=batch.edge_index,
            node2graph=batch.batch,
            edge_type=batch.edge_type,
        )
        loss = self.criterion(pred_perturbed_nodes - batch.pos, eps)
        return loss

    def training_step(self, batch):
        loss = self.forward(batch)
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizer.param_groups[0]["lr"])
        return loss

    def validation_step(self, batch):
        with self.ema.average_parameters():
            loss = self.forward(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())


if __name__ == "__main__":
    print(os.getcwd())
    config = parse_toml_file("config_egnn.toml")
    directory_path = "data/sys"
    val_path = "data/sys"
    cutoff = config["cutoff"]
    scale = config["scale"]
    node_dim = config["node_dim"]
    edge_dim = config["edge_dim"]
    vector_dim = config["vector_dim"]
    num_splits = config["num_splits"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    num_frames_to_process = config["num_frames_to_process"]
    os.environ["NCCL_P2P_DISABLE"] = "1"

    datamodule = LitData(config)
    model = LitModel(config)
    # model = LitModel.load_from_checkpoint('/home/ziyu/repos/small_molecule/output/solver1_egnn_test_beta_20_1.ckpt', config=config)

    print(model.model.time_embedding.B)
    torch.manual_seed(42)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=config["patience"], mode="min", verbose=True
    )

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename="/home/ziyu/repos/small_molecule/output/solver1_egnn_test_beta_20_1",
        monitor="val_loss",
        mode="min",
    )

    # Initialize Trainer with early stopping callback and model checkpoint callback
    trainer = L.Trainer(
        devices=[7],
        precision=16,
        accelerator="cuda",
        max_epochs=config["num_epochs"],
        callbacks=[early_stop_callback, checkpoint_callback],
        strategy="auto",
        log_every_n_steps=50,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)
