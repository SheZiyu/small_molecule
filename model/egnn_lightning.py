from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from prepocessing.preprocessing import parse_toml_file
from prepocessing.data_extend import TrajectoriesDataset_Efficient
from model.egnn import DynamicsEGNN
from utils.auxiliary import get_optimizer, subtract_means


# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigma = self.config.sigma

        #  Initialize your model, optimizer, scheduler, criterion, etc
        self.model = DynamicsEGNN(self.config["node_dim"], 4)
        # self.dpm = DDPM()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=300, gamma=0.1
        )  # Configure scheduler here
        # self.ema = ExponentialMovingAverage(
        #    self.model.parameters(), decay=config["decay"]
        # )

        # self.criterion = nn.MSELoss()
        # self.save_hyperparameters()
        self.train_outputs = []
        self.valid_outputs = []

    def forward(self, batch, outputs=None):
        with torch.no_grad():
            device = batch.pos.device
            number_nodes_batch = batch.batch.shape[0]
            position_noise = 0.1 * torch.randn_like(batch.pos)
            current_positions_noised = subtract_means(
                batch.pos + position_noise, batch.batch
            )
            p0_samples = torch.randn(number_nodes_batch, 3).to(device)
            p0_samples_mean_free = subtract_means(p0_samples, batch.batch)
            p1_samples_mean_free = subtract_means(
                batch.increments - position_noise, batch.batch
            )
            time = torch.rand(len(batch), 1).to(device)
            num_nodes_per_graph = torch.bincount(batch.batch)
            time_repeated = time.repeat_interleave(num_nodes_per_graph, dim=0)
            mu_t = (
                p0_samples_mean_free * (1 - time_repeated)
                + p1_samples_mean_free * time_repeated
            )
            sigma_t = self.sigma
            noise = torch.randn(number_nodes_batch, 3).to(device)
            noise_mean_free = subtract_means(noise, batch.batch)
            dx_mean_free = mu_t + sigma_t * noise_mean_free
            perturbed_nodes = subtract_means(
                current_positions_noised + dx_mean_free, batch.batch
            )
            target_vectors = p1_samples_mean_free - p0_samples_mean_free
        one_hot_atom = torch.nn.functional.one_hot(
            batch.atom_type, num_classes=self.config.one_hot_atom_dim
        )
        perturbed_nodes_updated = self.model(
            t=time_repeated,
            h=one_hot_atom,
            original_nodes=current_positions_noised,
            perturbed_nodes=perturbed_nodes,
            edge_index=batch.edge_index,
            node2graph=batch.batch,
            edge_type=batch.edge_type,
        )
        predicted_vectors = subtract_means(
            perturbed_nodes_updated - perturbed_nodes, batch.batch
        )
        loss = torch.mean((target_vectors - predicted_vectors) ** 2)
        if loss is not None:
            outputs.append({"loss": loss})
        else:
            print("Loss is None, skipping append.")
        print("loss_total: ", loss)
        return loss

    @staticmethod
    def epoch_end_metrics(outputs, label: str, stride: int = 1):
        """Compute all metrics at the end of an epoch"""
        losses = [
            output["loss"]
            for output in outputs[::stride]
            if output["loss"] is not None and not torch.isnan(output["loss"])
        ]
        metrics = {
            f"{label}_loss": (
                torch.tensor(losses).mean().item() if losses else float("inf")
            ),
        }
        return metrics

    def training_step(self, batch):
        loss = self.forward(batch, self.train_outputs)
        if loss is None:
            print("Skipping step due to None loss")
            return None  # Or return 0 if necessary
        return loss

    def on_train_epoch_end(self):
        """Training epoch end (logging)"""
        metrics = self.epoch_end_metrics(self.train_outputs, "train", stride=1)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_outputs = []

    def validation_step(self, batch):
        loss = self.forward(batch, self.valid_outputs)
        if loss is None:
            print("Skipping step due to None loss")
            return None  # Or return 0 if necessary
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end (logging)"""
        metrics = self.epoch_end_metrics(self.valid_outputs, "valid", stride=1)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.valid_outputs = []

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def load_checkpoint(self, checkpoint_path):
        assert os.path.exists(
            checkpoint_path
        ), f"resume_path ({checkpoint_path}) does not exist"
        self.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        )
        print(f"Loaded checkpoint from {checkpoint_path}")

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # self.ema.update(self.model.parameters())

    def on_after_backward(self):
        detect_unused = False
        if detect_unused:
            print("After backward pass:")
            unused_parameters = []
            for name, param in self.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} is UNUSED.")
                    unused_parameters.append(name)
                else:
                    print(f"Parameter {name} is used.")
            if unused_parameters:
                print(f"Unused parameters: {unused_parameters}")
            else:
                print("All parameters are used.")
