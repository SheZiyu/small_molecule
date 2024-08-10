import torch
from utils.auxiliary import subtract_means


class BlackBoxDynamics(torch.nn.Module):
    """Black box dynamics that allows to use any dynamics function."""

    def __init__(self, dynamics_function, config):
        super().__init__()
        self._dynamics_function = dynamics_function
        self.config = config

    def forward(self, t, *xs, batch):
        """predict the vector field at time t and position x

        Args:
            t: time point
            xs: location
            batch : batch of molecule graphs

        Returns:
            predicted_vectors: the vector field at time t and location xs
        """
        with torch.no_grad():
            device = batch.pos.device
            time = t * torch.ones(batch.batch.shape[0], 1).to(device)
            x_reshaped = xs[0].reshape((-1, 3))
            perturbed_nodes = batch.pos + x_reshaped
            one_hot_atom = torch.nn.functional.one_hot(
                batch.atom_type, num_classes=self.config.train.one_hot_atom_dim
            )
            perturbed_nodes_updated = self._dynamics_function.model(
                t=time,
                h=one_hot_atom,
                original_nodes=batch.pos,
                perturbed_nodes=perturbed_nodes,
                edge_index=batch.edge_index,
                node2graph=batch.batch,
                edge_type=batch.edge_type,
            )
            predicted_vectors = subtract_means(
                perturbed_nodes_updated - perturbed_nodes, batch.batch
            )
            return predicted_vectors
