"""generate a trajectory of molecules using the trained flow matching model
"""

import resource
from utils.auxiliary import set_seed
from tqdm import tqdm
import time
from functools import partial
import torch
import yaml
from torch_geometric.loader import DataLoader
from utils.auxiliary import subtract_means
from torchdiffeq import odeint
import numpy as np
from model.ode import BlackBoxDynamics
from utils.auxiliary import subtract_means
import hydra
from model.egnn_lightning import LitModel, LitData


@hydra.main(config_path="../config/", config_name="diffusion_egnn", version_base="1.1")
def main(config):
    datamodule = LitData(config)
    train_dataloader = datamodule.train_dataloader()
    # extract first element:
    data_iterator = iter(train_dataloader)
    first_batch = next(data_iterator)
    device = first_batch.x.device
    model = LitModel(config)
    # Load checkpoint
    model.load_checkpoint(config.resume_path)
    ###########################################################################################
    number_nodes_batch = first_batch.batch.shape[0]
    t_max = 1.0
    t_max = 1.0
    n_time_steps = 20
    ts = torch.linspace(0.0, t_max, n_time_steps).to(device)
    rtol = 1e-7
    atol = 1e-9
    method = "euler"
    trajectory = []
    traj_len = 200
    bb_dynamics = BlackBoxDynamics(model, config)
    for ind in tqdm(range(traj_len), desc="Generating Trajectory"):
        trajectory.append(first_batch.pos)
        bb_dynamics.forward = partial(bb_dynamics.forward, batch=first_batch)
        noise = torch.randn(number_nodes_batch, 3).to(device)
        noise_mean_free = subtract_means(noise, first_batch.batch)
        state = noise_mean_free
        with torch.no_grad():
            solution = odeint(
                bb_dynamics,
                state,
                ts,
                rtol=rtol,
                atol=atol,
                method=method,
            )
            first_batch.pos = first_batch.pos + solution[-1]
            # state = state
    trajectory_np = np.array(trajectory)
    np.save("results/trajectory.npy", trajectory_np)
    print("here")


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
