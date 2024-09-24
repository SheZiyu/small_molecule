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
from model.egnn_lightning import LitModel
from data_loading.data_loader import create_dataloaders


@hydra.main(
    config_path="../config/", config_name="flowmatching_egnn", version_base="1.1"
)
def main(config):
    device = config.inference.device
    # device = "cpu"
    train_loader = create_dataloaders(config.data, config.inference)
    # extract first element:
    data_iterator = iter(train_loader)
    first_batch = next(data_iterator).to(device)
    model = LitModel(config.train).to(device)
    # Load checkpoint
    model.load_checkpoint(config.train.resume_path)
    ###########################################################################################
    number_nodes_batch = first_batch.batch.shape[0]
    t_max = 1.0
    n_time_steps = 10
    ts = torch.linspace(0.0, t_max, n_time_steps).to(device)
    rtol = 1e-7
    atol = 1e-9
    # rtol = 1e-5
    # atol = 1e-7
    method = "euler"
    # method = "fehlberg2"
    trajectory = []
    traj_len = 40000
    bb_dynamics = BlackBoxDynamics(model, config)
    start_time = time.time()
    for ind in tqdm(range(traj_len), desc="Generating Trajectory"):
        trajectory.append(
            subtract_means(first_batch.pos, first_batch.batch).detach().cpu()
        )
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
            first_batch.pos = subtract_means(
                first_batch.pos + solution[-1], first_batch.batch
            )
            # state = state
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    trajectory_np = np.array(trajectory)
    np.save(config.inference.output_traj_path, trajectory_np)


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
