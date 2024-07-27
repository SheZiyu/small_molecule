"""Training the model
env: ziyu4
"""

from torch_geometric.loader import DataLoader
from utils.auxiliary import subtract_means
import resource
import torch
import hydra
import wandb
from omegaconf import OmegaConf
from lightning import Trainer, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from model.egnn_lightning import LitModel, LitData
import os
from utils.auxiliary import set_seed
from lightning.pytorch.strategies import DDPStrategy


class InitialCheckpoint(Callback):
    def __init__(self, repository_location):
        super().__init__()
        self.repository_location = repository_location

    def on_train_start(self, trainer, pl_module):
        file_name = os.path.join(
            self.repository_location, "outputs/initials/initial_checkpoint.ckpt"
        )
        # Create a checkpoint file name based on your criteria
        # Save the initial model checkpoint
        trainer.save_checkpoint(file_name)


@hydra.main(config_path="../config/", config_name="diffusion_egnn", version_base="1.1")
def main(config):
    # Initialize W&B Run
    wandb.init(
        project="molecule_trajectory",
        config=OmegaConf.to_container(config),
        reinit=True,
    )

    datamodule = LitData(config)
    # for batch in datamodule.train_dataloader():
    #    diff1=batch.pos[batch.batch==1]-batch.pos[batch.batch==0]
    #    diff2=batch.dd[batch.batch==0]
    #    print(batch)

    # data_iterator = iter(datamodule.train_dataloader())
    # first_batch = next(data_iterator)
    # first_batch.pos[first_batch.batch==0]

    model = LitModel(config)
    # If resuming, load checkpoint
    if config.resume_path is not None and config.load_checkpoint is True:
        model.load_checkpoint(config.resume_path)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath="outputs",
            monitor="valid_loss" if config.do_validation else None,
            mode="min",
            save_last=True,
        ),
    ]

    # Create PyTorch Lightning trainer
    trainer = Trainer(
        logger=WandbLogger(project="hs_position"),
        max_epochs=config.num_epochs,
        accelerator="gpu",
        devices=config.cuda_ids,
        precision=16,
        accumulate_grad_batches=config.acc_grad_batches,
        callbacks=callbacks,
        # strategy="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    # Train neural network
    if config.do_validation:
        trainer.fit(model=model, datamodule=datamodule)
    else:
        trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader())


if __name__ == "__main__":
    set_seed()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    main()
    exit()
