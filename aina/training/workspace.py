# Copyright (c) Meta Platforms, Inc. and affiliates.

import glob
import os

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from tqdm import tqdm

from aina.dataset.dataloaders import get_dataloaders
from aina.learning.initialize_learner import init_learner
from aina.utils.logger import Logger
from aina.utils.vector_ops import calculate_stats


class TrainingWorkspace:
    def __init__(self, cfg: DictConfig) -> None:
        cprint(f"Workspace config: {OmegaConf.to_yaml(cfg)}", "magenta")

        try:
            self.hydra_dir = (
                f"{HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}"
            )
        except:
            cprint(f"Hydra sweep not found, using run directory", "red")
            self.hydra_dir = HydraConfig.get().run.dir

        self.checkpoint_dir = os.path.join(self.hydra_dir, "models")

        # Create the checkpoint directory - it will be inside the hydra directory
        os.makedirs(
            self.checkpoint_dir, exist_ok=True
        )  # Doesn't give an error if dir exists when exist_ok is set to True
        os.makedirs(self.hydra_dir, exist_ok=True)

        # Set device and config
        self.cfg = cfg

    def train(self) -> None:
        device = torch.device(self.cfg.device)

        # Calculate the stats of the dataset
        if isinstance(self.cfg.task.all_data_directories, str):
            self.cfg.task.all_data_directories = glob.glob(
                self.cfg.task.all_data_directories
            )
        print(f"ALL DATA DIRECTORIES: {self.cfg.task.all_data_directories}")

        # Calculate the stats of the dataset
        object_stats = calculate_stats(
            self.cfg.task.all_data_directories,
            mean_std_norm=self.cfg.dataset.mean_std_norm,
        )

        cprint(f"OBJECT STATS: {object_stats}", "magenta")
        self.cfg.task.object_stats = object_stats

        # It looks at the datatype type and returns the train and test loader accordingly
        train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
            self.cfg
        )
        cprint(f"LENGTH OF TRAIN LOADER: {len(train_loader)}", "magenta")
        cprint(f"LENGTH OF TEST LOADER: {len(test_loader)}", "magenta")
        cprint(f"LENGTH OF TRAIN DATASET: {len(train_dataset)}", "magenta")
        cprint(f"LENGTH OF TEST DATASET: {len(test_dataset)}", "magenta")

        # Save the config to the hydra directory
        with open(f"{self.hydra_dir}/.hydra/config.yaml", "w") as f:
            OmegaConf.save(
                self.cfg, f
            )  # Save the input / output dim to the config so that you can load it directly
        learner = init_learner(cfg=self.cfg, device=device)

        best_loss = torch.inf
        pbar = tqdm(total=self.cfg.train_epochs)

        # Initialize logger (wandb)
        if self.cfg.log:
            job_id = self.hydra_dir.split("/")[-1]
            wandb_exp_name = f"{self.cfg.experiment}-{job_id}"
            cprint(f"wandb_exp_name: {wandb_exp_name}", "magenta")
            self.logger = Logger(self.cfg, wandb_exp_name, out_dir=self.hydra_dir)
        else:
            self.logger = None

        # Start the training
        for epoch in range(self.cfg.train_epochs):

            # Train the models for one epoch
            train_loss = learner.train_epoch(train_loader, epoch, logger=self.logger)

            pbar.set_description(
                f"Epoch {epoch}, Train loss: {train_loss:.5f}, Best loss: {best_loss:.5f}"
            )
            pbar.update(1)

            # Testing and saving the model
            if epoch % 10 == 0:
                # if epoch % self.cfg.save_frequency == 0: # NOTE: Uncomment if you want to save every 10 epoch
                #     learner.save(self.checkpoint_dir, model_type=epoch)

                # Test for one epoch
                test_loss = learner.test_epoch(test_loader, epoch, logger=self.logger)

                # Save if it's the best loss
                if test_loss < best_loss:
                    best_loss = test_loss
                    learner.save(self.checkpoint_dir, model_type="best")

                # Logging
                pbar.set_description(f"Epoch {epoch}, Test loss: {test_loss:.5f}")
                if self.cfg.log:
                    self.logger.log({"best loss": best_loss})

            learner.save(self.checkpoint_dir, model_type="last")

        pbar.close()
        wandb.finish()
