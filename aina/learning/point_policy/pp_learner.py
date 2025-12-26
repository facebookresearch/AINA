import os

import torch
import torch.nn as nn
from termcolor import cprint

from dex_aria.learning.learner import Learner
from dex_aria.learning.networks.gpt import GPT, GPTConfig
from dex_aria.learning.networks.mlp import MLP
from dex_aria.learning.networks.policy_head import DeterministicHead, DiffusionHead
from dex_aria.utils.file_ops import suppress
from dex_aria.utils.model_utils import schedule, weight_init


class PointPolicy(Learner):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self._init_models(cfg)
        self.set_optimizer(cfg)

        self.supervise_object_points = cfg.learner.supervise_object_points

        if cfg.learner.apply_weight_init:
            self._apply_weight_init()

    def _apply_weight_init(self):
        self.encoder.apply(weight_init)
        self.policy.apply(weight_init)
        self.head.apply(weight_init)

    def print_info(self):
        cprint(f"[PointPolicy Learner] Initializing models", "blue")
        policy_params = sum(p.numel() for p in self.policy.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total_params = policy_params + encoder_params + head_params
        cprint(
            f"Policy parameters: {policy_params / 1e6:.4f}M ({policy_params / total_params:.2%})",
            "blue",
        )
        cprint(
            f"Encoder parameters: {encoder_params / 1e6:.4f}M ({encoder_params / total_params:.2%})",
            "blue",
        )
        cprint(
            f"Head parameters: {head_params / 1e6:.4f}M ({head_params / total_params:.2%})",
            "blue",
        )
        cprint(f"Total parameters: {total_params / 1e6:.4f}M", "blue")
        cprint(f"--------------------------------", "blue")

    def _init_models(self, cfg):
        self.encoder = MLP(
            in_channels=3 * cfg.dataset.obs_horizon,
            hidden_channels=[cfg.learner.encoder.repr_dim],
            dropout=cfg.learner.encoder.dropout,
        )  # Point projector

        self.policy = GPT(
            GPTConfig(
                block_size=cfg.dataset.num_object_points + 21,
                input_dim=cfg.learner.encoder.repr_dim,
                output_dim=cfg.learner.policy.output_dim,
                n_layer=cfg.learner.policy.n_layer,
                n_head=cfg.learner.policy.n_head,
                n_embd=cfg.learner.policy.output_dim,
                dropout=cfg.learner.policy.dropout,
                use_position_encoding=(
                    cfg.learner.policy.use_position_encoding
                    if "use_position_encoding" in cfg.learner.policy
                    else True
                ),
                position_encoding_type=(
                    (
                        cfg.learner.policy.position_encoding_type
                    )  # TODO: Delete these after
                    if "position_encoding_type" in cfg.learner.policy
                    else "all"
                ),
            )
        )

        assert (
            cfg.learner.head.type == "deterministic"
        ), "Only deterministic head is supported for now"
        if cfg.learner.head.type == "deterministic":
            if not "predict_distribution" in cfg.learner:
                cfg.learner.predict_distribution = True  # This was the previous default

            self.head = DeterministicHead(
                input_size=cfg.learner.policy.output_dim,
                output_size=3
                * cfg.dataset.pred_horizon,  # This will predict next states of all the points
                hidden_size=cfg.learner.head.hidden_dim,
                num_layers=cfg.learner.head.num_layers,
                action_squash=cfg.learner.head.action_squash,
                loss_coef=cfg.learner.head.loss_coef,
                loss_reduction="sum",
                predict_distribution=cfg.learner.predict_distribution,
            )
        else:
            self.head = DiffusionHead(
                input_size=cfg.net.policy.output_dim,
                output_size=3 * cfg.dataset.pred_horizon,
                obs_horizon=1,  # TODO: Not sure what's up here
                pred_horizon=cfg.dataset.pred_horizon,
                hidden_size=cfg.net.head.hidden_dim,
                num_layers=cfg.net.head.num_layers,
                normalization_scale=cfg.net.head.normalization_scale,
                device=self.device,
            )

    def set_optimizer(self, cfg):
        self.optimizer = torch.optim.AdamW(
            params=list(self.encoder.parameters())
            + list(self.policy.parameters())
            + list(self.head.parameters()),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
        # self.loss_fn = torch.nn.MSELoss(reduction=cfg.learner.loss_fn_reduction)

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.policy.to(device)
        self.head.to(device)

    def train(self):
        self.encoder.train()
        self.policy.train()
        self.head.train()

    def eval(self):
        self.encoder.eval()
        self.policy.eval()
        self.head.eval()

    def save(self, checkpoint_dir, model_type="best"):
        torch.save(
            self.head.state_dict(),
            os.path.join(checkpoint_dir, f"head_{model_type}.pt"),
            _use_new_zipfile_serialization=False,
        )

        torch.save(
            self.encoder.state_dict(),
            os.path.join(checkpoint_dir, f"encoder_{model_type}.pt"),
            _use_new_zipfile_serialization=False,
        )

        torch.save(
            self.policy.state_dict(),
            os.path.join(checkpoint_dir, f"policy_{model_type}.pt"),
            _use_new_zipfile_serialization=False,
        )

    def load(self, checkpoint_dir, training_cfg, device=None, model_type="best"):

        # Get the weigths
        with suppress(stdout=True):
            encoder_state_dict = torch.load(
                os.path.join(checkpoint_dir, f"encoder_{model_type}.pt"),
                map_location=device,
            )
            policy_state_dict = torch.load(
                os.path.join(checkpoint_dir, f"policy_{model_type}.pt"),
                map_location=device,
            )
            head_state_dict = torch.load(
                os.path.join(checkpoint_dir, f"head_{model_type}.pt"),
                map_location=device,
            )

            self.encoder.load_state_dict(encoder_state_dict)
            self.policy.load_state_dict(policy_state_dict)
            self.head.load_state_dict(head_state_dict)

    def _calculate_logging_loss(self, output_data, predicted_output_data):
        # output_data: (B, num_object_points, pred_horizon * 3)
        # predicted_output_data: (B, num_object_points, pred_horizon * 3)
        B = output_data.shape[0]
        action_points = int(self.cfg.task.action_dim / 3)
        gt_action_points = output_data[:, -action_points:, :]
        pred_action_points = predicted_output_data[:, -action_points:, :]

        # breakpoint()

        # flatten the whole batch into N, 3
        flattened_prediction_error = torch.abs(
            pred_action_points.reshape(-1, 3) - gt_action_points.reshape(-1, 3)
        )
        # cprint(
        #     f"Flattened prediction error: {flattened_prediction_error.shape} | Should be ({output_data.shape[0] * 5 * self.cfg.dataset.pred_horizon}, 3)",
        #     "red",
        # )
        # TODO: Denormalize the loss for it to be metric
        return flattened_prediction_error

    def train_epoch(self, train_loader, epoch, logger=None, **kwargs):
        self.train()

        # Save the train loss
        train_loss = 0.0
        train_losses = []

        # Training loop
        stddev = schedule(self.cfg.learner.head.stddev_schedule, epoch)
        for batch in train_loader:
            # input_data: (B, obs_horizon, N+21, 3) 3: for x, y, z, N is for the number of object points, 21 is for the hand keypoints
            # output data: (B, pred_horizon, 21, 3)
            input_data, output_data = [x.to(self.device) for x in batch]
            batch_size = input_data.shape[0]
            self.optimizer.zero_grad()

            # Change the shape of the output data to (B, N+21, pred_horizon*3)
            num_total_points = input_data.shape[2]
            pred_horizon = output_data.shape[1]
            output_data = output_data.permute(0, 2, 1, 3).reshape(
                batch_size, num_total_points, pred_horizon * 3
            )

            # Forward pass
            action_pred_dist = self.forward(
                input_data, stddev=stddev, target=output_data
            )

            # Calculate the loss
            if not self.supervise_object_points:
                mask = torch.zeros_like(output_data)
                mask[:, -int(self.cfg.task.action_dim / 3) :, :] = 1
            else:
                mask = torch.ones_like(output_data)
            loss = self.head.loss_fn(action_pred_dist, output_data, mask=mask)

            logging_loss = self._calculate_logging_loss(output_data, action_pred_dist)
            train_losses.append(logging_loss)

            train_loss += loss.item()

            # Backprop
            loss.backward()
            self.optimizer.step()

        # Calculate the mean and std of the logging loss
        # breakpoint()
        train_losses = torch.concat(train_losses, dim=0)
        train_losses_mean = train_losses.mean(dim=0)
        train_losses_std = train_losses.std(dim=0)

        # Logging
        if logger is not None:
            logger.log(
                {
                    "stddev": stddev,
                    "train loss": train_loss / len(train_loader),
                    "epoch": epoch,
                    "mean train loss/all": train_losses_mean.mean(),
                    "mean train loss/0": train_losses_mean[0],
                    "mean train loss/1": train_losses_mean[1],
                    "mean train loss/2": train_losses_mean[2],
                    "std train loss/all": train_losses_std.mean(),
                    "std train loss/0": train_losses_std[0],
                    "std train loss/1": train_losses_std[1],
                    "std train loss/2": train_losses_std[2],
                }
            )

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader, epoch, logger=None, **kwargs):
        self.eval()

        # Save the train loss
        test_loss = 0.0
        test_losses = []

        # Training loop
        stddev = schedule(self.cfg.learner.head.stddev_schedule, epoch)
        for batch in test_loader:
            # input_data: (B, obs_horizon, N+21, 3) 3: for x, y, z, N is for the number of object points, 21 is for the hand keypoints
            # output data: (B, pred_horizon, N+21, 3)
            input_data, output_data = [x.to(self.device) for x in batch]
            batch_size = input_data.shape[0]
            # Change the shape of the output data to (B, N+21, pred_horizon*3)
            num_total_points = input_data.shape[2]
            pred_horizon = output_data.shape[1]
            output_data = output_data.permute(0, 2, 1, 3).reshape(
                batch_size, num_total_points, pred_horizon * 3
            )

            # Forward pass
            with torch.no_grad():
                predicted_action = self.forward(
                    input_data, stddev=stddev, target=output_data
                )

            # Calculate the loss
            if not self.supervise_object_points:
                mask = torch.zeros_like(output_data)
                mask[:, -int(self.cfg.task.action_dim / 3) :, :] = 1
            else:
                mask = torch.ones_like(output_data)
            loss = self.head.loss_fn(predicted_action, output_data, mask=mask)

            logging_loss = self._calculate_logging_loss(output_data, predicted_action)
            test_losses.append(logging_loss)

            test_loss += loss.item()

        # Calculate the mean and std of the logging loss
        test_losses = torch.concat(test_losses, dim=0)
        test_losses_mean = test_losses.mean(dim=0)
        test_losses_std = test_losses.std(dim=0)

        if logger is not None:
            logger.log(
                {
                    "test loss": test_loss / len(test_loader),
                    "mean test loss/all": test_losses_mean.mean(),
                    "mean test loss/0": test_losses_mean[0],
                    "mean test loss/1": test_losses_mean[1],
                    "mean test loss/2": test_losses_mean[2],
                    "std test loss/all": test_losses_std.mean(),
                    "std test loss/0": test_losses_std[0],
                    "std test loss/1": test_losses_std[1],
                    "std test loss/2": test_losses_std[2],
                }
            )

        return test_loss / len(test_loader)

    def forward(self, input_data, stddev=None, target=None):

        # Project each point to a "token"
        # input_data: B, obs_horizon, num_total_points, 3
        batch_size, obs_horizon, num_total_points, _ = input_data.shape
        input_rearranged = input_data.permute(0, 2, 1, 3).reshape(
            batch_size, num_total_points, obs_horizon * 3
        )
        cprint(f"input_rearranged: {input_rearranged.shape}", "magenta")
        point_repr_tokens = self.encoder(input_rearranged)
        cprint(f"point_repr_tokens: {point_repr_tokens.shape}", "magenta")
        features = self.policy(point_repr_tokens)
        cprint(f"features: {features.shape}", "magenta")

        # Get the head output and calculate the loss
        pred_action = self.head(
            features,
            stddev,
            **{
                "action_seq": target if target is not None else None,
            },
        )

        cprint(
            f"pred_action: {pred_action.shape} - should be (B, 505, 3*pred_horizon)",
            "magenta",
        )

        return pred_action

    def predict_action(self, input_data):
        self.eval()
        with torch.no_grad():
            pred_action = self.forward(
                input_data, stddev=None
            )  # When predicting actions we predict the mean either way

        point_num = int(self.cfg.task.action_dim / 3)
        if self.cfg.learner.predict_distribution:
            pred_action = pred_action.mean

        future_hand_points = pred_action[:, -point_num:, :].view(
            -1, point_num, self.cfg.dataset.pred_horizon, 3
        )
        future_hand_points = future_hand_points.permute(0, 2, 1, 3)

        return future_hand_points


import hydra
from omegaconf import DictConfig

from dex_aria.dataset.dataloaders import get_dataloaders
from dex_aria.utils.vector_ops import calculate_stats


@hydra.main(config_path="../../../cfgs", config_name="train_point_policy.yaml")
def main(cfg: DictConfig):
    learner = PointPolicy(cfg)
    cfg.device = "cpu"

    learner.to(cfg.device)
    learner.train()

    # Initialize the dataloader
    object_stats = calculate_stats(
        cfg.task.all_data_directories, mean_std_norm=cfg.dataset.mean_std_norm
    )
    print(f"OBJECT STATS: {object_stats}")

    cfg.task.object_stats = object_stats
    train_loader, test_loader, train_dset, test_dset = get_dataloaders(cfg)
    # cprint(f"LENGTH OF TRAIN LOADER: {len(train_loader)}", "blue")
    # cprint(f"LENGTH OF TEST LOADER: {len(test_loader)}", "blue")
    # cprint(f"LENGTH OF TRAIN DATASET: {len(train_dset)}", "blue")
    # cprint(f"LENGTH OF TEST DATASET: {len(test_dset)}", "blue")

    # train_loss = learner.train_epoch(train_loader, 0)
    # test_loss = learner.test_epoch(test_loader, 0)
    # cprint(f"TRAIN LOSS: {train_loss}", "red")
    # cprint(f"TEST LOSS: {test_loss}", "green")

    batch = next(iter(train_loader))
    input_data, output_data = [x.to(cfg.device) for x in batch]
    predicted_action = learner.predict_action(input_data)
    cprint(f"PREDICTED ACTION: {predicted_action.shape}", "magenta")


if __name__ == "__main__":
    main()
