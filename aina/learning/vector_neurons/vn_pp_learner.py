# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import torch
import torch.nn as nn
from termcolor import cprint

from aina.learning.learner import Learner
from aina.learning.networks.gpt import GPT, GPTConfig
from aina.learning.networks.policy_head import DeterministicHead
from aina.learning.point_policy.pp_learner import PointPolicy
from aina.learning.vector_neurons.vn_mlp import VNMLP


class VNPPLearner(PointPolicy):
    def __init__(self, cfg, device="cuda"):
        # super().__init__()
        self.cfg = cfg
        self.device = device

        self._init_models(cfg)
        self.set_optimizer(cfg)

        self.supervise_object_points = cfg.learner.supervise_object_points

        if cfg.learner.apply_weight_init:
            self._apply_weight_init()

        self.print_info()

    def print_info(self):
        cprint(f"[VNPPLearner] Initializing models", "blue")
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
        self.encoder = VNMLP(
            in_channels=self.cfg.dataset.obs_horizon,
            hidden_channels=[
                cfg.learner.encoder.repr_dim,
                cfg.learner.encoder.repr_dim,
            ],
            dropout=cfg.learner.encoder.dropout,
        )

        self.policy = GPT(
            GPTConfig(
                block_size=cfg.dataset.num_object_points + 21,
                input_dim=cfg.learner.encoder.repr_dim
                * 3,  # With VNMLP, we'll flatten the 3 embeddings per point into a single embedding
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
                    (cfg.learner.policy.position_encoding_type)
                    if "position_encoding_type" in cfg.learner.policy
                    else "all"
                ),
            )
        )

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

    def forward(self, input_data, stddev=None, target=None):

        # Project each point to a "token"
        # input_data: B, obs_horizon, num_total_points, 3
        batch_size, obs_horizon, num_total_points, _ = input_data.shape

        # Use of the VN Encoder will be different from how it was used in the point policy
        # VNMLP expects the input to be (B, obs_horizon, 3, N) # obs_horizon will be considered to be the feature dimension
        # because we're "flattening" the history
        input_rearranged = input_data.permute(0, 1, 3, 2)  # (B, obs_horizon, 3, N)

        # Input it to the VNMLP
        vn_repr_tokens = self.encoder(input_rearranged)
        # Now the output will be (B, repr_dim, 3, N)
        repr_dim = vn_repr_tokens.shape[1]
        vn_repr_tokens = vn_repr_tokens.reshape(
            batch_size, repr_dim * 3, num_total_points
        )  # (B, repr_dim * 3, N)
        vn_repr_tokens = vn_repr_tokens.permute(0, 2, 1)  # (B, N, repr_dim * 3)

        # Input it to the policy
        features = self.policy(vn_repr_tokens)  # (B, N, out_features)

        # Input to the head
        pred_action = self.head(
            features,
            stddev,
            **{
                "action_seq": target if target is not None else None,
            },  # NOTE: This action_seq is only used for the diffusion head
        )

        return pred_action
