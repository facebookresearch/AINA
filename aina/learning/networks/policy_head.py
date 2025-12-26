# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

import aina.utils.model_utils as utils
from aina.learning.networks.mlp import MLP


######################################### Deterministic Head #########################################
class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=False,
        loss_coef=1.0,
        loss_reduction="sum",
        predict_distribution=True,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.mse_loss = torch.nn.MSELoss(reduction=loss_reduction)
        self.predict_distribution = predict_distribution

    def forward(self, x, stddev=None, ret_action_value=False, **kwargs):
        if self.predict_distribution:
            mu = self.net(x)
            std = stddev if stddev is not None else 0.1
            std = torch.ones_like(mu) * std
            dist = utils.Normal(mu, std)
            if ret_action_value:
                return dist.mean
            else:
                return dist
        else:
            return self.net(x)

    def loss_fn(self, pred, target, mask=None, reduction="mean", **kwargs):
        if self.predict_distribution:
            log_probs = pred.log_prob(target)
            if mask is not None:
                log_probs = log_probs * mask / mask.mean()
            loss = -log_probs

            if reduction == "mean":
                loss = loss.mean() * self.loss_coef
            elif reduction == "none":
                loss = loss * self.loss_coef
            elif reduction == "sum":
                loss = loss.sum() * self.loss_coef
            else:
                raise NotImplementedError

            return loss

        if mask is not None:
            pred = pred * mask
            target = target * mask

        return self.mse_loss(pred, target) * self.loss_coef

    def pred_loss_fn(self, pred, target, mask=None, reduction="mean", **kwargs):
        if self.predict_distribution:
            dist = utils.TruncatedNormal(pred, 0.1)
            log_probs = dist.log_prob(target)
            loss = -log_probs

            if reduction == "mean":
                loss = loss.mean() * self.loss_coef
            elif reduction == "none":
                loss = loss * self.loss_coef
            elif reduction == "sum":
                loss = loss.sum() * self.loss_coef
            else:
                raise NotImplementedError

            return loss

        if mask is not None:
            dist = dist * mask
            target = target * mask
        return self.mse_loss(pred, target) * self.loss_coef
