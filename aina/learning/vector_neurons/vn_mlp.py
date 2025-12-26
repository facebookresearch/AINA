from typing import Callable, List, Optional

import torch
import torch.nn as nn

from aina.learning.vector_neurons.vn_layers import VNLeakyReLU, VNLinear


class VNMLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(VNLinear(in_dim, hidden_dim))
            layers.append(
                VNLeakyReLU(in_channels=hidden_dim, negative_slope=negative_slope)
            )
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(VNLinear(in_dim, hidden_channels[-1]))
        layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)
