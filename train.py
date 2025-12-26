# Copyright (c) Meta Platforms, Inc. and affiliates.

import hydra
from omegaconf import DictConfig

from aina.training.workspace import TrainingWorkspace


@hydra.main(
    version_base=None,
    config_path="cfgs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    workspace = TrainingWorkspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
