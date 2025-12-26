# Copyright (c) Meta Platforms, Inc. and affiliates.

from aina.utils.file_ops import load_function


def init_learner(cfg, device):
    fn = load_function(f"aina.learning.initialize_learner.init_{cfg.learner.name}")
    return fn(cfg, device)


def init_vn_point_policy(cfg, device):
    from aina.learning.vector_neurons.vn_pp_learner import VNPPLearner

    learner = VNPPLearner(cfg=cfg)
    learner.to(device)

    return learner
