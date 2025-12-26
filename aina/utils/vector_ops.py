# Copyright (c) Meta Platforms, Inc. and affiliates.

import glob
import os

import numpy as np
import torch


def handle_normalization(input, stats, normalize, mean_std=False):
    if mean_std:
        mean, std = stats[0], stats[1]
        if normalize:
            input = (input - mean) / (std + 1e-10)
        else:
            input = (input * std + mean).numpy()
    else:
        min, max = stats[0], stats[1]
        if normalize:
            if isinstance(input, np.ndarray):
                input = np.clip(input, min, max)
            else:
                input = torch.clamp(input, min, max)
            input = (input - min) / (max - min + 1e-10)
        else:
            input = input * (max - min) + min
            if not isinstance(input, np.ndarray):
                input = input.numpy()

    return input


def calculate_stats(all_data_dirs, mean_std_norm=False):
    """
    Method to calculate stats of the whole dataset.
    This is without per demo but by with all of the data.
    """
    object_points = []
    for data_dir in all_data_dirs:
        print(f"Data dir: {data_dir}")

        if not os.path.exists(f"{data_dir}/object-poses-in-base.npy"):
            continue
        object_points.append(
            np.load(f"{data_dir}/object-poses-in-base.npy").reshape(-1, 3)
        )
        object_points.append(
            np.load(f"{data_dir}/hand-poses-in-base.npy").reshape(-1, 3)
        )

        if np.isnan(object_points[-1]).any():
            print(f"NANs in {data_dir}")
            print(f"Object points: {object_points[-1]}")
            raise ValueError("NANs in object points")

    object_points = np.concatenate(object_points, axis=0)
    print(f"All object points: {object_points.shape}")
    if mean_std_norm:
        object_stats = [object_points.mean(axis=(0)), object_points.std(axis=(0))]
    else:
        mean_points = object_points.mean(axis=(0))
        min_points = mean_points - np.array(
            [1, 1, 1]
        )  # There are very minimum points that causes this stat to be wrong!
        max_points = mean_points + np.array(
            [1, 1, 1]
        )  # There are very maximum points that causes this stat to be wrong!
        object_stats = [min_points, max_points]

    return [arr.tolist() for arr in object_stats]
