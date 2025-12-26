# Copyright (c) Meta Platforms, Inc. and affiliates.

import glob
import os

import hydra
import numpy as np
import torch
import torch.utils.data as data
from termcolor import cprint


# Script to return dataloaders
def get_dataloaders(cfg):

    if "test_data_directories" in cfg.task and "all_data_directories" in cfg.task:
        train_dset, test_dset = get_split_datasets(cfg)

    elif "all_data_directories" in cfg.task:
        train_dset, test_dset = get_shuffled_datasets(cfg)

    else:
        raise ValueError("No data directories provided")

    return return_dataloaders_given_dsets(cfg, train_dset, test_dset)


def get_split_datasets(cfg):
    # Method to return train and test datasets, when a single list of directories is given
    if isinstance(cfg.task.all_data_directories, str):
        all_data_directories = glob.glob(cfg.task.all_data_directories)
    else:
        all_data_directories = cfg.task.all_data_directories

    if isinstance(cfg.task.test_data_directories, str):
        test_data_directories = glob.glob(cfg.task.test_data_directories)
    else:
        test_data_directories = cfg.task.test_data_directories

    print(f"TRAIN DATA DIRECTORIES: {all_data_directories}")
    print(f"TEST DATA DIRECTORIES: {test_data_directories}")
    train_dset = get_dataset_from_directories(
        all_data_directories, cfg.dataset, cfg.task.object_stats, cfg.task.text_prompts
    )
    test_dset = get_dataset_from_directories(
        test_data_directories, cfg.dataset, cfg.task.object_stats, cfg.task.text_prompts
    )

    return train_dset, test_dset


def get_shuffled_datasets(cfg):
    # Method to return train and test datasets, when a single list of directories is given

    if isinstance(cfg.task.all_data_directories, str):
        all_data_directories = glob.glob(cfg.task.all_data_directories)
    else:
        all_data_directories = cfg.task.all_data_directories

    train_dset = get_dataset_from_directories(
        all_data_directories, cfg.dataset, cfg.task.object_stats, cfg.task.text_prompts
    )

    train_dset_size = int(len(train_dset) * cfg.train_dset_split)
    test_dset_size = len(train_dset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(
        train_dset,
        [train_dset_size, test_dset_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    return train_dset, test_dset


def get_demos_with_object_poses(processed_demos):
    good_demos = []
    for processed_demo in processed_demos:
        if os.path.exists(f"{processed_demo}/object-poses-in-base.npy"):
            good_demos.append(processed_demo)
    return good_demos


def get_demos_with_all_objects(text_prompts, processed_demos):
    good_demos = []
    for processed_demo in processed_demos:
        if os.path.exists(f"{processed_demo}/detected-points-counter.npy"):
            detected_objects = np.load(f"{processed_demo}/detected-points-counter.npy")
            if len(detected_objects) == len(text_prompts):
                good_demos.append(processed_demo)
        else:
            good_demos.append(processed_demo)
    return good_demos


def get_dataset_from_directories(
    directories, dataset_cfg, object_stats, text_prompts=None
):
    datasets = []
    for data_dir in directories:
        if isinstance(
            data_dir, str
        ):  # This allows strs inside the list as well -> will make it easier to provide larger datasets
            data_dirs = glob.glob(data_dir)
        else:
            data_dirs = [data_dir]

        # Eliminate the demos that have less than the number of text prompts
        if text_prompts is not None:
            pre_filter_len = len(data_dirs)
            data_dirs = get_demos_with_all_objects(text_prompts, data_dirs)
            cprint(
                f"Eliminated {pre_filter_len - len(data_dirs)}/{pre_filter_len} demos with less than {len(text_prompts)} objects | Remaining: {len(data_dirs)}",
                "yellow",
            )

        # Eliminate the demos that have no object poses
        pre_filter_len = len(data_dirs)
        data_dirs = get_demos_with_object_poses(data_dirs)
        cprint(
            f"Eliminated {pre_filter_len - len(data_dirs)}/{pre_filter_len} demos with no object poses | Remaining: {len(data_dirs)}",
            "yellow",
        )

        for data_dir in data_dirs:
            datasets.append(
                hydra.utils.instantiate(
                    dataset_cfg,
                    data_dir=data_dir,
                    object_stats=object_stats,  # NOTE: Test dataset also receives augmented datapoints!
                )
            )
    whole_dset = data.ConcatDataset(datasets)
    return whole_dset


def return_dataloaders_given_dsets(cfg, train_dset, test_dset):
    train_loader = data.DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = data.DataLoader(
        test_dset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, test_loader, train_dset, test_dset
