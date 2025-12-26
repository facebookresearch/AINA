import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from aina.utils.constants import FOUNDATION_STEREO_PATH
from aina.utils.points import *


# Taken from https://github.com/NVlabs/FoundationStereo/blob/master/scripts/run_demo.py and modified with respect to users need
# Wrapper class to use foundation stereo
class FoundationStereoWrapper:
    def __init__(self, foundation_stereo_path=FOUNDATION_STEREO_PATH, device="cuda"):

        sys.path.append(FOUNDATION_STEREO_PATH)
        from core.foundation_stereo import FoundationStereo

        self.foundation_stereo_path = foundation_stereo_path
        checkpoint_dir = f"{foundation_stereo_path}/pretrained_models/23-51-11"
        model_path = f"{checkpoint_dir}/model_best_bp2.pth"
        cfg_path = f"{checkpoint_dir}/cfg.yaml"

        random_seed = 0
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_grad_enabled(False)

        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg = OmegaConf.create(cfg)
        self.foundation_stereo = FoundationStereo(cfg)

        # Load the checkpoint
        checkpoint = torch.load(model_path)
        self.foundation_stereo.load_state_dict(checkpoint["model"])
        self.foundation_stereo.to(device)
        self.foundation_stereo.eval()

    def get_disparity(self, image0, image1, points_2d=None, visualize=False):
        """
        Method to get the disparity with respect to the image0.
        image0: Left image
        image1: Right image
        Left and Right should not be mixed
        """

        sys.path.append(self.foundation_stereo_path)
        from core.utils.utils import InputPadder

        # breakpoint()

        if len(image0.shape) == 2:  # Image is black and white
            image0 = np.repeat(np.expand_dims(image0, -1), 3, -1)
        if len(image1.shape) == 2:
            image1 = np.repeat(np.expand_dims(image1, -1), 3, -1)
            # image0 = np.expand_dims(image0, -1)
            # image1 = np.expand_dims(image1, -1)

        # breakpoint()

        img0_ori = image0.copy()
        H, W = image0.shape[:2]
        img0 = torch.as_tensor(image0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(image1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            disparity = self.foundation_stereo.forward(
                img0, img1, iters=32, test_mode=True
            )

        disparity = padder.unpad(disparity.float())
        disparity = disparity.data.cpu().numpy().reshape(H, W)

        yy, xx = np.meshgrid(
            np.arange(disparity.shape[0]), np.arange(disparity.shape[1]), indexing="ij"
        )
        us_right = xx - disparity
        invalid = us_right < 0
        disparity[invalid] = np.inf

        if not points_2d is None:
            points_depth = []
            for p in points_2d:
                x_p = min(p[0], disparity.shape[0] - 1)
                y_p = min(p[1], disparity.shape[1] - 1)
                points_depth.append(disparity[x_p, y_p])
            return np.stack(points_depth, axis=0)
        return disparity

    def get_pcd(
        self,
        image_left,
        K,
        baseline=None,
        disparity=None,
        depth=None,
        xyz_map=None,
        points_2d=None,
        sample=True,
    ):

        if depth is None:
            depth = (
                K[0, 0] * baseline / disparity
            )  # NOTE: These are values used in the original demo
        if xyz_map is None:
            xyz_map = depth_to_xyz_map(depth, K, zmin=0.005)
            cprint(f"XYZ map shape: {xyz_map.shape}", "green")
        if not points_2d is None:
            points_2d[:, 0] = np.clip(points_2d[:, 0], 0, image_left.shape[1] - 1)
            points_2d[:, 1] = np.clip(points_2d[:, 1], 0, image_left.shape[0] - 1)
            xyz_map = xyz_map[points_2d[:, 1], points_2d[:, 0]]
            image_left = image_left[points_2d[:, 1], points_2d[:, 0]]

        # If the image is black and white, then we need to repeat the image 3 times
        print(f"image_left shape: {image_left.shape}")
        if len(image_left.shape) < len(xyz_map.shape):
            image_left = np.repeat(np.expand_dims(image_left, -1), 3, -1)

        print(
            f"repeated image left shape: {image_left.shape} | xyz_map shape: {xyz_map.shape}"
        )

        pcd = to_open3d_pcd(
            xyz_map.reshape(-1, 3),
            image_left.reshape(-1, 3),
        )

        if sample:
            pcd = pcd.farthest_point_down_sample(10000)

        return pcd
