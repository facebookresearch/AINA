# Copyright (c) Meta Platforms, Inc. and affiliates.

import cv2
import numpy as np
import torch


@torch.jit.script
def project_kannala_brandt(
    points_optical: torch.Tensor,  # (..., 3)
    params: torch.Tensor,  # (N_params,)
    num_k: int,
    use_tangential: bool = True,
    use_thin_prism: bool = True,
    use_single_focal_length: bool = True,
):
    """
    PyTorch version of the Eigen projection model.
    points_optical: (..., 3)  -- 3D points in optical frame
    params: camera parameter vector
    num_k: number of radial distortion coefficients
    """

    eps = 1e-9
    x, y, z = points_optical.unbind(-1)
    inv_z = 1.0 / (z + eps)
    a = x * inv_z
    b = y * inv_z
    r_sq = a * a + b * b
    r = torch.sqrt(r_sq + eps)
    th = torch.atan(r)
    theta_sq = th * th

    # Radial distortion polynomial
    th_radial = torch.ones_like(th)
    theta2is = theta_sq.clone()
    start_k = 3 if use_single_focal_length else 4
    for i in range(num_k):
        th_radial += theta2is * params[start_k + i]
        theta2is *= theta_sq

    # th / r (handle small r)
    th_divr = torch.where(r < eps, torch.ones_like(r), th / r)

    # Radially distorted coordinates
    xr = (th_radial * th_divr) * a
    yr = (th_radial * th_divr) * b
    xr_yr = torch.stack([xr, yr], dim=-1)
    xr_yr_sqnorm = (xr_yr**2).sum(
        -1, keepdim=True
    )  # NOTE: is squaredNorm equal to this?

    uv_distorted = xr_yr.clone()

    # Optional tangential distortion
    if use_tangential:
        start_p = start_k + num_k
        p = params[
            start_p : start_p + 2
        ]  # There are 2 tangential distortion coefficients
        temp = 2.0 * (xr_yr @ p)
        uv_distorted = uv_distorted + temp.unsqueeze(-1) * xr_yr + xr_yr_sqnorm * p

    # Optional thin prism
    if use_thin_prism:  # There are 4 thin prism coefficients
        start_s = start_k + num_k + (2 if use_tangential else 0)
        s = params[start_s : start_s + 4]
        r2 = xr_yr_sqnorm.squeeze(-1)
        r4 = r2 * r2
        uv_distorted[..., 0] += s[0] * r2 + s[1] * r4
        uv_distorted[..., 1] += s[2] * r2 + s[3] * r4

    # Focal lengths and principal point
    if use_single_focal_length:
        f = params[0]
        cu = params[1]
        cv = params[2]
        uv = f * uv_distorted + torch.stack([cu, cv]).to(
            device=params.device, dtype=torch.float32
        )
    else:
        fu = params[0]
        fv = params[1]
        cu = params[2]
        cv = params[3]
        uv = uv_distorted * torch.stack([fu, fv]).to(
            device=params.device, dtype=torch.float32
        ) + torch.stack([cu, cv]).to(device=params.device, dtype=torch.float32)

    return uv


@torch.jit.script
def rectify_image_with_kb(
    image: torch.Tensor, Rn_Kinv: torch.Tensor, params: torch.Tensor, num_k: int
):
    device = image.device
    H, W = image.shape
    jj, ii = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="xy",
    )
    homog = torch.stack([jj, ii, torch.ones_like(jj)], dim=-1).reshape(-1, 3)
    direction_rays = torch.matmul(Rn_Kinv, homog.T).T
    coords = project_kannala_brandt(direction_rays, params, num_k)
    u, v = coords[:, 0], coords[:, 1]

    grid_u = (u / (W - 1)) * 2 - 1
    grid_v = (v / (H - 1)) * 2 - 1
    grid = torch.stack([grid_u, grid_v], dim=1).reshape(H, W, 2)
    rectified = torch.nn.functional.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return rectified.squeeze(0).permute(1, 2, 0)


def get_baseline_between_cameras(homogenous_matrix):
    """
    Will return the horizontal displacement between two rectified cameras.
    Assuming the cameras are rectified, this will be equal to the norm of the translation only.

    Parameters:
    -----------
    homogenous_matrix: np.darray
        (4,4) matrix that is [R|t] with [0,0,0,1] appended as the last row.

    """
    return np.linalg.norm(
        homogenous_matrix[:3, 3]
    )  # Norm of the translation between cameras
