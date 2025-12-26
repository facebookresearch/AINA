# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from aina.utils.constants import HAMER_PATH
from aina.utils.file_ops import suppress

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(HAMER_PATH)

try:
    from hamer.hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.hamer.models import DEFAULT_CHECKPOINT, load_hamer
    from hamer.hamer.utils import recursive_to
    from hamer.hamer.utils.renderer import Renderer, cam_crop_to_full
    from hamer.vitpose_model import ViTPoseModel
except ModuleNotFoundError:
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.models import DEFAULT_CHECKPOINT, load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from vitpose_model import ViTPoseModel

from torch.utils.data._utils.collate import default_collate

# Modified from https://github.com/vliu15/egozero/blob/main/utils/hand_utils.py


@torch.no_grad()
def run_hamer_from_video(
    video,
    checkpoint=DEFAULT_CHECKPOINT,
    body_detector="vitdet",
    render=True,
    is_right_hand=False,
):

    from tqdm import tqdm

    with suppress(stdout=True):
        os.chdir(os.path.join(HAMER_PATH))
        model, model_cfg, detector, cpm, device, renderer = load_hamer_model(
            checkpoint, body_detector
        )
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    if isinstance(video, str):
        frames = iio.imread(video)
    else:
        frames = np.stack(video)

    assert frames.ndim == 4  # shape (n, h, w, 3)

    all_frames = []
    all_fingertips = []
    n_detected = 0
    n_missing = 0
    pbar = tqdm(total=len(frames), desc="processing hamer")
    for frame in frames:
        # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # print(f"Frame: {frame.shape}")
        fingertips, hamer_frame = detect_hamer_in_frame(
            model,
            model_cfg,
            detector,
            cpm,
            device,
            renderer,
            frame,
            render=render,
            is_right_hand=is_right_hand,
        )
        if fingertips is not None:
            n_detected += 1
        else:
            n_missing += 1
            fingertips = np.zeros((21, 3))

        hamer_frame_rgb = cv2.cvtColor(hamer_frame, cv2.COLOR_BGR2RGB)
        hamer_frame_rgb_uint8 = np.uint8(hamer_frame_rgb)
        all_frames.append(hamer_frame_rgb_uint8)
        all_fingertips.append(fingertips)

        pbar.update(1)

    pbar.close()
    del pbar
    return all_frames, all_fingertips


def load_hamer_model(checkpoint, body_detector):
    model, model_cfg = load_hamer(checkpoint)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    from pathlib import Path

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    detector = None
    if body_detector == "vitdet":
        from detectron2.config import LazyConfig

        try:
            from hamer import hamer

            cfg_path = (
                Path(hamer.__file__).parent
                / "configs"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )
        except:
            import hamer

            cfg_path = (
                Path(hamer.__file__).parent
                / "configs"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )

        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == "regnety":
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    return model, model_cfg, detector, cpm, device, renderer


def detect_hamer_in_frame(
    model,
    model_cfg,
    detector,
    cpm,
    device,
    renderer,
    img_cv2,
    render=True,
    focal_length=torch.tensor([916.229]),
    rescale_factor=2,
    is_right_hand=False,
    return_2d=False,  # If this is set to True, it won't predict the 3d keypoints
):
    # Detect humans in image
    # print(f"img_cv2.shape: {img_cv2.shape}")
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )
    bboxes = []
    bbox_sizes = []
    is_hand = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        # print(f"left_hand_keyp: {left_hand_keyp.shape} - right_hand_keyp: {right_hand_keyp.shape}")

        # Rejecting not confident detections and left hand
        if is_right_hand:
            keyp = right_hand_keyp
        else:
            keyp = left_hand_keyp
        # print(f"keyp: {keyp.shape} - keyp: {keyp}")
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_size < 1000:
                continue
            bboxes.append(bbox)
            bbox_sizes.append(bbox_size)
            is_hand.append(1 if is_right_hand else 0)

    if len(bboxes) == 0:
        return None, img, None, None

    # if return_2d:
    #     return keyp[:, :2], img, None

    # Choose largest bbox prediction
    # largest_bbox_idx = np.argmax(bbox_sizes)
    # boxes = np.array([bboxes[largest_bbox_idx]])
    # is_hand = np.array([1]) if is_right_hand else np.array([0])

    boxes = np.array(bboxes)
    is_hand = np.array(is_hand)

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(
        model_cfg, img, boxes, is_hand, rescale_factor=rescale_factor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )
    all_predicted_3d = []
    all_predicted_2d = []
    all_verts = []
    all_cam_t = []

    with suppress(stdout=True):

        for batch in dataloader:
            # batch = default_collate([dataset[i]])
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            all_predicted_3d.append(out["pred_keypoints_3d"][0:1])

            multiplier = 2 * batch["right"][0:1] - 1
            pred_cam = out["pred_cam"][0:1]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()[0:1]
            box_size = batch["box_size"].float()[0:1]
            img_size = batch["img_size"].float()[0:1]

            centered_keypoints = out["pred_keypoints_2d"][0:1]
            centered_keypoints *= box_size[0]
            # breakpoint()
            # cprint(f"Centered keypoints: {centered_keypoints.shape}", "magenta")
            # cprint(f"Box center: {box_center.shape}", "magenta")
            all_predicted_2d.append(centered_keypoints + box_center)

            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            verts = out["pred_vertices"][0].detach().cpu().numpy()
            is_right = batch["right"][0].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[0].reshape(3)
            all_verts.append(verts)
            all_cam_t.append(cam_t)

        closest_hand_idx = np.argmin(np.array(all_cam_t)[:, -1])
        verts = all_verts[closest_hand_idx]
        cam_t = all_cam_t[closest_hand_idx]
        fingertips_2d = all_predicted_2d[closest_hand_idx].cpu().numpy().squeeze(0)
        fingertips = all_predicted_3d[closest_hand_idx].cpu().numpy().squeeze(0)
        if render:
            misc_args = dict(
                mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
                scene_bg_color=(1, 1, 1),
                focal_length=focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                [verts],
                cam_t=[cam_t],
                render_res=img_size[0],
                is_right=[1 if is_right_hand else 0],
                **misc_args,
            )

            input_img = img.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )  # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )
        else:
            input_img_overlay = img.astype(np.float32)[:, :, ::-1] / 255.0

        if not is_right_hand:  # Flip x coordinates if left hand
            fingertips[:, 0] = -1 * fingertips[:, 0], cam_t
        return (
            fingertips,
            255 * input_img_overlay[:, :, ::-1],
            cam_t,
            fingertips_2d,
        )


class HandPoseDetectionWrapper:
    def __init__(self, hamer_path, return_2d=False):
        print(f"*** Initializing HamerWrapper with hamer_path: {hamer_path} ***")
        with suppress(stdout=True):
            os.chdir(os.path.join(hamer_path))
            (
                self.model,
                self.model_cfg,
                self.detector,
                self.cpm,
                self.device,
                self.renderer,
            ) = load_hamer_model(
                checkpoint=DEFAULT_CHECKPOINT,
                body_detector="vitdet",
            )
            os.chdir(os.path.join(os.path.dirname(__file__), ".."))

        # If this is true, hamer will not predict the 3d keypoints, but they will be triangulated from multiple 2d views
        self.return_2d = return_2d

    def get_hand_pose(
        self,
        image,
        render=True,
        is_right_hand=True,
        focal_length=torch.tensor([916.229], device="cuda"),
    ):
        """
        Method to return hand pose with respect to the image.
        It will return the fingertips and the rendered image.
        """
        fingertips, hamer_frame, wrist_pose, fingertips_2d = detect_hamer_in_frame(
            self.model,
            self.model_cfg,
            self.detector,
            self.cpm,
            self.device,
            self.renderer,
            image,
            render=render,
            is_right_hand=is_right_hand,
            focal_length=focal_length,
            return_2d=self.return_2d,
        )

        # breakpoint()

        return fingertips, hamer_frame, wrist_pose, fingertips_2d

    def get_hand_poses_in_video(self, video, render=True, is_right_hand=True):
        if isinstance(video, str):
            frames = iio.imread(video)
        else:
            frames = np.stack(video)

        assert frames.ndim == 4  # shape (n, h, w, 3)

        all_frames = []
        all_fingertips = []
        n_detected = 0
        n_missing = 0
        pbar = tqdm(total=len(frames), desc="processing hamer")
        for frame in frames:
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # print(f"Frame: {frame.shape}")
            fingertips, hamer_frame = self.get_hand_pose(
                frame,
                render=render,
                is_right_hand=is_right_hand,
            )
            if fingertips is not None:
                n_detected += 1
            else:
                n_missing += 1
                fingertips = np.zeros((21, 3))

            hamer_frame_rgb = cv2.cvtColor(hamer_frame, cv2.COLOR_BGR2RGB)
            hamer_frame_rgb_uint8 = np.uint8(hamer_frame_rgb)
            all_frames.append(hamer_frame_rgb_uint8)
            all_fingertips.append(fingertips)

            pbar.update(1)

        pbar.close()
        del pbar
        return all_frames, all_fingertips
