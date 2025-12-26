import os
import pickle
from base64 import b64encode
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.models.core.model_utils import get_points_on_a_grid
from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from PIL import Image as im
from tqdm import tqdm

from aina.utils.file_ops import get_repo_root
from aina.utils.video_recorder import VideoRecorder

# NOTE: Taken from https://github.com/irmakguzey/object-rewards/blob/main/object_rewards/point_tracking/co_tracker.py
# modified to work with points


class CoTrackerPoints:
    def __init__(
        self,
        device="cuda",
        checkpoint_path=None,
        is_online=True,
        frame_by_frame=False,
    ):
        self.device = device
        self.is_online = is_online

        if checkpoint_path is None:
            checkpoint_path = (
                f"{get_repo_root()}/submodules/co-tracker/checkpoints/cotracker2.pth"
            )

        if is_online:

            self.cotracker = CoTrackerOnlinePredictor(
                checkpoint=checkpoint_path, single_frame=frame_by_frame
            ).to(
                device
            )  # Try this if doesn't work it should work like above
        else:
            self.cotracker = CoTrackerPredictor(checkpoint=checkpoint_path).to(device)

        self.frame_by_frame = frame_by_frame

    def to(self, device):
        self.device = device
        self.cotracker.to(device)

    def track_points_from_video(
        self, video_path, points, frame_matches=-1, return_numpy=False
    ):
        """
        Code to track points from video.
        With respect to given parameters, it'll either use sliding window or the whole frames to track the points

        Parameters:
        ----------

        frames: np.ndarray
            The frames to track the points on. Shape: (T, H, W, C)

        points: np.ndarray
            2D points to track. Shape: (N, 2). Cotracker inputs queries in format (T, N, 2) where T is the frame id per point. Since we use initial points, we set all T to be 0s.
        """

        if frame_matches != -1:
            frames = read_video_from_path(video_path)[-frame_matches:, :]
        else:
            frames = read_video_from_path(video_path)[:, :]

        return self.track_points_from_frames(
            frames, points, permute_frames=False, return_numpy=return_numpy
        )

    def track_points_from_frames(
        self,
        frames,
        points,
        return_numpy=False,
        permute_frames=True,
        return_visibility=False,
    ):
        """
        Code to track points from frames.
        With respect to given parameters, it'll either use sliding window or the whole frames to track the points

        Parameters:
        ----------

        frames: np.ndarray
            The frames to track the points on. Shape: (T, H, W, C)

        points: np.ndarray
            2D points to track. Shape: (N, 2). Cotracker inputs queries in format (B, N, 3).
            B: Batch size, N: Number of Points, 3: (t,x,y) - t is the frame id per point
        """

        # Preprocess images and points
        if permute_frames:
            frames = (
                torch.from_numpy(frames)
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(self.device)
            )

        points = torch.from_numpy(points)
        points = torch.cat([torch.zeros(points.shape[0], 1), points], dim=1).to(
            self.device
        )[None]
        # print(f"points.shape: {points.shape}")

        if self.is_online:
            if self.frame_by_frame:
                return self.track_online_by_single_frame(
                    frames,
                    queries=points,
                    return_numpy=return_numpy,
                    return_visibility=return_visibility,
                )
            return self.track_online_by_batch(
                frames,
                queries=points,
                return_numpy=return_numpy,
                return_visibility=return_visibility,
            )

        else:
            return self.track_offline(
                frames,
                queries=points,
                return_numpy=return_numpy,
                return_visibility=return_visibility,
            )

    def track_offline(
        self,
        frames,
        queries,
        return_numpy=True,
    ):

        print("** GETTING TRACKS OFFLINE **")

        pred_tracks, _ = self.cotracker(frames, queries=queries)

        if (not pred_tracks is None) and return_numpy:
            return pred_tracks[0, :].detach().cpu().numpy()

        return pred_tracks

    def track_online_by_batch(
        self,
        frames,
        queries=None,
        return_numpy=True,
        return_visibility=False,
        # text_prompt=None,
    ):

        print("** GETTING TRACKS BY BATCH **")

        # Start the tracking online
        self.cotracker(
            video_chunk=frames,
            is_first_step=True,
            queries=queries,
        )
        # batch_size = 64
        pbar = tqdm(
            total=len(
                range(0, frames.shape[1] - self.cotracker.step, self.cotracker.step)
            )
        )
        pred_tracks = None  # Initialize variables
        for ind in range(0, frames.shape[1] - self.cotracker.step, self.cotracker.step):
            pred_tracks, visibility = self.cotracker(
                video_chunk=frames[:, ind : ind + (self.cotracker.step * 2)],
                is_first_step=False,
            )
            pbar.update(1)
            pbar.set_description(
                "ind: {}, pred_tracks.shape: {} | visibility.shape: {}".format(
                    ind, pred_tracks.shape, visibility.shape
                )
            )

        pbar.close()

        if (not pred_tracks is None) and return_numpy:
            pred_tracks = pred_tracks[0, :].detach().cpu().numpy()
            visibility = visibility[0, :].detach().cpu().numpy()

        if return_visibility:
            return pred_tracks, visibility

        return pred_tracks

    def track_online_by_single_frame(self, frames, queries, return_numpy=True):

        print("** GETTING TRACKS BY SINGLE FRAME **")

        frames = torch.from_numpy(frames).float().to(self.device)

        window_frames = [
            frames[0] for _ in range(self.cotracker.step * 2)
        ]  # For the beginning
        is_first_frame = True
        pbar = tqdm(total=len(frames))

        pred_tracks = None  # Initialize pred_tracks to avoid unbound error
        for frame_id, frame in enumerate(frames):

            pred_tracks, _ = self.single_frame_track(
                queries=queries,
                is_first_step=is_first_frame,
                window_frames=window_frames,
            )
            is_first_frame = False

            window_frames.append(frame)
            pbar.update(1)
            if frame_id != 0:
                pbar.set_description(f"pred_tracks: {pred_tracks.shape}")

        pbar.close()

        if (not pred_tracks is None) and return_numpy:
            return (
                pred_tracks[0, (self.cotracker.step - 1) * 2 :].detach().cpu().numpy()
            )
        return pred_tracks

    def single_frame_track(self, queries, is_first_step, window_frames):

        # Create the video_chunk
        video_chunk = (
            torch.tensor(
                torch.stack(window_frames[-self.cotracker.step * 2 :]),
                device=window_frames[0].device,
            )  # NOTE: If this doesn't work, then will include all the current frames
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        # Pass it through the model
        return self.cotracker(
            video_chunk=video_chunk,
            is_first_step=is_first_step,
            queries=queries if is_first_step else None,
        )

    def visualize_tracks(
        self,
        pred_tracks,
        frames,
        video_name,
        dump_dir="./videos",
    ):

        video_recorder = VideoRecorder(
            save_dir=Path(dump_dir), resize_and_transpose=False
        )

        for i in range(len(pred_tracks)):
            curr_features = pred_tracks[i, :]

            frame_img = frames[i, :]

            for feat in curr_features:
                x, y = np.int32(feat.ravel())
                frame_img = cv2.circle(frame_img, (x, y), 1, (255, 0, 0), -1)

            if frame_img.shape[0] == 3:
                frame_img = np.transpose(
                    frame_img, (2, 0, 1)
                )  # NOTE: I have not idea how we didn't have this problem previously

            # print(f"frame_img.shape: {frame_img.shape}")
            video_recorder.record(obs=frame_img)

        video_recorder.save(f"{video_name}.mp4")
