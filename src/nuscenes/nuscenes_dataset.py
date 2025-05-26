import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


# helpers for debugging arbitrary python objects
def get_type_tree(data):
    """
    Recursively builds a tree of types from the input data, which can be a
    nested collection (dict, set, list, tuple, etc.).
    """
    if isinstance(data, (dict, list, tuple, set)):
        tree = {}
        if isinstance(data, dict):
            for key, value in data.items():
                tree[key] = get_type_tree(value)
        else:
            for idx, item in enumerate(data):
                tree[idx] = get_type_tree(item)
        return tree
    else:
        return type(data).__name__


def print_type_tree(tree, level=0):
    """
    Recursively prints the type tree with indentation to represent its structure.
    """
    indent = "  " * level
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}{key}:")
            print_type_tree(value, level + 1)
    else:
        print(f"{indent}{tree}")


class NuScenesDataset(Dataset):
    def __init__(
        self, nuscenes_path, version="v1.0-mini", future_seconds=3.0, future_hz=2
    ):
        """
        Initialize the NuScenes dataset

        Args:
            nuscenes_path: Path to the NuScenes dataset root
            version: NuScenes dataset version
            future_seconds: Number of seconds in the future for trajectory prediction
            future_hz: Frequency in Hz for trajectory points
        """
        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
        self.samples = self.nusc.sample
        self.future_seconds = future_seconds
        self.future_hz = future_hz
        self.future_steps = int(future_seconds * future_hz)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get item at the given index

        Returns:
            tuple: (sensor_data, trajectory)
                - sensor_data: dict of dicts with keys like "CAM_FRONT" containing
                  "img", "T_global_to_cam", "intrinsics"
                - trajectory: future trajectory as numpy array
        """
        sample = self.samples[idx]

        # Collect sensor data
        sensor_data = {}
        for sensor_name, sensor_token in sample["data"].items():
            # Only process camera sensors
            if "CAM" not in sensor_name:
                continue

            sensor_dict = {}
            sensor_sample_data = self.nusc.get("sample_data", sensor_token)

            # Get image
            img_path = os.path.join(self.nusc.dataroot, sensor_sample_data["filename"])
            sensor_dict["img"] = np.array(Image.open(img_path))

            # Get calibration info
            calibrated_sensor = self.nusc.get(
                "calibrated_sensor", sensor_sample_data["calibrated_sensor_token"]
            )

            # Get intrinsic matrix
            sensor_dict["intrinsics"] = np.array(calibrated_sensor["camera_intrinsic"])

            # Get extrinsic transformation matrix (global to camera)
            rotation = calibrated_sensor["rotation"]
            translation = calibrated_sensor["translation"]

            # Convert rotation quaternion to rotation matrix
            rotation_matrix = Quaternion(rotation).rotation_matrix

            # Build the transformation matrix (4x4)
            T_global_to_cam = np.eye(4)
            T_global_to_cam[:3, :3] = rotation_matrix
            T_global_to_cam[:3, 3] = translation

            sensor_dict["T_global_to_cam"] = T_global_to_cam

            sensor_data[sensor_name] = sensor_dict

        # Get future trajectory
        trajectory = self._get_future_trajectory(sample)

        # position at t+1 is position at t, translated.
        # thus, we want to assume position at t=t0 is origin
        # and position at t+1 is in the same coordinate system
        # thus shifting the trajectory to be relative to the first point

        # Get the first point in the trajectory
        # --- NEW: transform future way-points from world → ego (vehicle) frame ---
        current_lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose_0 = self.nusc.get("ego_pose", current_lidar_sd["ego_pose_token"])
        t0 = np.asarray(ego_pose_0["translation"])  # (3,)
        q0 = Quaternion(ego_pose_0["rotation"])  # ego←world
        R_world_to_ego = q0.inverse.rotation_matrix  # 3×3

        # trajectory is (N,3) in world.  Convert in one shot:
        trajectory_local = (R_world_to_ego @ (trajectory - t0).T).T  # (N,3)
        trajectory_local = trajectory_local.astype(np.float32)

        return (sensor_data, trajectory_local)

    def _get_future_trajectory(self, sample):
        """
        Extract the future trajectory of the ego vehicle

        Args:
            sample: Current sample

        Returns:
            np.ndarray: Array of shape (num_future_points, 3) containing future positions
        """
        # Get the current sample and timestamp
        current_sample = sample
        current_timestamp = current_sample["timestamp"]

        # Get current ego pose
        current_ego_pose = self.nusc.get(
            "ego_pose",
            self.nusc.get("sample_data", current_sample["data"]["LIDAR_TOP"])[
                "ego_pose_token"
            ],
        )

        # Initialize trajectory array
        trajectory = []

        # Get future samples
        sample_token = current_sample["next"]
        while len(trajectory) < self.future_steps and sample_token != "":
            # Get next sample
            next_sample = self.nusc.get("sample", sample_token)

            # Get ego pose for next sample
            ego_pose = self.nusc.get(
                "ego_pose",
                self.nusc.get("sample_data", next_sample["data"]["LIDAR_TOP"])[
                    "ego_pose_token"
                ],
            )

            # Extract position
            position = np.array(ego_pose["translation"])
            trajectory.append(position)

            # Move to next sample
            sample_token = next_sample["next"]

        # If we don't have enough future points, pad with the last position
        if trajectory:
            last_pos = trajectory[-1]
            while len(trajectory) < self.future_steps:
                trajectory.append(last_pos.copy())
        else:
            # If we have no future points, use the current position
            current_pos = np.array(current_ego_pose["translation"])
            trajectory = [current_pos] * self.future_steps

        return np.array(trajectory)

    def project_traj_on_img(self, traj, img, cam_intrinsic, T_global_to_cam):
        """
        Project 3D trajectory points onto a 2D camera image

        Args:
            traj: Array of shape (num_points, 3) containing future positions in global frame
            img: Camera image as numpy array
            cam_intrinsic: Camera intrinsic matrix (3x3)
            T_global_to_cam: Transformation matrix from global to camera frame (4x4)

        Returns:
            numpy.ndarray: Image with the projected trajectory drawn on it
        """
        # Make a copy of the image to avoid modifying the original
        img_with_traj = img.copy()

        # Convert trajectory points from shape (N, 3) to (3, N) for view_points
        traj_points = traj.transpose()

        # Create homogeneous coordinates for the trajectory points
        homo_traj = np.vstack((traj_points, np.ones(traj_points.shape[1])))

        # Transform trajectory points from global to camera coordinates
        print(
            f"[shape debug]: homo_traj={homo_traj.shape}, T_global_to_cam={T_global_to_cam.shape}"
        )
        homo_traj = T_global_to_cam @ homo_traj
        print(f"[shape debug]: after transform homo_traj={homo_traj.shape}")
        cam_traj = homo_traj[:3, :]

        cam_traj = cam_traj.squeeze()[:3, ...]

        print("[debug] ", cam_traj.shape)

        # Only keep points in front of the camera (positive z)
        """
        valid_mask = cam_traj[2, :] > 0
        if not np.any(valid_mask):
            # No valid points to project
            return img_with_traj
            
        cam_traj = cam_traj[:, valid_mask]
        """

        # Project points from 3D camera coordinates to 2D image coordinates
        img_points = view_points(cam_traj, cam_intrinsic, normalize=True)

        # Convert to pixel coordinates and filter points outside image
        height, width = img.shape[:2]
        pixel_coords = img_points[:2, :].T

        in_image = (
            (0 <= pixel_coords[:, 0])
            & (pixel_coords[:, 0] < width)
            & (0 <= pixel_coords[:, 1])
            & (pixel_coords[:, 1] < height)
        )

        if not np.any(in_image):
            print("Trajectory points not in image.")
            # No points visible in the image
            return img_with_traj

        valid_pixels = pixel_coords[in_image].astype(int)

        # Draw the trajectory points with increasing size based on distance into the future
        future_indices = np.arange(traj.shape[0])

        # Draw points and lines
        prev_point = None
        for i, (x, y) in enumerate(valid_pixels):
            # Calculate point size and color based on how far in the future it is
            future_idx = future_indices[i]
            radius = max(4, int(5 + future_idx / 2))
            alpha = 0.7 + 0.3 * (future_idx / traj.shape[0])  # Increasing opacity

            # Interpolate color from green to red based on future time
            green = int(255 * (1 - future_idx / traj.shape[0]))
            red = int(255 * (future_idx / traj.shape[0]))
            color = (0, green, red)

            # Draw circle
            cv2.circle(img_with_traj, (x, y), radius, color, -1)

            # Connect with line to previous point
            if prev_point is not None:
                cv2.line(img_with_traj, prev_point, (x, y), (0, 255, 0), 2)
            prev_point = (x, y)

        return img_with_traj


def collect_clip(dataset, seq_len=20, min_disp=0.01):
    """
    Return a list of (img_front, traj_xyz) tuples for `seq_len`
    *consecutive* samples.

    seq_len   - number of successive NuScenes samples to grab
    """
    clip = []
    for i in range(seq_len):
        print(f"[collecting] {i}/{seq_len}", end="\r")
        sensor_data, traj = dataset[i]
        img = sensor_data["CAM_FRONT"]["img"].squeeze()
        clip.append((img, traj.squeeze()))
    return clip


def build_anim(fig, ax_cam, ax_top, img, traj_xyz, fps=3, tail=True):
    """
    Returns a fully-configured matplotlib FuncAnimation object.
    """
    # ── static camera image ────────────────────────────────────────────────────
    ax_cam.imshow(img)
    ax_cam.set_title("Front Camera View")
    ax_cam.axis("off")

    # ── pre-compute data we need every frame ───────────────────────────────────
    x_all, y_all = traj_xyz[:, 0], traj_xyz[:, 1]
    n_frames = len(x_all)
    cmap_vals = np.linspace(0, 1, n_frames)  # 0→early  1→late
    colors_rgba = plt.cm.viridis(cmap_vals)

    # ── artists we’ll mutate per frame ─────────────────────────────────────────
    scat = ax_top.scatter([], [], s=[], c=[], cmap="viridis", edgecolors="w")
    (line,) = ax_top.plot([], [], lw=2, color="lime")

    ax_top.set_aspect("equal")
    ax_top.grid(True)
    ax_top.set_title("Trajectory Waypoints (Top-Down View)")
    ax_top.set_xlabel("X position (m)")
    ax_top.set_ylabel("Y position (m)")

    # ⬇️ ADD - fix invisible points by pre-setting limits
    margin = 2.0
    ax_top.set_xlim(x_all.min() - margin, x_all.max() + margin)
    ax_top.set_ylim(y_all.min() - margin, y_all.max() + margin)

    # ── init + update callbacks for FuncAnimation ──────────────────────────────
    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_sizes([])
        scat.set_array([])
        line.set_data([], [])
        return scat, line

    def update(frame):
        # indices of points we want visible this frame
        idx = np.arange(frame + 1)

        # update scatter
        offsets = np.column_stack([x_all[idx], y_all[idx]])
        scat.set_offsets(offsets)
        scat.set_array(cmap_vals[idx])

        # size or alpha for fading “tail”
        if tail:
            sizes = np.linspace(40, 120, len(idx))  # older→small, newer→big
            scat.set_sizes(sizes)
        else:
            scat.set_sizes(np.full(len(idx), 100))

        # update connecting poly-line
        line.set_data(x_all[idx], y_all[idx])
        return scat, line

    # ── build animation object ────────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=1000 / fps, blit=True
    )
    return anim


def main(
    dataset_root="/scratch/gilbreth/mgagvani/nuscenes",
    seq_len=400,
    fps=5,
    outfile="trajectory_timelapse.mp4",
):
    # 0. ── dataset -----------------------------------------------------------
    ds = NuScenesDataset(
        dataset_root, version="v1.0-mini", future_seconds=3.0, future_hz=2.0
    )

    # 1. ── fetch a *moving* contiguous clip ----------------------------------
    clip = collect_clip(ds, seq_len=seq_len, min_disp=-1)  # list of tuples

    # 2. ── figure & axis boilerplate ----------------------------------------
    fig, (ax_cam, ax_traj) = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"wspace": 0.3}
    )

    # dummy artists, overwritten each frame
    img_artist = ax_cam.imshow(clip[0][0])
    ax_cam.set_title("Front Camera View")
    ax_cam.axis("off")

    scat = ax_traj.scatter([], [], c=[], s=[], cmap="viridis", edgecolors="w")
    (line,) = ax_traj.plot([], [], lw=2, color="lime")
    ax_traj.set_title("Trajectory Waypoints (Top-Down View)")
    ax_traj.set_xlabel("X position (m)")
    ax_traj.set_ylabel("Y position (m)")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True)

    # axis limits based on *entire* clip, so they stay stable
    xs = np.concatenate([t[1][:, 0] for t in clip])
    ys = np.concatenate([t[1][:, 1] for t in clip])
    margin = 2.0
    ax_traj.set_xlim(xs.min() - margin, xs.max() + margin)
    ax_traj.set_ylim(ys.min() - margin, ys.max() + margin)

    # 3. ── animation callbacks ----------------------------------------------
    cmap_vals = plt.cm.viridis(np.linspace(0, 1, clip[0][1].shape[0]))

    def update(frame_idx):
        print(f"[animating] frame {frame_idx}", end="\r")
        img, traj = clip[frame_idx]
        # camera panel
        img_artist.set_data(img)
        # trajectory panel - overwrite with *full* future traj for this sample
        offsets = traj[:, :2]
        scat.set_offsets(offsets)
        scat.set_array(np.linspace(0, 1, len(offsets)))  # colour by step
        scat.set_sizes(np.full(len(offsets), 90))
        line.set_data(offsets[:, 0], offsets[:, 1])
        return img_artist, scat, line

    anim = FuncAnimation(fig, update, frames=len(clip), interval=1000 / fps, blit=True)

    # 4. ── save --------------------------------------------------------------
    print(f"[saving] {outfile}")
    anim.save(outfile, writer="ffmpeg", dpi=120, fps=fps)
    plt.close(fig)
    print("Done ✔ Open the MP4 to watch the timelapse.")


if __name__ == "__main__":
    main()
