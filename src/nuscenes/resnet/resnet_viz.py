"""policy_viz.py
Visualise predictions of an E2E policy on a NuScenes sample and export an MP4
animation.

Layout
------
* Top two rows: the six camera views (animated over time).
* Bottom row : top‑down view showing complete GT and prediction trajectories
  for each time-step.

Usage
-----
python policy_viz.py \
    --ckpt resnet_mlp.ckpt \
    --nusc_root /data/nuscenes \
    --sample 42 \
    --nframes 10 \
    --out viz.mp4
"""
from __future__ import annotations
import argparse, os, sys
from typing import Sequence, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter
import torch
import numpy as np
from tqdm import tqdm

# project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simple_resnet import LitResNetMLP, ResNetMLPBaseline, preprocess_img  # type: ignore
from nuscenes_dataset import NuScenesDataset  # frame‑level dataset wrapper

CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

###############################################################################
#                               Helper functions                              #
###############################################################################

def load_sample_sequence(ds: NuScenesDataset, start_idx: int, hw: Tuple[int, int], 
                        nframes: int):
    """Load a sequence of samples for animation."""
    samples_data = []
    
    for i in range(nframes):
        try:
            sample_idx = start_idx + i
            if sample_idx >= len(ds):
                print(f"Warning: Sample {sample_idx} beyond dataset size, stopping at {len(samples_data)} frames")
                break
                
            sample = ds[sample_idx]
            
            # Extract data for this sample
            sensor_data = sample["sensor_data"]
            traj_gt = torch.tensor(sample["trajectory"][:, :2])
            cmd = torch.from_numpy(sample["command"].astype("float32").squeeze()).unsqueeze(0)
            img_np = sensor_data["CAM_FRONT"]["img"]
            img_t = preprocess_img(img_np, hw).unsqueeze(0)
            
            samples_data.append({
                'sensor_data': sensor_data,
                'traj_gt': traj_gt,
                'cmd': cmd,
                'img_t': img_t,
                'sample_idx': sample_idx
            })
            
        except (IndexError, KeyError) as e:
            print(f"Error loading sample {sample_idx}: {e}")
            break
    
    return samples_data


def predict_waypoints(ckpt: str, img: torch.Tensor, cmd: torch.Tensor):
    dummy = ResNetMLPBaseline(
        model_name="resnet34.a1_in1k",
        pretrained=True,
        cmd_dim=3,
        hidden_dim=512,
        num_waypoints=3,
    )
    lit = LitResNetMLP.load_from_checkpoint(ckpt, model=dummy, map_location="cpu")
    lit.eval()
    with torch.no_grad():
        pred = lit(img, cmd).squeeze(0).cpu()
    return pred  # (N,2)


def make_animation(samples_data: List[Dict[str, Any]], ckpt: str, cams: Sequence[str], 
                   outfile: str, fps: int = 2):
    """Save an MP4 showing camera views and trajectories over time."""
    nframes = len(samples_data)
    
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.2)

    # Create camera subplot axes
    cam_axes = {}
    cam_images = {}
    
    for i, cam in enumerate(cams):
        if cam not in samples_data[0]['sensor_data']:
            continue  # skip missing cameras
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_title(cam.replace("CAM_", ""), fontsize=10)
        ax.axis("off")
        
        # Initialize with first frame
        img_data = samples_data[0]['sensor_data'][cam]["img"]
        im = ax.imshow(img_data)
        cam_axes[cam] = ax
        cam_images[cam] = im

    # top‑down axis
    ax_td = fig.add_subplot(gs[2, :])
    ax_td.set_xlabel("x [m]")
    ax_td.set_ylabel("y [m]")
    ax_td.set_title("Trajectory Predictions Over Time")
    ax_td.grid(True, alpha=0.3)
    
    # Calculate axis limits from all trajectories
    all_x_coords = []
    all_y_coords = []
    
    for sample_data in samples_data:
        gt = sample_data['traj_gt']
        all_x_coords.extend(gt[:, 0].tolist())
        all_y_coords.extend(gt[:, 1].tolist())
    
    # Add some margin
    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)
    
    x_margin = max((x_max - x_min) * 0.1, 2.0)  # At least 2m margin
    y_margin = max((y_max - y_min) * 0.1, 2.0)
    
    ax_td.set_xlim(x_min - x_margin, x_max + x_margin)
    ax_td.set_ylim(y_min - y_margin, y_max + y_margin)
    ax_td.set_aspect('equal', adjustable='box')

    # Initialize trajectory lines
    gt_line, = ax_td.plot([], [], "g-o", linewidth=3, markersize=8, label="Ground Truth", alpha=0.8)
    pred_line, = ax_td.plot([], [], "r-x", linewidth=3, markersize=10, label="Prediction", alpha=0.8)
    
    # Add ego vehicle marker at origin
    ego_marker, = ax_td.plot([0], [0], 'ko', markersize=12, label="Ego Vehicle")
    ax_td.legend(loc='upper right')
    
    # Add frame counter
    frame_text = ax_td.text(0.02, 0.98, '', transform=ax_td.transAxes, 
                           verticalalignment='top', fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame_idx: int):
        """Update function for animation frame."""
        sample_data = samples_data[frame_idx]
        
        # Update camera views
        for cam in cam_images:
            if cam in sample_data['sensor_data']:
                new_img = sample_data['sensor_data'][cam]["img"]
                cam_images[cam].set_array(new_img)
        
        # Get predictions for this frame
        pred_wp = predict_waypoints(ckpt, sample_data['img_t'], sample_data['cmd'])
        gt = sample_data['traj_gt']
        
        # Update complete trajectories for this time-step
        gt_line.set_data(gt[:, 0], gt[:, 1])
        pred_line.set_data(pred_wp[:, 0], pred_wp[:, 1])
        
        # Update frame counter
        frame_text.set_text(f'Time-step: {frame_idx+1}/{nframes}\nSample: {sample_data["sample_idx"]}')
        
        return [gt_line, pred_line, frame_text, ego_marker] + list(cam_images.values())

    # Create animation
    print(f"Creating animation with {nframes} time-steps at {fps} FPS...")
    print(f"Each frame shows complete trajectories for that time-step")
    
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, outfile, dpi=120):
        for f in tqdm(range(nframes), desc="Rendering frames"):
            update(f)
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Animation saved to {outfile}")
    print(f"Duration: {nframes / fps:.1f} seconds")

###############################################################################
#                                   CLI                                       #
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--nusc_root", required=True, help="Path to NuScenes dataset")
    parser.add_argument("--sample", type=int, default=0, help="Starting sample index in dataset")
    parser.add_argument("--nframes", type=int, default=10, help="Number of time-steps to animate")
    parser.add_argument("--hw", type=int, nargs=2, default=[224, 224], help="Image height and width")
    parser.add_argument("--out", default="policy_viz.mp4", help="Output video file")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    args = parser.parse_args()

    print(f"Loading NuScenes dataset from {args.nusc_root}")
    ds = NuScenesDataset(
        nuscenes_path=args.nusc_root,
        version="v1.0-test",
        future_seconds=3,
        future_hz=1,
    )
    print(f"Dataset loaded with {len(ds)} samples")

    print(f"Loading {args.nframes} samples starting from index {args.sample}")
    samples_data = load_sample_sequence(ds, args.sample, tuple(args.hw), args.nframes)
    
    if len(samples_data) == 0:
        print("Error: No samples loaded!")
        return
    
    print(f"Successfully loaded {len(samples_data)} samples")
    print(f"Sample trajectory shapes: GT={samples_data[0]['traj_gt'].shape}")

    make_animation(
        samples_data,
        args.ckpt,
        CAMERAS,
        outfile=args.out,
        fps=args.fps,
    )

if __name__ == "__main__":
    main()