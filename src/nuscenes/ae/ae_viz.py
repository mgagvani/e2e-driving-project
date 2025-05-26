"""ae_viz.py – Standalone visualization script for the β‑VAE
================================================================
Imports the model + dataset wrappers from `ae.py` (same folder) and
produces a side‑by‑side grid of original vs. reconstructed frames.

Usage
-----
```bash
python ae_viz.py \
  --ckpt runs/vae/last.ckpt \
  --nusc_root /data/nuscenes \
  --cam CAM_FRONT \
  --seq_len 4 --stride 1 \
  --hw 180 320 \
  --sample_idx 0
```
The plot is saved as **`recon_grid.png`** in the current directory.
"""
from __future__ import annotations
import argparse, os, sys, torch, matplotlib.pyplot as plt

# Make sure we can import from sibling file `ae.py`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from ae import SpatioTemporalVAE, LitVAE, SequenceDataset, preprocess_rgb  # type: ignore

sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
try:
    from nuscenes_dataset import NuScenesDataset  # your frame‑level dataset
except ImportError:
    raise SystemExit("Cannot import NuScenesDataset – make sure it is on PYTHONPATH.")

###############################################################################
#                    Utility: show original vs. reconstructed                 #
###############################################################################

def show_reconstruction(model: SpatioTemporalVAE, clip: torch.Tensor, save_path="recon_grid.png"):
    """Save a 2×T grid comparing original and reconstruction."""
    model.eval()
    with torch.no_grad():
        _, _, _, recon = model(clip.unsqueeze(0))
    recon = recon.squeeze(0).cpu()
    clip = clip.cpu()
    T = clip.shape[1]

    fig, axes = plt.subplots(2, T, figsize=(T * 2.5, 5))
    for t in range(T):
        for row, tensor in enumerate([clip, recon]):
            ax = axes[row, t]
            img = tensor[:, t].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t}")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

###############################################################################
#                                   CLI                                       #
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Visualize VAE reconstructions")
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--nusc_root", required=True)
    p.add_argument("--cam", default="CAM_FRONT")
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hw", type=int, nargs=2, default=[80, 160])
    p.add_argument("--sample_idx", type=int, default=0, help="Which clip index to visualise")
    args = p.parse_args()

    # ---------------- Load dataset ---------------
    frame_ds = NuScenesDataset(args.nusc_root, version="v1.0-trainval",)
    clip_ds = SequenceDataset(frame_ds, seq_len=args.seq_len, stride=args.stride,
                              cam=args.cam, hw=tuple(args.hw))
    clip, _ = clip_ds[args.sample_idx]

    # ---------------- Restore model --------------
    dummy_vae = SpatioTemporalVAE(video_shape=(args.seq_len, *args.hw))
    lit = LitVAE.load_from_checkpoint(args.ckpt, vae=dummy_vae, map_location="cpu")
    vae: SpatioTemporalVAE = lit.model

    # ensure resolution matches expectation
    if clip.shape[1:] != torch.Size(vae.video_shape):
        raise ValueError(f"Clip shape {clip.shape[1:]} ≠ VAE.video_shape {vae.video_shape}.")

    show_reconstruction(vae, clip, save_path="recon_grid.png")

if __name__ == "__main__":
    main()
