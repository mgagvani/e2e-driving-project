"""mae_visualize_recon_fixed.py
--------------------------------------------------------------------
Improved visualisation for **Masked Auto‑Encoder** outputs.
The first version plotted the raw decoder prediction for *all* patches,
so visible‑patch predictions looked noisy / banded.  Here we instead
swap‑in the MAE's prediction **only at masked locations** and keep the
original RGB for un‑masked patches – the expected qualitative view in
MAE papers.

Grid layout
===========
1. **Original** – untouched RGB frame
2. **Masked Input** – grey squares over hidden patches
3. **Reconstruction (masked‑only)** – original image with decoder predictions 
   pasted *only into the masked squares*, giving a clearer sense of how well 
   the model in‑paints.
4. **Error heatmap** – absolute pixel error where mask==1 (optional, off
   by default).

Usage (same flags as before)
---------------------------
```bash
python mae_visualize_recon_fixed.py \
       --ckpt mae_poc_best.ckpt --nusc_root /data/nuscenes --num 12 \
       --show_error
```
"""

from __future__ import annotations
import argparse, os
from pathlib import Path

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from mae_poc import LitMAE, CollateFrontCam
from nuscenes_dataset import NuScenesDataset

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def unpatchify(patches: torch.Tensor, patch_size: int, img_size: int) -> torch.Tensor:
    """Convert patches back to image format."""
    P = patch_size; C = 3; G = img_size // P
    patches = patches.view(G, G, P, P, C).permute(4, 0, 2, 1, 3).contiguous()
    return patches.view(C, img_size, img_size)


def patchify(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert image to patches."""
    C, H, W = img.shape
    P = patch_size
    G = H // P
    patches = img.view(C, G, P, G, P).permute(1, 3, 2, 4, 0).contiguous()
    return patches.view(G * G, P * P * C)


def paste_pred_into_masked_regions(orig: torch.Tensor, pred_patches: torch.Tensor, 
                                 mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Return an RGB image where:
    - Visible patches (mask=0) show the original image
    - Masked patches (mask=1) show the model's reconstruction
    
    This is the standard MAE visualization approach.
    """
    P = patch_size
    G = orig.shape[-1] // P
    out = orig.clone()
    
    # Convert predictions back to image format
    pred_img = unpatchify(pred_patches, P, orig.shape[-1])
    
    # Reshape mask to grid
    mask_grid = mask.view(G, G)
    
    # Replace only the masked patches with predictions
    for i in range(G):
        for j in range(G):
            if mask_grid[i, j] == 1:  # This patch was masked
                out[:, i*P:(i+1)*P, j*P:(j+1)*P] = pred_img[:, i*P:(i+1)*P, j*P:(j+1)*P]
    
    return out


def apply_mask_visualization(img: torch.Tensor, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Apply grey mask to show which patches were hidden from the encoder."""
    img = img.clone()
    G = img.shape[-1] // patch_size
    mask = mask.view(G, G)
    grey = torch.tensor([0.5, 0.5, 0.5], dtype=img.dtype, device=img.device)[:, None, None]
    
    for i in range(G):
        for j in range(G):
            if mask[i, j] == 1:
                img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = grey
    
    return img


def compute_reconstruction_error(orig: torch.Tensor, pred_patches: torch.Tensor, 
                               mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Compute pixel-wise error only in masked regions."""
    # Get original patches
    orig_patches = patchify(orig, patch_size)
    
    # Compute error per patch
    patch_errors = (pred_patches - orig_patches).abs().mean(dim=1)  # Average over patch pixels
    
    # Create error image
    G = orig.shape[-1] // patch_size
    error_img = torch.zeros_like(orig[0])  # Single channel
    
    mask_grid = mask.view(G, G)
    error_grid = patch_errors.view(G, G)
    
    for i in range(G):
        for j in range(G):
            if mask_grid[i, j] == 1:  # Only show error for masked patches
                error_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = error_grid[i, j]
    
    return error_img

from mae_poc import CollateFrontCam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--out_dir", default="recon_vis_fixed")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--show_error", action="store_true")
    ap.add_argument("--mask_ratio", type=float, default=0.75,)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: LitMAE = LitMAE.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    mae = model
    mae.mask_ratio = args.mask_ratio

    ds = NuScenesDataset(args.nusc_root, split="val", version="v1.0-trainval",
                         future_seconds=3, future_hz=1)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4,
                    collate_fn=CollateFrontCam(cam=args.cam))

    shown = 0
    for idx, imgs in enumerate(dl):
        if shown >= args.num:
            break
            
        imgs = imgs.to(device)
        
        with torch.no_grad():
            losses, pred, mask = mae(imgs)

        P, H = mae.patch_size, mae.img_size

        # debug 1
        full = unpatchify(pred[0].cpu(), P, H)   
        plt.figure()
        plt.imshow(full.clamp(0,1).permute(1,2,0))
        plt.title(f"Full prediction for sample {idx:04d}")
        plt.axis("off")
        plt.savefig(out_dir / f"full_pred_{idx:04d}.png", dpi=150, bbox_inches='tight')
        
        # Get first sample from batch
        img = imgs[0].cpu()
        mask0 = mask[0].cpu()
        pred0 = pred[0].cpu()
        
        
        
        # Create visualizations
        masked_input = apply_mask_visualization(img, mask0, P)
        reconstruction = paste_pred_into_masked_regions(img, pred0, mask0, P)
        
        # Setup plot
        ncols = 4 if args.show_error else 3
        fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
        
        panels = [img, masked_input, reconstruction]
        titles = ["Original", "Masked Input", "Reconstruction (masked regions only)"]
        
        # Optional error heatmap
        if args.show_error:
            error_map = compute_reconstruction_error(img, pred0, mask0, P)
            import matplotlib.cm as cm
            error_normalized = error_map.numpy()
            if error_normalized.max() > 0:
                error_normalized = error_normalized / error_normalized.max()
            error_colored = cm.jet(error_normalized)[:, :, :3].transpose(2, 0, 1)
            panels.append(torch.tensor(error_colored))
            titles.append("Error (masked regions)")
        
        # Plot all panels
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel.permute(1, 2, 0).clamp(0, 1))
            ax.set_title(title)
            ax.axis("off")
        
        total_loss, lpips_loss, mse_loss, l1_loss = losses
        fig.suptitle(
            f"Sample {idx:04d} | Total Loss: {total_loss.item():.4f} "
            f"| LPIPS: {lpips_loss.item():.4f} | MSE: {mse_loss.item():.4f} | L1: {l1_loss.item():.4f} "
            f"| Mask ratio: {mask0.float().mean():.2f}"
        )
        fig.tight_layout()
        
        # Save
        fig.savefig(out_dir / f"sample_{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        shown += 1

    print(f"Wrote {shown} visualization images to {out_dir}")


if __name__ == "__main__":
    main()