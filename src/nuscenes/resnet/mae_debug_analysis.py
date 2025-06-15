"""mae_debug_analysis.py
--------------------------------------------------------------------
Debug script to diagnose MAE "gray sludge" prediction issues.
This script will help identify common problems like normalization 
mismatches, untrained models, or configuration issues.
"""

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from mae_poc import LitMAE, CollateFrontCam
from nuscenes_dataset import NuScenesDataset


def analyze_model_weights(model):
    """Check if model has been properly trained."""
    print("=== MODEL WEIGHT ANALYSIS ===")
    
    # Check if weights are initialized or trained
    decoder_weights = []
    encoder_weights = []
    
    for name, param in model.named_parameters():
        if 'decoder' in name and param.requires_grad:
            decoder_weights.append(param.data.flatten())
        elif 'encoder' in name and param.requires_grad:
            encoder_weights.append(param.data.flatten())
    
    if decoder_weights:
        decoder_tensor = torch.cat(decoder_weights)
        print(f"Decoder weights - Mean: {decoder_tensor.mean():.6f}, Std: {decoder_tensor.std():.6f}")
        print(f"Decoder weights - Min: {decoder_tensor.min():.6f}, Max: {decoder_tensor.max():.6f}")
        
        # Check if weights look like random initialization
        if decoder_tensor.std() < 0.01:
            print("⚠️  WARNING: Decoder weights have very low variance - might be untrained!")
    
    if encoder_weights:
        encoder_tensor = torch.cat(encoder_weights)
        print(f"Encoder weights - Mean: {encoder_tensor.mean():.6f}, Std: {encoder_tensor.std():.6f}")
        print(f"Encoder weights - Min: {encoder_tensor.min():.6f}, Max: {encoder_tensor.max():.6f}")


def analyze_data_statistics(dataloader, num_samples=5):
    """Analyze input data statistics."""
    print("\n=== INPUT DATA ANALYSIS ===")
    
    pixel_values = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        pixel_values.append(batch.flatten())
    
    if pixel_values:
        all_pixels = torch.cat(pixel_values)
        print(f"Input data - Mean: {all_pixels.mean():.6f}, Std: {all_pixels.std():.6f}")
        print(f"Input data - Min: {all_pixels.min():.6f}, Max: {all_pixels.max():.6f}")
        
        # Check if data is normalized
        if all_pixels.min() >= -3 and all_pixels.max() <= 3:
            print("✓ Data appears to be normalized (range roughly [-3, 3])")
        elif all_pixels.min() >= 0 and all_pixels.max() <= 1:
            print("✓ Data appears to be in [0, 1] range")
        elif all_pixels.min() >= 0 and all_pixels.max() <= 255:
            print("⚠️  WARNING: Data appears to be in [0, 255] range - might need normalization!")
        else:
            print(f"⚠️  WARNING: Unusual data range: [{all_pixels.min():.3f}, {all_pixels.max():.3f}]")


def analyze_model_outputs(model, dataloader, device):
    """Analyze model prediction statistics."""
    print("\n=== MODEL OUTPUT ANALYSIS ===")
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader)).to(device)
        loss, pred, mask = model.net(batch)
        
        print(f"Loss: {loss.item():.6f}")
        print(f"Prediction shape: {pred.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask ratio: {mask.float().mean():.3f}")
        
        # Analyze predictions
        pred_flat = pred.flatten()
        print(f"Predictions - Mean: {pred_flat.mean():.6f}, Std: {pred_flat.std():.6f}")
        print(f"Predictions - Min: {pred_flat.min():.6f}, Max: {pred_flat.max():.6f}")
        
        # Check if predictions are collapsed
        if pred_flat.std() < 0.01:
            print("🚨 CRITICAL: Predictions have very low variance - model is predicting constant values!")
        
        # Check prediction distribution
        pred_np = pred_flat.cpu().numpy()
        percentiles = np.percentile(pred_np, [1, 5, 25, 50, 75, 95, 99])
        print(f"Prediction percentiles: {percentiles}")


def test_different_normalizations(model, raw_image, device):
    """Test model with different normalization schemes."""
    print("\n=== NORMALIZATION TESTING ===")
    
    model.eval()
    
    # Test different normalizations
    normalizations = {
        "None (0-1)": lambda x: x,
        "ImageNet": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        "Zero-centered": T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        "Custom (-1,1)": lambda x: (x - 0.5) * 2,
    }
    
    # Ensure raw image is in [0,1] range
    if raw_image.max() > 1:
        raw_image = raw_image / 255.0
    
    results = {}
    
    for name, norm_fn in normalizations.items():
        try:
            normalized = norm_fn(raw_image.clone())
            batch = normalized.unsqueeze(0).to(device)
            
            with torch.no_grad():
                loss, pred, mask = model.net(batch)
                pred_stats = {
                    'mean': pred.mean().item(),
                    'std': pred.std().item(),
                    'loss': loss.item()
                }
                results[name] = pred_stats
                print(f"{name:12} - Loss: {loss.item():.4f}, Pred mean: {pred.mean().item():.4f}, Pred std: {pred.std().item():.4f}")
        
        except Exception as e:
            print(f"{name:12} - Error: {e}")
            results[name] = None
    
    return results


def create_debug_visualization(model, dataloader, device, out_dir="debug_output"):
    """Create debug visualizations."""
    print("\n=== CREATING DEBUG VISUALIZATIONS ===")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    model.eval()
    batch = next(iter(dataloader)).to(device)
    
    with torch.no_grad():
        loss, pred, mask = model.net(batch)
    
    # Get first sample
    img = batch[0].cpu()
    pred0 = pred[0].cpu()
    mask0 = mask[0].cpu()
    
    # Create various visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Raw input (denormalized if needed)
    img_vis = img.clone()
    if img_vis.min() < 0:  # Likely normalized
        # Try to denormalize (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img_vis = img_vis * std + mean
    
    axes[0,0].imshow(img_vis.permute(1,2,0).clamp(0,1))
    axes[0,0].set_title("Input (denormalized)")
    axes[0,0].axis('off')
    
    # Raw input (as-is)
    axes[0,1].imshow(img.permute(1,2,0).clamp(0,1))
    axes[0,1].set_title("Input (raw)")
    axes[0,1].axis('off')
    
    # Prediction statistics
    pred_reshaped = pred0.view(-1, 3)  # Reshape to [num_patches, 3*patch_size^2]
    axes[0,2].hist(pred_reshaped.flatten().numpy(), bins=50, alpha=0.7)
    axes[0,2].set_title("Prediction Histogram")
    axes[0,2].set_xlabel("Pixel Value")
    
    # Mask visualization
    P = model.net.patch_size
    G = img.shape[-1] // P
    mask_img = mask0.view(G, G).repeat_interleave(P, 0).repeat_interleave(P, 1)
    axes[1,0].imshow(mask_img, cmap='gray')
    axes[1,0].set_title(f"Mask (ratio: {mask0.float().mean():.2f})")
    axes[1,0].axis('off')
    
    # Prediction patches (first few)
    n_show = min(9, pred0.shape[0])
    patch_grid = pred0[:n_show].view(n_show, P, P, 3)
    grid_size = int(np.ceil(np.sqrt(n_show)))
    combined = torch.zeros(grid_size * P, grid_size * P, 3)
    
    for i in range(n_show):
        row, col = i // grid_size, i % grid_size
        combined[row*P:(row+1)*P, col*P:(col+1)*P] = patch_grid[i]
    
    axes[1,1].imshow(combined.clamp(0,1))
    axes[1,1].set_title(f"First {n_show} Prediction Patches")
    axes[1,1].axis('off')
    
    # Loss over patches
    if hasattr(model.net, 'patchify'):
        # get target patches and move to CPU to match pred0
        target_patches = model.net.patchify(batch)[0].cpu()
        patch_losses = ((pred0 - target_patches) ** 2).mean(dim=1)
        masked_losses = patch_losses[mask0.bool()]
        
        axes[1,2].hist(masked_losses.numpy(), bins=20, alpha=0.7)
        axes[1,2].set_title("Per-patch Loss (masked)")
        axes[1,2].set_xlabel("MSE Loss")
    else:
        axes[1,2].text(0.5, 0.5, "Cannot compute\nper-patch loss", 
                      ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title("Per-patch Loss")
    
    plt.tight_layout()
    plt.savefig(out_dir / "debug_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved to {out_dir}/debug_analysis.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--nusc_root", required=True, help="NuScenes dataset root")
    parser.add_argument("--cam", default="CAM_FRONT", help="Camera to use")
    args = parser.parse_args()
    
    print("🔍 MAE Debug Analysis Starting...")
    print("=" * 50)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = LitMAE.load_from_checkpoint(args.ckpt, map_location=device)
        model.eval().to(device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load dataset
    try:
        ds = NuScenesDataset(args.nusc_root, split="val", version="v1.0-trainval",
                           future_seconds=3, future_hz=1)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                       collate_fn=CollateFrontCam(cam=args.cam))
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Run analyses
    analyze_model_weights(model.net)
    analyze_data_statistics(dl)
    analyze_model_outputs(model, dl, device)
    
    # Test normalization
    sample_batch = next(iter(dl))
    test_different_normalizations(model, sample_batch[0], device)
    
    # Create debug visualizations
    create_debug_visualization(model, dl, device)
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG RECOMMENDATIONS:")
    print("1. Check the 'Model Output Analysis' - if predictions have very low std, the model isn't trained")
    print("2. Check the 'Normalization Testing' - find which normalization gives reasonable loss")
    print("3. Look at debug_analysis.png for visual inspection")
    print("4. If loss is very high (>1.0), check training configuration")
    print("5. If predictions are constant, model likely needs more training")


if __name__ == "__main__":
    main()