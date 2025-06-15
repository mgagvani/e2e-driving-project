import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import torchvision.transforms as T
from torchvision.transforms.functional import resize as tv_resize

# Use a proper vision encoder instead of placeholder
class VisionEncoder(nn.Module):
    """Vision encoder using a pre-trained model from timm."""
    
    def __init__(self, model_name="resnet18", pretrained=True, c_f1=128, h_f1=14, w_f1=14):
        super().__init__()
        # Use timm model as backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            global_pool=""  # Remove global pooling to keep spatial dimensions
        )

        # freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get the feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_out = self.backbone(dummy_input)
            backbone_channels = backbone_out.shape[1]
        
        # Add projection layer to get desired output dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(backbone_channels, c_f1, kernel_size=1),
            nn.AdaptiveAvgPool2d((h_f1, w_f1))
        )
        
    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)  # (B, backbone_channels, H', W')
        return self.projection(features)  # (B, c_f1, h_f1, w_f1)

class FEMoE_Expert(nn.Module):
    def __init__(self, c_in_f1, c_out_f2_expert, h_f1, w_f1, h_f2_expert_target, w_f2_expert_target):
        super().__init__()
        # Calculate stride and kernel size for deconv to achieve target dimensions
        stride_h = h_f2_expert_target // h_f1
        stride_w = w_f2_expert_target // w_f1
        
        # Use stride=2 for doubling dimensions (common case)
        if stride_h == stride_w == 2:
            kernel_size = 4
            stride = 2
            padding = 1
        else:
            # Generic case - use stride = stride_h (assuming square)
            kernel_size = stride_h + 1
            stride = stride_h
            padding = 0
            
        self.deconv = nn.ConvTranspose2d(
            c_in_f1, c_out_f2_expert, 
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(c_out_f2_expert, c_out_f2_expert, kernel_size=3, padding=1)
        
        self.h_f2_expert_target = h_f2_expert_target
        self.w_f2_expert_target = w_f2_expert_target

    def forward(self, x_f1):
        # x_f1: (B, c_in_f1, h_f1, w_f1)
        x_f2 = self.deconv(x_f1)
        
        # Ensure exact target dimensions with interpolation if needed
        if x_f2.shape[2] != self.h_f2_expert_target or x_f2.shape[3] != self.w_f2_expert_target:
            x_f2 = F.interpolate(
                x_f2, 
                size=(self.h_f2_expert_target, self.w_f2_expert_target), 
                mode='bilinear', 
                align_corners=False
            )
        
        x_f2 = self.relu(x_f2)
        x_f2 = self.conv(x_f2)
        return x_f2

# Spatial gate network for multi-camera features
class SpatialGateNetwork(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        # Channel attention branch - squeeze and excitation style
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention branch - learn spatial importance
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final compression layer
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def forward(self, x):
        # x: (B, C, H, W) input feature map
        batch_size, channels, height, width = x.shape
        
        # Channel-wise gating
        channel_weights = self.channel_gate(x)  # (B, C)
        channel_weights = channel_weights.view(batch_size, channels, 1, 1)
        
        # Spatial gating
        spatial_weights = self.spatial_gate(x)  # (B, 1, H, W)
        
        # Apply both gates
        gated_features = x * channel_weights * spatial_weights
        
        # Compress to feature vector
        compressed_output = self.compress(gated_features)  # (B, C)
        
        return compressed_output

class MiniDriveE2E(pl.LightningModule):
    def __init__(self,
                 num_cameras=6,
                 # Vision Encoder HParams
                 vision_encoder_name="resnet18",
                 c_f1=128, h_f1=14, w_f1=14,

                 # FE-MoE HParams
                 num_experts=4,
                 c_f2_expert=64,
                 h_f2_expert=28,
                 w_f2_expert=28,

                 # Waypoint Head HParams
                 num_waypoints=3,
                 waypoint_dims=2,
                 mlp_hidden_dim=512,
                 learning_rate=1e-4,
                 weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()

        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_encoder_name,
            pretrained=True,
            c_f1=c_f1,
            h_f1=h_f1,
            w_f1=w_f1
        )
        # Freeze vision encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False        

        # 2. FE-MoE Gate Network
        gate_cnn_out_h = h_f1 // 2
        gate_cnn_out_w = w_f1 // 2
        
        self.femoe_gate_cnn_part = nn.Sequential(
            nn.Conv2d(c_f1, c_f1 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c_f1 // 2, c_f1 // 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.femoe_gate_mlp_part = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_f1 // 4, num_experts)
        )

        # 3. FE-MoE Experts
        self.femoe_experts = nn.ModuleList([
            FEMoE_Expert(c_f1, c_f2_expert, h_f1, w_f1, h_f2_expert, w_f2_expert)
            for _ in range(num_experts)
        ])

        # 4. Spatial Gate Networks for each camera's MoE output
        # One gate network per camera, operating on c_f2_expert channels
        self.spatial_gates = nn.ModuleList([
            SpatialGateNetwork(c_f2_expert) for _ in range(num_cameras)
        ])

        # 5. Waypoint Prediction Head
        # Input dimension is num_cameras * c_f2_expert (concatenated spatially compressed features)
        mlp_input_dim_aggregated = num_cameras * c_f2_expert
        
        self.waypoint_head = nn.Sequential(
            nn.Linear(mlp_input_dim_aggregated, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # You might want to tune this dropout
            nn.Linear(mlp_hidden_dim, num_waypoints * waypoint_dims)
        )

    # --------------- metrics ---------------
    @staticmethod
    def l2_error(pred: torch.Tensor, gt: torch.Tensor):  # both (B,N,2)
        return torch.sqrt(
            ((pred - gt) ** 2).sum(dim=-1)
        ).mean()  # mean over pts & batch

    @staticmethod
    def collision_rate(pred: torch.Tensor) -> torch.Tensor:
        """Dummy placeholder - returns 0."""
        return torch.tensor(0.0, device=pred.device)

    def forward(self, multi_camera_images):
        # multi_camera_images: (B, num_cameras, 3, H_img, W_img)
        batch_size = multi_camera_images.size(0)
        num_cameras = self.hparams.num_cameras

        # Reshape to process all camera images together
        images_reshaped = multi_camera_images.view(
            batch_size * num_cameras,
            3,
            multi_camera_images.size(3),
            multi_camera_images.size(4)
        )

        # 1. Vision Encoder for all images
        self.vision_encoder.eval() # Ensure backbone is in eval mode if frozen
        with torch.no_grad(): # Explicitly no_grad for frozen backbone
            f1_features_all_cams = self.vision_encoder(images_reshaped)
        # f1_features_all_cams: (B * num_cameras, c_f1, h_f1, w_f1)

        # 2. FE-MoE Gate computation
        gate_cnn_out_all_cams = self.femoe_gate_cnn_part(f1_features_all_cams)
        gate_logits_all_cams = self.femoe_gate_mlp_part(gate_cnn_out_all_cams)
        gate_weights_all_cams = F.softmax(gate_logits_all_cams, dim=-1)
        # gate_weights_all_cams: (B * num_cameras, num_experts)

        # 3. Apply experts
        expert_outputs_list = []
        for expert_idx in range(self.hparams.num_experts):
            expert_output = self.femoe_experts[expert_idx](f1_features_all_cams)
            expert_outputs_list.append(expert_output)

        # Stack expert outputs: (B * num_cameras, num_experts, c_f2_expert, h_f2_expert, w_f2_expert)
        expert_outputs_tensor = torch.stack(expert_outputs_list, dim=1)

        # Reshape gate weights for broadcasting
        gate_weights_reshaped = gate_weights_all_cams.view(
            batch_size * num_cameras, self.hparams.num_experts, 1, 1, 1
        )

        # Weighted sum of expert outputs
        v_moe_all_cams = torch.sum(gate_weights_reshaped * expert_outputs_tensor, dim=1)
        # v_moe_all_cams: (B * num_cameras, c_f2_expert, h_f2_expert, w_f2_expert)

        # 4. Reshape MoE outputs per camera and apply Spatial Attention
        # Reshape v_moe_all_cams to (B, num_cameras, C_expert, H_expert, W_expert)
        v_moe_per_camera_view = v_moe_all_cams.view(
            batch_size,
            num_cameras,
            self.hparams.c_f2_expert,
            self.hparams.h_f2_expert,
            self.hparams.w_f2_expert
        )

        compressed_features_from_each_camera = []
        for i in range(num_cameras):
            # Get features for the i-th camera: (B, C_expert, H_expert, W_expert)
            current_camera_features = v_moe_per_camera_view[:, i, :, :, :]
            # Apply spatial gate network: (B, C_expert)
            compressed_cam_feat = self.spatial_gates[i](current_camera_features)
            compressed_features_from_each_camera.append(compressed_cam_feat)

        # Concatenate compressed features from all cameras along the channel dimension
        # Each element in list is (B, C_expert), so result is (B, num_cameras * C_expert)
        aggregated_features = torch.cat(compressed_features_from_each_camera, dim=1)

        # 5. Waypoint prediction
        waypoints_flat = self.waypoint_head(aggregated_features)
        pred_waypoints = waypoints_flat.view(
            batch_size, self.hparams.num_waypoints, self.hparams.waypoint_dims
        )
        return pred_waypoints

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch, stage: str):
        multi_camera_images, gt_wp = batch
        pred = self(multi_camera_images)
        l2 = self.l2_error(pred, gt_wp)
        loss = l2
        coll = self.collision_rate(pred)
    
        log_dict = {f"{stage}_loss": loss, f"{stage}_l2": l2, f"{stage}_coll": coll}
    
        if stage == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            log_dict["lr"] = current_lr
    
        self.log_dict(log_dict, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

# Multi-camera preprocessing
def preprocess_multicam_img(img_np, hw=(224, 224)):
    """Same as simple_resnet but for multi-camera setup."""
    img = torch.from_numpy(img_np.astype("float32") / 255.0).permute(2, 0, 1)
    H, W = img.shape[1:]
    out_h, out_w = hw

    scale = H / out_h
    crop_w = int(out_w * scale)
    crop_h = H

    x1 = (W - crop_w) // 2
    y1 = 0
    img_cropped = img[:, y1:y1+crop_h, x1:x1+crop_w]

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    img_normalized = normalize(img_cropped)
    return tv_resize(img_normalized, hw, interpolation=3)

class MultiCamCollate:
    def __init__(self, hw=(224, 224), num_waypoints=3):
        self.hw = hw
        self.num_waypoints = num_waypoints
        # Define camera order for consistent multi-camera tensor
        self.camera_names = [
            "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", 
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"
        ]

    def __call__(self, samples):
        multi_cam_imgs, wps = [], []
        
        for sample in samples:
            sd = sample["sensor_data"]
            traj = sample["trajectory"]
            
            # Collect images from all 6 cameras in consistent order
            cam_imgs = []
            for cam_name in self.camera_names:
                if cam_name in sd:
                    img_np = sd[cam_name]["img"]
                    cam_imgs.append(preprocess_multicam_img(img_np, self.hw))
                else:
                    # If camera missing, use zeros (fallback)
                    cam_imgs.append(torch.zeros(3, self.hw[0], self.hw[1]))
            
            # Stack camera images: (6, 3, H, W)
            multi_cam_tensor = torch.stack(cam_imgs, dim=0)
            multi_cam_imgs.append(multi_cam_tensor)
            
            # Extract waypoints
            wps.append(torch.tensor(traj[:self.num_waypoints, :2], dtype=torch.float32))
        
        # Final batch tensors
        multi_cam_batch = torch.stack(multi_cam_imgs, dim=0)  # (B, 6, 3, H, W)
        wp_batch = torch.stack(wps, dim=0)  # (B, num_waypoints, 2)
        
        return multi_cam_batch, wp_batch

def make_multicam_collate(hw=(224, 224), num_waypoints=3):
    return MultiCamCollate(hw, num_waypoints)

###############################################################################
#                             Training Script                                 #
###############################################################################
if __name__ == "__main__":
    import os
    import sys
    import argparse
    from torch.utils.data import DataLoader
    from pytorch_lightning.loggers import TensorBoardLogger

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from nuscenes_dataset import NuScenesDataset

    # Parse arguments
    parser = argparse.ArgumentParser(description="MiniDrive E2E Multi-Camera Training")
    parser.add_argument("--nusc_root", default="/scratch/gautschi/mgagvani/nuscenes", 
                       help="Path to NuScenes dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--num_waypoints", type=int, default=3, help="Number of waypoints to predict")
    
    # Model hyperparameters
    parser.add_argument("--vision_encoder", default="resnet18", help="Vision encoder model name")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of FE-MoE experts")
    parser.add_argument("--c_f1", type=int, default=128, help="Vision encoder output channels")
    parser.add_argument("--h_f1", type=int, default=14, help="Vision encoder output height")
    parser.add_argument("--w_f1", type=int, default=14, help="Vision encoder output width")
    parser.add_argument("--c_f2_expert", type=int, default=64, help="Expert output channels")
    parser.add_argument("--h_f2_expert", type=int, default=28, help="Expert output height")
    parser.add_argument("--w_f2_expert", type=int, default=28, help="Expert output width")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512, help="MLP hidden dimension")
    
    # Training setup
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory for data loading")
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="Use persistent workers")
    
    args = parser.parse_args()

    # Create model
    model = MiniDriveE2E(
        num_cameras=6,
        vision_encoder_name=args.vision_encoder,
        c_f1=args.c_f1,
        h_f1=args.h_f1,
        w_f1=args.w_f1,
        num_experts=args.num_experts,
        c_f2_expert=args.c_f2_expert,
        h_f2_expert=args.h_f2_expert,
        w_f2_expert=args.w_f2_expert,
        num_waypoints=args.num_waypoints,
        waypoint_dims=2,
        mlp_hidden_dim=args.mlp_hidden_dim,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = NuScenesDataset(
        nuscenes_path=args.nusc_root,
        version="v1.0-trainval",
        future_seconds=3,
        future_hz=1,
        split="train",
        get_img_data=True  # Need images for multi-camera
    )
    
    val_dataset = NuScenesDataset(
        nuscenes_path=args.nusc_root,
        version="v1.0-trainval",
        future_seconds=3,
        future_hz=1,
        split="val",
        get_img_data=True
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_multicam_collate(hw=(224, 224), num_waypoints=args.num_waypoints),
        pin_memory=args.pin_memory,
        pin_memory_device="cuda" if torch.cuda.is_available() else None,
        persistent_workers=args.persistent_workers,
        drop_last=False  # (Ensure consistent batch sizes) --?
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_multicam_collate(hw=(224, 224), num_waypoints=args.num_waypoints),
        pin_memory=args.pin_memory,
        pin_memory_device="cuda" if torch.cuda.is_available() else None,
        persistent_workers=args.persistent_workers,
        drop_last=False
    )

    # Setup logging
    logger = TensorBoardLogger(
        save_dir="/scratch/gautschi/mgagvani/runs",
        name="minidrive_e2e_multicam",
        default_hp_metric=False,
        version=None,
        log_graph=True,
    )

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="/scratch/gautschi/mgagvani/runs/minidrive_e2e/checkpoints",
            filename="minidrive_e2e-{epoch:02d}-{val_loss:.3f}",
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="32-true",
        devices=-1 if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        log_every_n_steps=10,
        default_root_dir="/scratch/gautschi/mgagvani/runs/minidrive_e2e",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Print model summary
    print(f"\nModel Summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print data shapes for verification
    print(f"\nData shapes:")
    sample_batch = next(iter(train_loader))
    print(f"Multi-camera images: {sample_batch[0].shape}")  # Should be (batch, 6, 3, 224, 224)
    print(f"Waypoints: {sample_batch[1].shape}")  # Should be (batch, num_waypoints, 2)

    # Verify forward pass works
    print(f"\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        test_output = model(sample_batch[0])
        print(f"Model output shape: {test_output.shape}")  # Should be (batch, num_waypoints, 2)
    model.train()

    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch}")
    print(f"Vision encoder: {args.vision_encoder}")
    print(f"Number of experts: {args.num_experts}")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")