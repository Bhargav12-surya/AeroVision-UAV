import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_dataset import PyTorchYOLODataset, collate_fn
from eval_utils import validate


def get_faster_rcnn_model(num_classes):
    """
    Build Faster R-CNN with ResNet-50-FPN backbone.
    - min_size=640, max_size=1024: faster than default (800, 1333)
    - Custom SMALL anchors: people in UAV images are 15-60px tall.
      Default anchors (32-512px) are far too large for UAV-scale people.
    - Backbone body frozen initially (unfrozen progressively)4 
    """
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=640,
            max_size=1024,
            rpn_anchor_generator=anchor_generator
        )
    except (ImportError, TypeError):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            min_size=640,
            max_size=1024
        )

    for param in model.backbone.body.parameters():
        param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform():
    """Standard transform: convert PIL Image to Tensor."""
    return T.Compose([
        T.ToTensor(),
    ])


def train():
    # NOTE: cudnn.benchmark=False for Faster R-CNN (variable image sizes).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_amp = device.type == 'cuda'
    print(f"Using device: {device} | AMP: {use_amp}")

    num_classes = 2  # background (0) + person (1)

    # --- Data Setup ---
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    print(f"Data directory: {data_dir}")

    num_workers = 0  # Most stable on Windows + OneDrive paths

    train_dataset = PyTorchYOLODataset(data_dir, split='train', transforms=get_transform())
    print(f"Train dataset: {len(train_dataset)} images")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )

    val_dataset = PyTorchYOLODataset(data_dir, split='val', transforms=get_transform())
    print(f"Val dataset:   {len(val_dataset)} images")

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # --- Model Setup ---
    model = get_faster_rcnn_model(num_classes)
    model.to(device)

    # Only optimize parameters that require grad (backbone is frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    num_epochs = 20       # Increased from 10 → better convergence
    best_mAP = 0.0

    # Early stopping config
    patience = 3  # Stop after 3 validation rounds with no improvement
    epochs_without_improvement = 0

    # Progressive unfreezing:
    # Epochs 1-5:  backbone FROZEN  → fast training, detection head learns
    # Epochs 6-20: backbone UNFROZEN → full fine-tune at low backbone LR
    unfreeze_epoch = 5
    val_every_n_epochs = 3

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'pytorch_models')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "faster_rcnn_rgb_best.pth")

    print("=" * 55)
    print(">>> Starting Pure PyTorch RGB Training <<<")
    print("   Model       : Faster R-CNN (ResNet-50-FPN)")
    print(f"   Epochs      : {num_epochs}")
    print(f"   Batch Size  : 2")
    print(f"   Input Size  : 640x1024 (reduced for speed)")
    print(f"   Backbone    : Frozen for {unfreeze_epoch} epochs, then unfrozen")
    print(f"   Val Every   : {val_every_n_epochs} epochs")
    print(f"   Device      : {device} | AMP: {use_amp}")
    print(f"   Train       : {len(train_dataset)} images")
    print(f"   Val         : {len(val_dataset)} images")
    print("=" * 55 + "\n")

    # --- Training Loop ---
    for epoch in range(num_epochs):
        # ---- Progressive Unfreezing ----
        if epoch == unfreeze_epoch:
            print(f"  [Epoch {epoch+1}] Unfreezing backbone for full fine-tuning...")
            for param in model.backbone.body.parameters():
                param.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
            backbone_params = list(model.backbone.body.parameters())
            head_params = [p for p in params if not any(p is bp for bp in backbone_params)]
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5},  # low LR for backbone
                {'params': head_params,    'lr': 1e-4},   # normal LR for head
            ], weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            trainable_now = sum(p.numel() for p in params)
            print(f"  Trainable params now: {trainable_now:,}")

        # ---- TRAIN PHASE ----
        model.train()
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for i, (images, targets) in enumerate(pbar):
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"  WARNING: Non-finite loss at batch {i}, skipping...")
                continue

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{losses.item():.4f}")

        lr_scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  -> Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # SPEED: Skip validation except every val_every_n_epochs epochs
        is_last_epoch = (epoch + 1 == num_epochs)
        should_validate = ((epoch + 1) % val_every_n_epochs == 0) or is_last_epoch

        if should_validate:
            metrics = validate(model, val_loader, device)
            mAP_50 = metrics["mAP_50"]
            mAP_50_95 = metrics["mAP_50_95"]
            print(f"  -> Val mAP@0.5: {mAP_50:.4f} | mAP@0.5:0.95: {mAP_50_95:.4f}")

            if mAP_50 > best_mAP:
                best_mAP = mAP_50
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_path)
                print(f"  -> NEW BEST MODEL SAVED (mAP@0.5: {best_mAP:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  -> No improvement. Early stopping counter: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {patience} validation checks.")
                break
        else:
            print(f"  -> Validation skipped (next at epoch {((epoch // val_every_n_epochs) + 1) * val_every_n_epochs})")

        print()

    # --- Save Final Model ---
    final_path = os.path.join(save_dir, "faster_rcnn_rgb_last.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete!")
    print(f"  Best model : {best_path} (mAP@0.5: {best_mAP:.4f})")
    print(f"  Last model : {final_path}")


if __name__ == '__main__':
    train()
