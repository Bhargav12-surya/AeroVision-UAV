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
      Default anchors (32-512px) are far too large — RPN misses most people.
      Custom anchors (8-128px) dramatically improve recall for small objects.
    - Backbone body frozen initially (unfrozen progressively)
    """
    from torchvision.models.detection.rpn import AnchorGenerator

    # UAV-optimized anchor sizes: much smaller than default (32,64,128,256,512)
    # Each FPN level gets its own scale — sizes match typical person pixel heights
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # tall/square/wide people
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
        # Fallback: build without custom anchors if API doesn't support it
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            min_size=640,
            max_size=1024
        )

    # Freeze backbone initially — unfreeze progressively during training
    for param in model.backbone.body.parameters():
        param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform():
    """Standard transform: convert PIL Image to Tensor."""
    return T.Compose([T.ToTensor()])


def build_optimizer(model, backbone_unfrozen):
    """Build optimizer with differential LR depending on backbone state."""
    if backbone_unfrozen:
        backbone_params = list(model.backbone.body.parameters())
        head_params = [p for p in model.parameters()
                       if p.requires_grad and not any(p is bp for bp in backbone_params)]
        return torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # low LR for backbone
            {'params': head_params,    'lr': 1e-4},
        ], weight_decay=0.0005)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0005)


def train(resume_from=None):
    """
    Train Faster R-CNN for thermal data.

    Args:
        resume_from: path to a checkpoint .pth file to resume from.
                     Set this to resume after an interruption.
                     Example: resume_from='runs/pytorch_models/thermal_checkpoint_epoch11.pth'
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_amp = device.type == 'cuda'
    print(f"Using device: {device} | AMP: {use_amp}")

    num_classes = 2  # background (0) + person (1)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    print(f"Data directory: {data_dir}")

    num_workers = 0

    train_dataset = PyTorchYOLODataset(data_dir, split='train', transforms=get_transform())
    print(f"Train dataset: {len(train_dataset)} images")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=num_workers, pin_memory=False, collate_fn=collate_fn
    )

    val_dataset = PyTorchYOLODataset(data_dir, split='val', transforms=get_transform())
    print(f"Val dataset:   {len(val_dataset)} images")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=num_workers, pin_memory=False, collate_fn=collate_fn
    )

    # --- Model Setup ---
    model = get_faster_rcnn_model(num_classes)
    model.to(device)

    num_epochs = 25        # More epochs → better convergence (was 15)
    unfreeze_epoch = 5     # unfreeze backbone after this epoch
    val_every_n_epochs = 3
    start_epoch = 0
    best_mAP = 0.0
    backbone_unfrozen = False

    # Early stopping config
    patience = 3  # Stop after 3 validation rounds with no improvement
    epochs_without_improvement = 0

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'pytorch_models')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "faster_rcnn_thermal_best.pth")
    checkpoint_path = os.path.join(save_dir, "thermal_checkpoint_latest.pth")

    # --- Resume from checkpoint if provided ---
    if resume_from and os.path.exists(resume_from):
        print(f"\n>>> Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']      # resume from NEXT epoch
        best_mAP = ckpt.get('best_mAP', 0.0)
        backbone_unfrozen = ckpt.get('backbone_unfrozen', False)
        if backbone_unfrozen:
            for param in model.backbone.body.parameters():
                param.requires_grad = True
        epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        print(f"    Resumed from epoch {start_epoch} | best mAP: {best_mAP:.4f} | backbone_unfrozen: {backbone_unfrozen}")
    elif resume_from:
        print(f"WARNING: Checkpoint not found at {resume_from}. Starting fresh.")

    optimizer = build_optimizer(model, backbone_unfrozen)
    # CosineAnnealingLR: smoothly decays LR to near-zero, much better than
    # StepLR which drops LR by 10x suddenly and can stall training.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Fast-forward scheduler state to match resumed epoch
    for _ in range(start_epoch):
        lr_scheduler.step()

    scaler = GradScaler('cuda', enabled=use_amp)

    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    print("=" * 55)
    print(">>> Starting Pure PyTorch Thermal Training <<<")
    print("   Model       : Faster R-CNN (ResNet-50-FPN)")
    print(f"   Epochs      : {start_epoch+1} -> {num_epochs}")
    print(f"   Batch Size  : 2")
    print(f"   Input Size  : 640x1024 (reduced for speed)")
    print(f"   Backbone    : Frozen until epoch {unfreeze_epoch}, then unfrozen")
    print(f"   Val Every   : {val_every_n_epochs} epochs")
    print(f"   Device      : {device} | AMP: {use_amp}")
    print(f"   Checkpoint  : saved every epoch to {checkpoint_path}")
    print("=" * 55 + "\n")

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):

        # ---- Progressive Unfreezing ----
        if epoch == unfreeze_epoch and not backbone_unfrozen:
            print(f"  [Epoch {epoch+1}] Unfreezing backbone for full fine-tuning...")
            for param in model.backbone.body.parameters():
                param.requires_grad = True
            backbone_unfrozen = True
            optimizer = build_optimizer(model, backbone_unfrozen)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(num_epochs - epoch), eta_min=1e-6
            )
            params = [p for p in model.parameters() if p.requires_grad]
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

        # ---- Save checkpoint every epoch so we can resume if interrupted ----
        torch.save({
            'epoch': epoch + 1,          # next epoch to run on resume
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mAP': best_mAP,
            'backbone_unfrozen': backbone_unfrozen,
            'epochs_without_improvement': epochs_without_improvement,
        }, checkpoint_path)
        print(f"  -> Checkpoint saved (epoch {epoch+1})")

        # ---- VALIDATION PHASE ----
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
    final_path = os.path.join(save_dir, "faster_rcnn_thermal_last.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete!")
    print(f"  Best model : {best_path} (mAP@0.5: {best_mAP:.4f})")
    print(f"  Last model : {final_path}")


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # TO RESUME: set resume_from to the checkpoint path, e.g.:
    #   train(resume_from='runs/pytorch_models/thermal_checkpoint_latest.pth')
    # TO START FRESH: leave resume_from=None
    # ---------------------------------------------------------------
    train(resume_from=None)
