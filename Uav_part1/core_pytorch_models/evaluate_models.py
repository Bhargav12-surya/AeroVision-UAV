"""
evaluate_models.py
------------------
Loads all saved Faster R-CNN model checkpoints and evaluates them
on the validation set, reporting mAP@0.5 and mAP@0.5:0.95.

Usage:
    python core_pytorch_models\evaluate_models.py
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch_dataset import PyTorchYOLODataset, collate_fn
from eval_utils import validate


def get_faster_rcnn_model(num_classes):
    """Build Faster R-CNN — same architecture used during training."""
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        # Use no pretrained weights — we will load our own saved weights
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    except Exception:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate_model(model_path, val_loader, device, label):
    """Load a .pth file and evaluate it on the val loader."""
    print(f"\n{'='*55}")
    print(f"  Evaluating: {label}")
    print(f"  File      : {os.path.basename(model_path)}")
    print(f"{'='*55}")

    if not os.path.exists(model_path):
        print(f"  ❌ File not found: {model_path}")
        return None

    model = get_faster_rcnn_model(num_classes=2)

    # Load state dict — handle both plain state_dict and full checkpoint dicts
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        saved_epoch = ckpt.get('epoch', '?')
        saved_mAP   = ckpt.get('best_mAP', '?')
        print(f"  Checkpoint epoch : {saved_epoch}")
        print(f"  Saved best mAP   : {saved_mAP}")
    else:
        state_dict = ckpt  # plain state_dict

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    metrics = validate(model, val_loader, device)
    mAP_50    = metrics["mAP_50"]
    mAP_50_95 = metrics["mAP_50_95"]

    print(f"\n  ✅ Results:")
    print(f"     mAP@0.5      : {mAP_50:.4f}  ({mAP_50*100:.2f}%)")
    print(f"     mAP@0.5:0.95 : {mAP_50_95:.4f}  ({mAP_50_95*100:.2f}%)")

    return {"label": label, "mAP_50": mAP_50, "mAP_50_95": mAP_50_95}


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\nDevice: {device}")

    # Base paths
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'runs', 'pytorch_models')

    # Build validation DataLoader (shared across all models)
    print(f"\nLoading validation dataset from: {data_dir}")
    val_dataset = PyTorchYOLODataset(data_dir, split='val',
                                     transforms=T.Compose([T.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=0, pin_memory=False,
        collate_fn=collate_fn
    )
    print(f"Val images: {len(val_dataset)}")

    # ── Define all models to evaluate ──────────────────────────────────────
    models_to_eval = [
        {
            "path":  os.path.join(model_dir, "faster_rcnn_rgb_best.pth"),
            "label": "RGB  — Faster R-CNN (best checkpoint)"
        },
        {
            "path":  os.path.join(model_dir, "faster_rcnn_rgb_last.pth"),
            "label": "RGB  — Faster R-CNN (last epoch)"
        },
        {
            "path":  os.path.join(model_dir, "faster_rcnn_thermal_best.pth"),
            "label": "Thermal — Faster R-CNN (best checkpoint)"
        },
        {
            "path":  os.path.join(model_dir, "faster_rcnn_thermal_last.pth"),
            "label": "Thermal — Faster R-CNN (last epoch)"
        },
        {
            "path":  os.path.join(model_dir, "thermal_checkpoint_latest.pth"),
            "label": "Thermal — latest checkpoint (resume point)"
        },
    ]

    # ── Run evaluation ──────────────────────────────────────────────────────
    results = []
    for entry in models_to_eval:
        result = evaluate_model(entry["path"], val_loader, device, entry["label"])
        if result:
            results.append(result)

    # ── Summary table ───────────────────────────────────────────────────────
    if results:
        print(f"\n\n{'='*65}")
        print(f"  SUMMARY — All Models")
        print(f"{'='*65}")
        print(f"  {'Model':<42} {'mAP@0.5':>8}  {'mAP@0.5:0.95':>12}")
        print(f"  {'-'*42} {'-'*8}  {'-'*12}")
        best_result = max(results, key=lambda r: r["mAP_50"])
        for r in results:
            marker = " ⭐" if r is best_result else ""
            print(f"  {r['label']:<42} {r['mAP_50']*100:>7.2f}%  {r['mAP_50_95']*100:>11.2f}%{marker}")
        print(f"{'='*65}")
        print(f"\n  Best model: {best_result['label']}")
        print(f"             mAP@0.5 = {best_result['mAP_50']*100:.2f}%")


if __name__ == '__main__':
    main()
