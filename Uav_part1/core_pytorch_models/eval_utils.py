"""
Custom mAP Evaluation Utilities for PyTorch Object Detection.
No external dependencies needed — pure PyTorch implementation.
Computes mAP@0.5 and mAP@0.5:0.95 (COCO-style).
"""

import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    box1: (N, 4) tensor [x1, y1, x2, y2]
    box2: (M, 4) tensor [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection

    return intersection / (union + 1e-6)


def compute_ap(precision_list, recall_list):
    """
    Compute Average Precision using the 11-point interpolation method.
    """
    recall_list = np.concatenate(([0.0], recall_list, [1.0]))
    precision_list = np.concatenate(([0.0], precision_list, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision_list) - 2, -1, -1):
        precision_list[i] = max(precision_list[i], precision_list[i + 1])

    # Find points where recall changes
    indices = np.where(recall_list[1:] != recall_list[:-1])[0]

    ap = np.sum((recall_list[indices + 1] - recall_list[indices]) * precision_list[indices + 1])
    return ap


def evaluate_detections(all_predictions, all_targets, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth for a single IoU threshold.
    Returns AP per class and mAP.
    """
    # Group by class
    class_preds = defaultdict(list)
    class_gts = defaultdict(lambda: defaultdict(list))
    n_gt_per_class = defaultdict(int)

    for img_idx, (pred, gt) in enumerate(zip(all_predictions, all_targets)):
        gt_boxes = gt["boxes"].cpu()
        gt_labels = gt["labels"].cpu()

        for j in range(len(gt_labels)):
            label = gt_labels[j].item()
            class_gts[label][img_idx].append(gt_boxes[j])
            n_gt_per_class[label] += 1

        if len(pred["boxes"]) > 0:
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()

            for j in range(len(pred_labels)):
                label = pred_labels[j].item()
                class_preds[label].append({
                    "box": pred_boxes[j],
                    "score": pred_scores[j].item(),
                    "img_idx": img_idx,
                })

    # Compute AP for each class
    aps = []
    all_classes = set(list(n_gt_per_class.keys()) + list(class_preds.keys()))

    for cls in sorted(all_classes):
        if cls == 0:  # Skip background
            continue

        preds = class_preds[cls]
        n_gt = n_gt_per_class[cls]

        if n_gt == 0:
            continue

        # Sort predictions by confidence (descending)
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = defaultdict(set)

        for i, pred in enumerate(preds):
            img_idx = pred["img_idx"]
            pred_box = pred["box"].unsqueeze(0)

            gt_boxes_for_img = class_gts[cls].get(img_idx, [])
            if len(gt_boxes_for_img) == 0:
                fp[i] = 1
                continue

            gt_tensor = torch.stack(gt_boxes_for_img)
            ious = compute_iou(pred_box, gt_tensor)[0]

            best_iou, best_idx = ious.max(0)
            best_idx = best_idx.item()

            if best_iou.item() >= iou_threshold and best_idx not in matched[img_idx]:
                tp[i] = 1
                matched[img_idx].add(best_idx)
            else:
                fp[i] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(precision, recall)
        aps.append(ap)

    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    return mAP


@torch.no_grad()
def validate(model, data_loader, device, score_threshold=0.05):
    """
    Run validation and compute mAP@0.5 and mAP@0.5:0.95.

    Args:
        model: PyTorch detection model
        data_loader: Validation DataLoader
        device: torch device
        score_threshold: minimum confidence to count a detection

    Returns:
        dict with 'mAP_50' and 'mAP_50_95' values
    """
    model.eval()

    all_predictions = []
    all_targets = []

    from tqdm import tqdm
    print("  Running validation...")
    for i, (images, targets) in enumerate(tqdm(data_loader, desc="Validation")):
        images = [img.to(device) for img in images]

        predictions = model(images)

        for pred, gt in zip(predictions, targets):
            # Filter low-confidence detections
            keep = pred["scores"] >= score_threshold
            filtered_pred = {
                "boxes": pred["boxes"][keep].cpu(),
                "scores": pred["scores"][keep].cpu(),
                "labels": pred["labels"][keep].cpu(),
            }
            all_predictions.append(filtered_pred)

            # Keep ground truth on CPU
            cpu_gt = {k: v.cpu() for k, v in gt.items()}
            all_targets.append(cpu_gt)

    # Compute mAP@0.5
    mAP_50 = evaluate_detections(all_predictions, all_targets, iou_threshold=0.5)

    # Compute mAP@0.5:0.95 (average over IoU thresholds 0.5 to 0.95 step 0.05)
    mAPs = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        mAPs.append(evaluate_detections(all_predictions, all_targets, iou_threshold=iou_thresh))
    mAP_50_95 = np.mean(mAPs)

    model.train()

    return {"mAP_50": mAP_50, "mAP_50_95": mAP_50_95}
