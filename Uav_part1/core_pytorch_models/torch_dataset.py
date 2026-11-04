import os
import glob
import random
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PyTorchYOLODataset(Dataset):
    """
    Custom Dataset that reads YOLO format (images/ and labels/ with normalized txts)
    and mathematically converts them to PyTorch format (absolute coordinates).

    For training split: applies data augmentation (horizontal flip, brightness/contrast jitter).
    For val/test split: no augmentation, just ToTensor.
    """
    def __init__(self, data_dir, split='train', transforms=None):
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        self.is_train = (split == 'train')

        self.images_dir = os.path.join(data_dir, 'images', split)
        self.labels_dir = os.path.join(data_dir, 'labels', split)

        # Get all image paths
        self.image_paths = glob.glob(os.path.join(self.images_dir, '*.*'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 2. Find corresponding label file
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{basename}.txt")

        boxes = []
        labels = []

        # 3. Parse YOLO format and convert to absolute PyTorch format
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if len(line.strip()) == 0:
                        continue
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = map(float, parts)

                    # Mathematical conversion from normalized YOLO to absolute pixels
                    x_center_abs = x_center * w
                    y_center_abs = y_center * h
                    width_abs  = width * w
                    height_abs = height * h

                    x_min = x_center_abs - (width_abs / 2)
                    y_min = y_center_abs - (height_abs / 2)
                    x_max = x_center_abs + (width_abs / 2)
                    y_max = y_center_abs + (height_abs / 2)

                    # Clamp to image bounds
                    x_min = max(0.0, x_min)
                    y_min = max(0.0, y_min)
                    x_max = min(float(w), x_max)
                    y_max = min(float(h), y_max)

                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        # PyTorch expects class 0 = background, so shift YOLO class by +1
                        labels.append(int(class_id) + 1)

        # --- Data Augmentation (training only) ---
        if self.is_train and len(boxes) > 0:
            # Horizontal flip (50% chance)
            if random.random() > 0.5:
                img = TF.hflip(img)
                flipped_boxes = []
                for box in boxes:
                    x_min_f = w - box[2]
                    x_max_f = w - box[0]
                    flipped_boxes.append([x_min_f, box[1], x_max_f, box[3]])
                boxes = flipped_boxes

            # Brightness + contrast jitter (subtle, helps thermal generalization)
            img = T.ColorJitter(brightness=0.3, contrast=0.3)(img)

        # If no objects, return empty tensors
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
            area   = torch.zeros((0,),   dtype=torch.float32)
            iscrowd = torch.zeros((0,),  dtype=torch.int64)
        else:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": image_id,
            "area":     area,
            "iscrowd":  iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


# Helper function to collate batches for object detection
def collate_fn(batch):
    return tuple(zip(*batch))
