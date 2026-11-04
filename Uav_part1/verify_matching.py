import os
from pathlib import Path

def verify():
    images_dir = Path("data/thermal_images")
    labels_dir = Path("data/labels")

    image_files = {f.stem for f in images_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']}
    label_files = {f.stem for f in labels_dir.glob("*.txt") if f.name != 'classes.txt'}

    print(f"Total thermal images found: {len(image_files)}")
    print(f"Total label files found: {len(label_files)}")

    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files

    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Labels without images: {len(labels_without_images)}")

    if not images_without_labels and not labels_without_images:
        print("Verification passed! Exact matching achieved.")
    else:
        if images_without_labels:
            print("Sample images without labels (negative samples):", list(images_without_labels)[:5])
            # For YOLO training, these act as background images (to reduce false positives).
            # But to ensure strict matching requested, you might want to delete them if needed.
        if labels_without_images:
            print("Sample labels without images (orphaned labels):", list(labels_without_images)[:5])
            # Labels without images should usually be removed because YOLO needs the original image.
            
            print("Cleaning up orphaned label files...")
            for missing_stem in labels_without_images:
                orphan_file = labels_dir / (missing_stem + ".txt")
                if orphan_file.exists():
                    os.remove(orphan_file)
            print("Removed orphaned labels.")
            

if __name__ == "__main__":
    verify()
