import os
import shutil
from pathlib import Path
import time

def main():
    source_dir = Path("kaist-cvpr15")
    images_dest = Path("data/thermal_images")
    labels_dest = Path("data/labels")

    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    moved_labels = 0

    print("Scanning and moving files...")

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        
        parts = root_path.parts
        set_name = None
        v_name = None
        
        for part in parts:
            if part.startswith("set") and len(part) == 5:
                set_name = part
            if part.startswith("V") and len(part) == 4:
                v_name = part
                
        prefix = ""
        if set_name and v_name:
            prefix = f"{set_name}_{v_name}_"

        # Check for images in 'lwir' folder
        if root_path.name.lower() == 'lwir':
            for file in files:
                ext = file.lower()
                if ext.endswith('.png') or ext.endswith('.jpg') or ext.endswith('.jpeg'):
                    src_path = root_path / file
                    new_name = prefix + file
                    dst_path = images_dest / new_name
                    if not dst_path.exists():
                        try:
                            shutil.copy2(str(src_path), str(dst_path))
                            try:
                                os.remove(str(src_path))
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"Error copying {src_path}: {e}")
                    moved_images += 1

        # Check for annotations
        is_annotation_folder = any('annotation' in p.lower() for p in parts)
        if is_annotation_folder:
            for file in files:
                ext = file.lower()
                if ext.endswith('.xml') or ext.endswith('.vbb') or ext.endswith('.txt'):
                    src_path = root_path / file
                    new_name = prefix + file
                    dst_path = labels_dest / new_name
                    if not dst_path.exists():
                        try:
                            shutil.copy2(str(src_path), str(dst_path))
                            try:
                                os.remove(str(src_path))
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"Error copying {src_path}: {e}")
                    moved_labels += 1

    print(f"Moved {moved_images} thermal images.")
    print(f"Moved {moved_labels} label files.")

if __name__ == "__main__":
    main()
