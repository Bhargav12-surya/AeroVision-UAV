import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random

def convert_sampled(sample_rate=0.50): # Picks 50% of the data for better mAP
    labels_dir = Path("data/labels")
    classes = {"human": 0} # Pre-define to keep ID consistent
    
    print(f"Starting sampled conversion (Rate: {sample_rate*100}%)...")

    # Get all XML files and shuffle them for a random sample
    xml_files = list(labels_dir.glob("*.xml"))
    random.shuffle(xml_files)
    
    # Calculate how many to keep
    total_to_convert = int(len(xml_files) * sample_rate)
    sampled_files = xml_files[:total_to_convert]
    
    converted_count = 0
    
    for xml_file in sampled_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find("size")
            if size is None: continue
                
            img_w = float(size.find("width").text)
            img_h = float(size.find("height").text)
            yolo_lines = []

            for obj in root.findall("object"):
                name = obj.find("name").text
                # Only care about humans for your UAV project
                if "person" in name.lower() or "human" in name.lower():
                    c = 0 # Class 0 for human
                    bndbox = obj.find("bndbox")
                    x, y = float(bndbox.find("x").text), float(bndbox.find("y").text)
                    w, h = float(bndbox.find("w").text), float(bndbox.find("h").text)
                    
                    # Normalize for YOLOv8
                    x_center = (x + w / 2.0) / img_w
                    y_center = (y + h / 2.0) / img_h
                    w_norm, h_norm = w / img_w, h / img_h
                    
                    yolo_lines.append(f"{c} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # Only create the .txt if there is a human (reduces background noise)
            if yolo_lines:
                txt_path = xml_file.with_suffix('.txt')
                with open(txt_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                converted_count += 1
            
            # Optional: Remove XML after processing to save space
            # os.remove(xml_file)

        except Exception as e:
            print(f"Error: {e}")

    print(f"✅ Sampled Conversion Complete!")
    print(f"Processed {len(sampled_files)} files. Created {converted_count} human labels.")

if __name__ == "__main__":
    # 50% of data → significantly better mAP than the previous 15%
    # Increase to 1.0 for maximum accuracy (uses full dataset)
    convert_sampled(0.50)
    