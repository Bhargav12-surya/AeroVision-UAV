from ultralytics import YOLO

if __name__ == '__main__':
    # Load your freshly baked RGB morning model
    model = YOLO(r'runs\detect\train18\weights\best.pt')
    
    # Test how the mAP reacts when we filter out the low-confidence tree shadows
    for conf in [0.5, 0.4, 0.3, 0.25, 0.2]:
        print(f"\n--- Testing Confidence Threshold: {conf} ---")
        r = model.val(data='dataset.yaml', conf=conf)
        print(f"Result for conf={conf} -> mAP50: {r.box.map50:.3f}")