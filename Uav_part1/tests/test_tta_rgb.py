from ultralytics import YOLO

if __name__ == '__main__':
    # Load your RGB morning model
    model = YOLO(r'runs\detect\train18\weights\best.pt')
    
    print("\n--- Activating Test Time Augmentation (TTA) ---")
    print("The AI will now multi-scan each image to boost confidence...")
    
    # augment=True turns on the TTA multi-scanning engine
    r = model.val(data='dataset.yaml', conf=0.25, augment=True)
    
    print(f"\nFinal TTA Result -> mAP50: {r.box.map50:.3f}")