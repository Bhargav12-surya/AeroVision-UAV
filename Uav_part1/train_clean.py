from ultralytics import YOLO

if __name__ == '__main__':
    import os
    # Use P2 architecture for tiny objects and train completely independently
    model = YOLO('yolov8m-p2.yaml').load('yolov8m.pt')

    # Start the clean, thermal-focused training run
    model.train(
        project='runs/thermal_train',
        name='train',
        data='dataset.yaml',
        epochs=150,
        imgsz=1024,
        device=0,
        batch=4,
        mosaic=1.0,          
        mixup=0.0,           # Disabled to prevent false-positive thermal ghosting
        copy_paste=0.0,      # Disabled
        degrees=5.0,
        scale=0.1,
        fliplr=0.5,
        hsv_h=0.0,           # Disabled for thermal data
        hsv_s=0.0,           # Disabled for thermal data
        hsv_v=0.0,           # Disabled for thermal data
        box=10.0,            # Increased for tighter bounding boxes
        cls=1.0,             # Balanced with higher box loss
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,
        cos_lr=True
    )