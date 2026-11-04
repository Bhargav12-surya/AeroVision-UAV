from ultralytics import YOLO

if __name__ == '__main__':
    # Load the P2 medium model for small objects
    model = YOLO('yolov8m-p2.yaml').load('yolov8m.pt')

    # Start the RGB morning model training run
    model.train(
        project='runs/rgb_train',
        name='train',
        data='dataset.yaml',
        epochs=150,
        imgsz=640,
        device=0,
        batch=8,
        # Augmentations TURNED BACK ON for daytime RGB variation
        mosaic=1.0,          
        mixup=0.2,           
        copy_paste=0.15,     
        degrees=15.0,
        scale=0.6,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        box=10.0,            # Increased for tighter bounding boxes
        cls=1.0,             # Balanced with higher box loss
        lr0=0.001,
        lrf=0.0001,
        warmup_epochs=3,
        patience=20,
        cos_lr=True
    )