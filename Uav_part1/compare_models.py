import os
import pandas as pd
from ultralytics import YOLO, RTDETR  # type: ignore

def run_evaluation_comparison():
    # You can put ANY weights here! 
    # For example, base weights ('yolov8n.pt') or your trained weights ('runs/rgb_train/train18/weights/best.pt')
    models_to_test = {
        'YOLOv8_Nano_Base': ('yolov8n.pt', 'YOLO'),
        'YOLOv8_Small_Base': ('yolov8s.pt', 'YOLO'),
        'RT_DETR_Large_Base': ('rtdetr-l.pt', 'RTDETR'),
        # Your previously trained models!
        'My_Trained_RGB_Model': ('runs/rgb_train/train18/weights/last.pt', 'YOLO'),
        'My_Trained_Thermal_Model': ('runs/thermal_train/train23/weights/best.pt', 'YOLO')
    }

    dataset_path = 'dataset.yaml'
    imgsz = 640
    batch_size = 8
    
    results_summary = []

    print("==================================================")
    print("🚀 Starting Model EVALUATION Pipeline 🚀")
    print("This will test the models on your dataset WITHOUT training them.")
    print("==================================================\n")

    for model_name, (weights, model_type) in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name} using {weights}...")
        print(f"{'='*50}\n")
        
        try:
            # Initialize the appropriate model
            if model_type == 'YOLO':
                model = YOLO(weights)
            else:
                model = RTDETR(weights)
            
            # Evaluate the model on the validation set (no training!)
            results = model.val(
                data=dataset_path,
                imgsz=imgsz,
                batch=batch_size,
                device=0, # Assuming GPU 0
                split='val' # Evaluate on the validation set
            )
            
            # Extract metrics
            map50_95 = results.box.map    # mAP@0.5:0.95
            map50 = results.box.map50     # mAP@0.5
            
            # Inference speed
            inference_speed_ms = results.speed['inference'] if hasattr(results, 'speed') else "N/A"
            
            results_summary.append({
                'Model': model_name,
                'Weights': weights,
                'mAP50-95': round(map50_95, 4),
                'mAP50': round(map50, 4),
                'Inference Speed (ms)': round(inference_speed_ms, 2) if isinstance(inference_speed_ms, float) else inference_speed_ms
            })
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results_summary.append({
                'Model': model_name,
                'Weights': weights,
                'mAP50-95': 'Failed',
                'mAP50': 'Failed',
                'Inference Speed (ms)': 'Failed'
            })

    print("\n\n==================================================")
    print("🏆 Evaluation Complete 🏆")
    print("==================================================")
    
    # Print results as a nice table
    df = pd.DataFrame(results_summary)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # Save the summary to a CSV file
    os.makedirs('runs/compare', exist_ok=True)
    df.to_csv('runs/compare/evaluation_summary.csv', index=False)
    print("\nSummary saved to 'runs/compare/evaluation_summary.csv'")

if __name__ == '__main__':
    run_evaluation_comparison()
