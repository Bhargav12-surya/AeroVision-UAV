import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_yolo_plots():
    csv_path = 'runs/compare/evaluation_summary.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Filter for YOLO models only
    # Assuming 'My_Trained_RGB_Model' and 'My_Trained_Thermal_Model' are YOLOv8 models based on your training scripts
    yolo_models = [
        'YOLOv8_Nano_Base', 
        'YOLOv8_Small_Base', 
        'My_Trained_RGB_Model', 
        'My_Trained_Thermal_Model'
    ]
    df_yolo = df[df['Model'].isin(yolo_models)]

    # Rename trained models for clarity in plot
    df_yolo['Model'] = df_yolo['Model'].replace({
        'My_Trained_RGB_Model': 'YOLOv8m (Trained RGB)',
        'My_Trained_Thermal_Model': 'YOLOv8 (Trained Thermal)'
    })

    # Use a nice style if available
    plt.style.use('ggplot')

    # 1. Plot mAP50
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_yolo['Model'], df_yolo['mAP50'], color='#55A868')
    plt.title('YOLO Models Accuracy Comparison: mAP50 (Higher is Better)', fontsize=14, pad=15)
    plt.ylabel('mAP @ 0.50', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(df_yolo['mAP50']) * 1.2) # Add some headroom
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{round(yval, 4)}", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('runs/compare/yolo_map50_comparison.png', dpi=300)
    plt.close()

    # 2. Plot Inference Speed
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_yolo['Model'], df_yolo['Inference Speed (ms)'], color='#C44E52')
    plt.title('YOLO Models Speed Comparison: Inference Time (Lower is Better)', fontsize=14, pad=15)
    plt.ylabel('Speed (ms per image)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(df_yolo['Inference Speed (ms)']) * 1.2) # Add some headroom
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{round(yval, 1)} ms", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('runs/compare/yolo_speed_comparison.png', dpi=300)
    plt.close()

    print("Plots generated successfully!")

if __name__ == '__main__':
    generate_yolo_plots()
