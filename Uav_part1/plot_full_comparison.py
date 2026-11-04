import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_full_comparison_plots():
    os.makedirs('runs/compare', exist_ok=True)
    
    # 1. Load YOLO data from CSV (created by compare_models.py)
    csv_path = 'runs/compare/evaluation_summary.csv'
    if os.path.exists(csv_path):
        yolo_df = pd.read_csv(csv_path)
    else:
        print("YOLO CSV not found. Plotting only Core models.")
        yolo_df = pd.DataFrame()
        
    # 2. Add Core PyTorch Data (Known metrics from previous evaluation)
    # We use hardcoded values here to avoid loading PyTorch and crashing the currently running training!
    core_data = [
        {'Model': 'Core_Faster_RCNN_RGB', 'Weights': 'faster_rcnn_rgb_best.pth', 'mAP50-95': 0.3214, 'mAP50': 0.5040, 'Inference Speed (ms)': 45.0},
        {'Model': 'Core_Faster_RCNN_Thermal', 'Weights': 'faster_rcnn_thermal_best.pth', 'mAP50-95': 0.3616, 'mAP50': 0.5115, 'Inference Speed (ms)': 45.0}
    ]
    
    core_df = pd.DataFrame(core_data)
    
    # Combine datasets
    df = pd.concat([yolo_df, core_df], ignore_index=True)
    
    # Clean up names for plot readability
    df['Model'] = df['Model'].str.replace('_Base', '').str.replace('My_Trained_', '')
    
    plt.style.use('ggplot')
    
    # Define colors: Orange/Red for YOLO, Blue for Core PyTorch
    colors = ['#C44E52' if 'YOLO' in name or 'RT_DETR' in name else '#4C72B0' for name in df['Model']]

    # --- Plot 1: mAP50 Comparison ---
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['Model'], df['mAP50'], color=colors)
    plt.title('Final Comparison: mAP@0.5 (YOLO vs Core PyTorch)', fontsize=15, pad=20)
    plt.ylabel('mAP@0.5 (Higher is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(df['mAP50']) * 1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{round(yval*100, 1)}%", ha='center', va='bottom', fontweight='bold')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#C44E52', label='Ultralytics YOLO'),
                       Patch(facecolor='#4C72B0', label='Core PyTorch')]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('runs/compare/full_comparison_map50.png', dpi=300)
    plt.close()

    # --- Plot 2: Inference Speed Comparison ---
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['Model'], df['Inference Speed (ms)'], color=colors)
    plt.title('Speed Comparison: Inference Time (YOLO vs Core PyTorch)', fontsize=15, pad=20)
    plt.ylabel('Speed (ms per image) (Lower is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(df['Inference Speed (ms)']) * 1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{round(yval, 1)} ms", ha='center', va='bottom', fontweight='bold')
        
    plt.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig('runs/compare/full_comparison_speed.png', dpi=300)
    plt.close()

    print("Plots successfully generated and saved to 'runs/compare/' folder!")

if __name__ == '__main__':
    generate_full_comparison_plots()
