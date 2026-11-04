import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    csv_path = 'runs/compare/evaluation_summary.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Use a nice style if available
    plt.style.use('ggplot')

    # 1. Plot mAP50-95
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['mAP50-95'], color='#4C72B0')
    plt.title('Model Accuracy Comparison: mAP50-95 (Higher is Better)', fontsize=14, pad=15)
    plt.ylabel('mAP 50-95', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(df['mAP50-95']) * 1.2) # Add some headroom
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{round(yval, 4)}", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('runs/compare/map50_95_comparison.png', dpi=300)
    plt.close()

    # 2. Plot Inference Speed
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Inference Speed (ms)'], color='#C44E52')
    plt.title('Model Speed Comparison: Inference Time (Lower is Better)', fontsize=14, pad=15)
    plt.ylabel('Speed (ms per image)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(df['Inference Speed (ms)']) * 1.2) # Add some headroom
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{round(yval, 1)} ms", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('runs/compare/speed_comparison.png', dpi=300)
    plt.close()

    print("Plots generated successfully!")

if __name__ == '__main__':
    generate_plots()
