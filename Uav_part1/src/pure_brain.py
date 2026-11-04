import torch
import numpy as np
import matplotlib.pyplot as plt
from lstm_trainer import TrajectoryLSTM

def run_and_plot(input_coordinates):
    print("=== BOOTING INTEGRATED PREDICTION BRAIN ===")
    
    # 1. Load the LSTM Memory Network
    lstm = TrajectoryLSTM()
    try:
        lstm.load_state_dict(torch.load(r'models\lstm_model\lstm_trained.pt'))
        lstm.eval() # Lock weights for inference
        print("[OK] LSTM Brain Loaded.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not find lstm_trained.pt.\n{e}")
        return

    # 2. Execute the Neural Network
    input_tensor = torch.tensor([input_coordinates], dtype=torch.float32)
    with torch.no_grad():
        predicted_path = lstm(input_tensor)

    predicted_coords = np.round(predicted_path[0].numpy(), 1)
    
    print("\n[OK] Math Calculated. Launching Visualizer...")

    # 3. Automatically Extract X and Y for Plotting (No manual copy-pasting!)
    past_x = [coord[0] for coord in input_coordinates]
    past_y = [coord[1] for coord in input_coordinates]
    
    pred_x = [coord[0] for coord in predicted_coords]
    pred_y = [coord[1] for coord in predicted_coords]

    # 4. Generate the Graph
    plt.figure(figsize=(10, 6))

    # Plot Past Trajectory (Green)
    plt.plot(past_x, past_y, marker='o', color='green', label='Past Trajectory (YOLO)', linewidth=2, markersize=6)

    # Plot Predicted Trajectory (Red)
    plt.plot(pred_x, pred_y, marker='X', color='red', linestyle='--', label='LSTM Predicted Path (Crash)', linewidth=2, markersize=8)

    # Draw the jump line
    plt.plot([past_x[-1], pred_x[0]], [past_y[-1], pred_y[0]], color='gray', linestyle=':')

    # Set camera limits
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()

    # UI Elements
    plt.title('Automated UAV Target Trajectory', fontsize=14, fontweight='bold')
    plt.xlabel('X Pixel Coordinate', fontsize=12)
    plt.ylabel('Y Pixel Coordinate', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    # ---------------------------------------------------------
    # The drone's last 10 tracked frames
    # ---------------------------------------------------------
    my_exact_values = [
        [320.0, 240.0], [322.5, 241.0], [325.1, 241.5], [327.2, 242.0], [330.0, 242.1],
        [332.4, 243.0], [335.1, 243.5], [337.8, 244.0], [340.2, 244.5], [342.9, 245.0]
    ]
    
    if len(my_exact_values) != 10:
        print(f"ERROR: The brain requires exactly 10 coordinates. You provided {len(my_exact_values)}.")
    else:
        run_and_plot(my_exact_values)