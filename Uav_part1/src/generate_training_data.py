import numpy as np
import os

def generate_synthetic_trajectories(num_samples=8000, seq_length=10, pred_length=5):
    print(f"Generating {num_samples} synthetic smuggler trajectories...")
    
    # X will hold the past 10 frames (x, y coordinates)
    X = np.zeros((num_samples, seq_length, 2))
    # Y will hold the future 5 frames (x, y coordinates)
    y = np.zeros((num_samples, pred_length, 2))
    
    for i in range(num_samples):
        # Starting coordinate (randomized across a 640x640 grid)
        start_x = np.random.uniform(50, 590)
        start_y = np.random.uniform(50, 590)
        
        # Simulating human walking speed (1.0 - 2.0 m/s translated to pixels)
        velocity_x = np.random.uniform(-5.0, 5.0)
        velocity_y = np.random.uniform(-5.0, 5.0)
        
        # Generate the past 10 frames
        for j in range(seq_length):
            X[i, j, 0] = start_x + (velocity_x * j)
            X[i, j, 1] = start_y + (velocity_y * j)
            
        # Current position is the end of the input sequence
        curr_x = X[i, -1, 0]
        curr_y = X[i, -1, 1]
        
        # Simulating a slight change in direction for the future prediction
        drift_x = velocity_x + np.random.uniform(-1.0, 1.0)
        drift_y = velocity_y + np.random.uniform(-1.0, 1.0)
        
        # Generate the future 5 frames
        for k in range(pred_length):
            y[i, k, 0] = curr_x + (drift_x * (k + 1))
            y[i, k, 1] = curr_y + (drift_y * (k + 1))

    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save the arrays
    np.save('data/lstm_X.npy', X)
    np.save('data/lstm_y.npy', y)
    print("Data generation complete! Saved to data/lstm_X.npy and data/lstm_y.npy")
    print(f"Input Shape: {X.shape} | Target Shape: {y.shape}")

if __name__ == '__main__':
    generate_synthetic_trajectories()