import torch
import torch.nn as nn
import numpy as np
import os

# 1. Define the Drone's Prediction Brain (LSTM)
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2, pred_length=5):
        super().__init__()
        self.pred_length = pred_length
        # The memory cells
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # The output layer that calculates the future coordinates
        self.fc = nn.Linear(hidden_size, output_size * pred_length)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Grab the final memory state
        out = self.fc(out)
        return out.view(-1, self.pred_length, 2) # Reshape to 5 future (x,y) points

def train_brain():
    print("--- Booting Phase 3 LSTM Training Sequence ---")
    
    # 2. Check for the synthetic data
    if not os.path.exists('data/lstm_X.npy'):
        print("ERROR: Data not found! You must run generate_training_data.py first.")
        return
        
    print("Loading trajectory data...")
    X = np.load('data/lstm_X.npy')
    y = np.load('data/lstm_y.npy')
    
    # Convert numpy math to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # 3. Setup the AI learning parameters
    model = TrajectoryLSTM()
    criterion = nn.MSELoss() # Mean Squared Error (penalizes coordinate drift)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    print(f"Beginning gradient descent across {epochs} epochs...")
    
    # 4. The Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        
        # Calculate how far off the drone's guess was from the actual path
        loss = criterion(outputs, y_tensor) 
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Memory Drift Loss: {loss.item():.4f}")
            
    # 5. Lock in the weights
    os.makedirs('models/lstm_model', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_model/lstm_trained.pt')
    print("\nSUCCESS: Phase 3 Brain Trained! Weights locked at models/lstm_model/lstm_trained.pt")

if __name__ == '__main__':
    train_brain()