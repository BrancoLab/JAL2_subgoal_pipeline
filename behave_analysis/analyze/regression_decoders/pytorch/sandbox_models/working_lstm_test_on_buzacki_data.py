"""This script runs a working ugly code LSTM data on buzacki data.
The only thing that needs to be changed is the path to the data.
This is a sense check that I can get a working LSTM model before refactoring and making 
the model bigger and better."""

import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim

# import torch.distributions as dist
# from torch.utils.data import DataLoader
# from torch.utils.data import Subset

# Ensure GPU is used if available else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    logger.warning("Cuda is not available. Using CPU instead this will take forever fix.")
if device == "cuda":
    logger.success("Using CUDA device")


# Define LSTM model ----------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])  # Take the last sequence output
        out = torch.tanh(out) * 3.14  # scale the output to be between -pi and pi
        return out


# Load the data -------------------------------------------------

# Change the path to the data if you want to run this script
# X is of shape (spikes, neurons) and y is of shape (hdir, 1)
file_location = r"D:\\efizz\\JAL005\\005_flip1_2023_09_08T07_36_54\\processed_data\\buzacki_data"
with open(file_location + "\\" "spike_rate_cell.p", "rb") as f:
    x = pickle.load(f)
with open(file_location + "\\" + "angles.p", "rb") as f:
    angles = pickle.load(f)

# Preprocess the data -------------------------------------------

y_reshaped = np.asarray(angles).reshape(len(angles), 1)
y_adjusted = np.nan_to_num(y_reshaped) - np.pi
X_train, X_test, y_train, y_test = train_test_split(x, y_adjusted, test_size=0.2, random_state=42)

# Convert training and test sets to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

# Hyperparameters
input_dim = x.shape[1]
hidden_dim = 100  # Number of LSTM cells
num_layers = 1
output_dim = 1

# Create the LSTM model
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_for_loss = model(X_train_torch)
    loss = loss_fn(y_pred_for_loss, y_train_torch)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Predicting Y
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_torch).cpu().numpy()
    y_pred_train = model(X_train_torch).cpu().numpy()

# # Plot the predicted vs actual
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot training data predictions vs actual values
axes[0].plot(y_pred_train, label="Predicted Train", linewidth=2)
axes[0].plot(y_train_torch.cpu().numpy(), label="Actual Train", linewidth=2)
train_r2 = r2_score(y_train_torch.cpu().numpy(), y_pred_train)
axes[0].set_title(f"LSTM Model Training Set w r2: {train_r2}")
axes[0].legend()

# plot test fit
axes[1].plot(y_pred_test, label="Predicted Test", linewidth=2)
axes[1].plot(y_test_torch.cpu().numpy(), label="Actual labels", linewidth=2)
test_r2 = r2_score(y_test_torch.cpu().numpy(), y_pred_test)
axes[1].set_title(f"LSTM Model Test Set w r2: {test_r2}")
axes[1].legend()
plt.show()
