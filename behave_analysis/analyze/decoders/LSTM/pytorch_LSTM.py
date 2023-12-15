import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from behave_analysis.analyze.decoders.LSTM.helpers import get_spikes_with_history

class LSTMDataset(Dataset):
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    
    # Batch size is the number of sequences fed into the model at once
    # Sequence length is the number of time steps in each sequence
    # Features is the number of features in each sequence
    '''
    def __init__(self, X, Y, seq_len: int = 80): # 80 is 2 seconds of data
        self.X = torch.tensor(X, dtype=torch.float32)  # Shape: [batch_size, seq_len, features]
        self.Y = torch.tensor(Y, dtype=torch.float32)  # Shape can vary based on task
        self.seq_len = seq_len

    def __len__(self):
        """This method returns the total number of possible 
        sequences that can be generated from X given the specified seq_len."""
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        """This method retrieves a single item from the dataset at a specified index"""
        return (self.X[index:index+self.seq_len], self.Y[index:index+self.seq_len])

class LSTMRegression(nn.Module):
    """
    Class for the long short term memory (LSTM) decoder using PyTorch

    Parameters
    ----------
    input_size: integer
        The number of input features (neurons).

    output_size: integer
        The number of output features.

    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, 
                 input_size, 
                 output_size=1, 
                 hidden_units=400,
                 dropout=0,
                 num_epochs=100,
                 verbose=True):
        super(LSTMRegression, self).__init__()
        
        # Init parameters
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.device = self.get_device()

        # Model Layers
        self.lstm = nn.LSTM(input_size, hidden_units, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_units, output_size)
        
    def get_device(self):
        # Get cpu, gpu or mps device for training.
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if device != "cuda":
            logger.warning("Cuda is not available. Using CPU instead this will take forever fix.")
            
        if device == "cuda":
            logger.success("Using CUDA device")
            
        return device

    def forward(self, x):
        # Forward pass through LSTM layer
        x, _ = self.lstm(x)

        # Forward pass through the fully connected layer
        x = self.fc(x)
        return x
    
    def custom_loss_function(self, predictions, true_values):
        # Flatten the batch and sequence dimensions
        predictions_flat = predictions.view(-1, predictions.size(-1))
        true_values_flat = true_values.view(-1, true_values.size(-1))

        # Create a Von Mises distribution centered at the predicted angles
        concentration_tensor = torch.tensor([1.0], device=self.device) # second parameter is concentration for von mises
        von_mises_dist = dist.VonMises(predictions_flat, concentration_tensor)

        # Calculate the negative log likelihood
        loss = -von_mises_dist.log_prob(true_values_flat)
        return loss.mean()  # Return the average loss over all time steps and batches
    
    def optimizer(self, model):
        return optim.Adam(model.parameters())
    
    @staticmethod
    def train_loop(dataloader, model, loss_fn, optimizer, device):
        """Use backpropagation to train the model, updating the weights and biases
        
        Also known as training the model."""
        model.train() # Set model to training mode
        size = len(dataloader.dataset)
        losses = []
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward() # compute gradients
            optimizer.step() # update weights and biases
            optimizer.zero_grad() # clear gradients
            
            # Save loss
            losses.append(loss.item())

            if batch % 100 == 0:
                print(f'Batch {batch}/{len(dataloader)}, Loss: {loss.item()}')
                
        # Return average loss across batches
        avg_loss = np.mean(losses)
            
        return avg_loss
    
    @staticmethod
    def test_loop(dataloader, model, loss_fn, device):
        model.eval()
        ssr = 0  # Sum of squared residuals
        all_y = []  # To store all actual values for computing the mean

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                residuals = (pred - y) ** 2
                ssr += residuals.sum().item()  # Sum of squared residuals for the batch
                all_y.extend(y.view(-1).tolist())

        mean_y = np.mean(all_y)
        sst = sum([(y_val - mean_y) ** 2 for y_val in all_y])  # Total variance
        r_squared = 1 - (ssr / sst)
        
        print(f"Test R²: {r_squared:>8f}")
        return r_squared

def main(frame_by_cluster_matrix, Y):
    """Main function for running the LSTM model"""
    
    batch_size = 128
    
    # Initialize dataset
    X = frame_by_cluster_matrix
    data = LSTMDataset(X, Y)
    logger.success("Reshaped neural data for LSTM")
    
    # Define training/testing/validation sets
    total_samples = len(data)  # Replace 'your_dataset' with your dataset variable
    train_size = int(total_samples * 0.8)  # 70% of the dataset
    test_size = total_samples - train_size  # Remaining 20% for the test set
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    # Declare model
    model_lstm = LSTMRegression(input_size=frame_by_cluster_matrix.shape[1])
    model_lstm.to(model_lstm.device) # Move model to GPU if available
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    plot_losses = []
    plot_r2 = []

    for t in range(1, model_lstm.num_epochs + 1):
        print(f"Epoch {t}\n-------------------------------")
        losses = LSTMRegression.train_loop(dataloader = training_loader,
                                            model = model_lstm,
                                            loss_fn = nn.MSELoss(),
                                            # loss_fn = model_lstm.custom_loss_function,
                                            optimizer = model_lstm.optimizer(model_lstm),
                                            device = model_lstm.device)
        plot_losses.append(losses)
        r2 = LSTMRegression.test_loop(dataloader = test_loader,
                                 model = model_lstm,
                                 loss_fn = nn.MSELoss(),
                                 device = model_lstm.device)
        plot_r2.append(r2)
        
    # Plot losses across batches in one subplot and R² across epochs in another
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(plot_losses, label='Training Loss')
    ax2.plot(plot_r2, label='Test R²')
    plt.legend()
    plt.show()
    x = 10
    
    # # Predict on validation set
    # y_valid_predicted = model_lstm.predict(X_valid_tensor)
    
    # # Visualization (if needed)
    # # For example, plotting actual vs predicted head directions
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_valid, label='Actual Head Direction')
    # plt.plot(y_valid_predicted, label='Predicted Head Direction', alpha=0.7)
    # plt.title('Head Direction: Actual vs Predicted')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Head Direction')
    # plt.legend()
    # plt.show()

    
    