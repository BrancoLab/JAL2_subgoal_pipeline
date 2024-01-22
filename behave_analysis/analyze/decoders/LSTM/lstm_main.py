"""An LSTM decoder for head direction data. 
Input is a matrix of neural data, output is a vector of head direction data."""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from behave_analysis.analyze.decoders.LSTM.custom_dataset import LSTMDataset


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

    def __init__(
        self,
        input_size,
        output_size=1,
        hidden_units=128,
        num_epochs=40,  # 100 for e.g
    ):
        super(LSTMRegression, self).__init__()

        # Init parameters
        self.hidden_units = hidden_units
        self.num_epochs = num_epochs
        self.device = self._get_device()
        self.concentration = nn.Parameter(torch.tensor([1.0]))  # starting value as 1.0 for von mises

        # Model Layers
        # self.lstm = nn.LSTM(input_size, hidden_units, num_layers=2, batch_first=True, dropout=0.2) # bigger model
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)

    def _get_device(self) -> str:
        """Ensure GPU is used if available else use CPU"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # concentration_tensor = torch.tensor([1.0], device=self.device)  # Set concentration to 1.0 and don't learn it
        von_mises_dist = dist.VonMises(predictions_flat, self.concentration) # learn concentration as well

        # Calculate the negative log likelihood
        loss = -von_mises_dist.log_prob(true_values_flat)
        return loss.mean()  # Return the average loss over all time steps and batches

    def optimizer(self, model):
        return optim.Adam(model.parameters(), lr=0.001)

    @staticmethod
    def train_loop(dataloader, model, loss_fn, optimizer, device):
        """Use backpropagation to train the model, updating the weights and biases

        Also known as training the model."""
        model.train()  # Set model to training mode
        losses = []
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()  # compute gradients
            optimizer.step()  # update weights and biases
            optimizer.zero_grad()  # clear gradients

            # Save loss
            losses.append(loss.item())

            if batch % 100 == 0:
                print(f"Batch {batch}/{len(dataloader)}, Loss: {loss.item()}")

        # Return average loss across batches
        avg_loss = np.mean(losses)

        return avg_loss

    @staticmethod
    def test_loop(dataloader, model, loss_fn, device):
        model.eval()
        num_batches = len(dataloader)
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                # residuals = (pred - y) ** 2
                # ssr += residuals.sum().item()  # Sum of squared residuals for the batch
                # all_y.extend(y.view(-1).tolist())
                test_loss += loss_fn(pred, y).item()

        # Provide avg test loss across all batches for that epoch
        test_loss /= num_batches

        print("---------------------------")
        print(f"Test loss: {test_loss:>8f}")
        return test_loss

    @staticmethod
    def predict(model, dataloader, device):
        """Predict the output for a given input"""
        model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                predictions.append(pred.cpu())  # Move predictions to CPU
        predictions = torch.cat(predictions, dim=0)  # Concatenate all predictions into one tensor
        return predictions


def main(frame_by_cluster_matrix, Y):
    """Main function for running the LSTM model"""

    # Set hyperparameters
    batch_size = 128

    # Initialize dataset
    X = frame_by_cluster_matrix
    norm_y = Y / np.pi  # Divide by pi to get values between -1 and 1
    data = LSTMDataset(X, norm_y)
    logger.success("Reshaped neural data for LSTM")

    # Define training/testing/ sets
    total_samples = len(data)
    train_size = int(total_samples * 0.8)  # 80% of the dataset

    # Manually split the dataset to preserve temporal order
    train_dataset = Subset(data, range(0, train_size))
    test_dataset = Subset(data, range(train_size, total_samples))

    # Declare model
    model_lstm = LSTMRegression(input_size=frame_by_cluster_matrix.shape[1])
    model_lstm.to(model_lstm.device)  # Move model to GPU if available
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    plot_losses = []
    plot_test_losses = []

    for t in range(1, model_lstm.num_epochs + 1):
        print(f"Epoch {t}\n-------------------------------")
        losses = LSTMRegression.train_loop(
            dataloader=training_loader,
            model=model_lstm,
            # loss_fn=nn.MSELoss(),
            loss_fn=model_lstm.custom_loss_function,
            optimizer=model_lstm.optimizer(model_lstm),
            device=model_lstm.device,
        )
        plot_losses.append(losses)
        test_losses = LSTMRegression.test_loop(
            dataloader=test_loader,
            model=model_lstm,
            loss_fn=model_lstm.custom_loss_function,
            # loss_fn=nn.MSELoss(),
            device=model_lstm.device,
        )
        plot_test_losses.append(test_losses)

    # Plot losses for test and train across epochs
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    ax1.plot(plot_losses, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss across epochs")

    ax2.plot(plot_test_losses, label="Test Losses")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test loss loss across epochs")

    plt.suptitle("Training vs Test Losses for LSTM")
    plt.legend()
    plt.show()

    # Now predict ---------------------------------------------------------------
    actual_labels = np.asarray(test_dataset.dataset.Y).reshape(-1)
    predictions = model_lstm.predict(model=model_lstm, dataloader=test_loader, device=model_lstm.device).numpy()
    flat_predictions = predictions.reshape(-1)
    flat_predictions = flat_predictions[: len(test_dataset.dataset.Y)]

    plt.figure(figsize=(10, 6))
    plt.plot(flat_predictions * np.pi, label="Predicted", linewidth=2)
    plt.plot(actual_labels * np.pi, label="Actual", linewidth=2)  # Multiply by pi to get values between -pi and pi
    plt.title("Predictions vs Actual Labels for LSTM: {model_lstm.num_epochs} epochs}")
    plt.xlabel("Sample Index")
    plt.ylabel("Head Direction")
    plt.legend()
    plt.show()
