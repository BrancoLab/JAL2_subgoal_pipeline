"""
This module contains a PyTorch implementation of an LSTM model for regression tasks, 
specifically designed to handle sequences of data for predicting a single output dimension. 
The LSTM model architecture is defined, along with functions for training the model on a given dataset and 
reshaping input data to fit the sequential nature of the model.

Performance estimations indicate that training on 60k samples with a sequence length of 5 and 100 hidden units 
at a learning rate of 0.005 takes approximately 5 minutes.

TODO:
- Replace operations with eigensum for more efficient dimension changes.

Modules and Libraries:
- numpy: Used for handling arrays and mathematical operations.
- torch: The main PyTorch library for deep learning model definition, training, and inference.
- torch.nn: Module for defining layers and models in PyTorch.
- torch.optim: Provides optimization algorithms for training.
- torch.cuda.amp: Utilities for mixed-precision training.
- sklearn.metrics: For calculating the R^2 score to evaluate model performance.
- sklearn.model_selection: For splitting the dataset into training and testing sets.
- matplotlib.pyplot: For plotting training and testing losses, and comparing predicted values with actual values.
- loguru: For logging information and debugging.

Classes and Functions:
- LSTMModel: A class that defines the LSTM model architecture.
- run_LSTM: A function that trains the LSTM model on input data X and Y, plots the learning curve, and evaluates the model performance on both training and testing sets.
- reshape_sequences_1d: A utility function for reshaping input data into sequences suitable for training the LSTM model, ensuring that the number of 
predictions matches the input size without data leakage.

Usage:
This module is intended for use in regression tasks where the data is sequential, and the goal is to predict a 
single output dimension based on a series of input features. It is particularly useful for time-series analysis or any 
scenario where the relationship between sequential inputs and a target variable needs to be modeled. It expects the X data to be in the shape 
(num_samples, num_features) and the Y data to be in the shape (num_samples, ).

Example Usage Problem: Decode head direction from neural data
For neural decoding, typically we work with behavioural data and neural data. And thus each sample can be a spike count of a given neuron in a given video frame.
Where a feature, for example, refers to the cluster index of a neuron and the target Y is the angle of the animal's head at that frame. 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from loguru import logger


# -------------------------- LSTM Model Architecture --------------------------
class LSTMModel(nn.Module):
    """
    Defines an LSTM model architecture for regression tasks with a single output dimension.

    This model is designed to process sequential data and predict a single output dimension. It uses an LSTM layer followed by a linear
    layer to map the hidden states to the output.

    Attributes:
    - input_dim (int): The number of features in the input data.
    - hidden_dim (int): The number of features in the hidden state of the LSTM.
    - num_layers (int): The number of stacked LSTM layers.
    - device (str): The device ('cuda' or 'cpu') the model is running on.

    Methods:
    - forward(x): Defines the forward pass of the model.

    Args:
    - input_dim (int): Dimensionality of the input features.
    - hidden_dim (int): Number of hidden units in the LSTM layer.
    - num_layers (int): Number of LSTM layers stacked together.

    Returns:
    - The LSTMModel instance.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        _output_dim = 1
        self.hidden_dim = hidden_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, _output_dim)

        if self.device == "cuda":
            logger.info("Using GPU")
        else:
            logger.info("Using CPU")

    def forward(self, x):
        """Forward pass through the network.

        Note that the output extracts the last value from each of the sequences but because the X
        sequences are overlapping the len of input Y and predicted Y will be the same. Check the function
        reshape_sequences_1d for more details.

        Args:
        - x (torch.Tensor): Input data with shape (num_samples, sequence_length, input_dim).]

        Returns:
        - out (torch.Tensor): Output data with shape (num_samples, output_dim) - I.e the prediction for each sequence."""

        # x.size(0) represents the batch size, i.e., the number of sequences to consider before updating the internal parameters
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)  # initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)  # initial cell state
        out, (_, _) = self.lstm(x, (h0, c0))  # where out.shape = (Batch size, sequence length, hidden_dimens)
        # Given that we are using batch gradient descent batch size == total number of samples
        # Now Squash the output of the hidden to the output layer (Samples, output_dim)
        out = self.linear(out[:, -1, :])  # Take the last sequence output
        # Shape of out is now (Samples, output_dim)
        out = torch.tanh(out) * 3.14  # scale the output to be between -pi and pi
        return out


# ---------------------------- fit the model ----------------------------
def fit_LSTM(X, Y) -> tuple[object, int]:
    """Returns a fitted LSTM model for the given training input data.

    This function expects training data as the X and Y input.

    Args:
    - X (numpy.ndarray): Input features with shape (num_samples, num_features), where num_samples is the number of samples and num_features is the number of features.
    - Y (numpy.ndarray): Corresponding labels with shape (num_samples,), where each entry is the target value for the corresponding sample in X.

    Returns:
    -- model (object): A fitted LSTM model.
    -- sequence_length (int): The sequence length used for training the model required for predicted reshaping
    """

    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters ---------------------------------------------------
    test_size = 0.2
    num_layers = 1  # Increasing also gives better results but is slower
    input_dim = X.shape[1]
    epoch_saves = 50  # Save and print the loss every 50 epochs

    # Hyperparameters ----------------------------------------------
    hidden_dim = 100  # Number of LSTM cells - 100 worrks but 256 is better though much slower
    num_epochs = 250  # Number of iterations to train the model - increasing this gives better results but is slower
    sequence_length = 5  # Number of time steps to consider for each prediction - increasing this gives better results but is slower
    # and because not mini-batching you will get an error if model becomes to big
    learning_rate = 0.005  # 0.001 works better but is slower

    # Preprocess the data -------------------------------------------
    X_train_torch, Y_train_torch = reshape_sequences_1d(
        X,
        Y,
        seq_length=sequence_length,
    )

    # Create the LSTM model
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Init --------------------------------------------------------
    train_losses = []
    test_losses = []

    # Initialize scaler for mixed precision
    scaler = GradScaler()

    # Training the model ----------------------------------------------
    logger.info("Training the model")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Clear the gradients

        # Automated mixed precision training for faster training ----------------
        with autocast():
            y_pred_for_loss = model(X_train_torch)  # Forward pass to get the predictions
            y_pred_for_loss = torch.squeeze(y_pred_for_loss)  # Remove the extra dimension
            assert y_pred_for_loss.shape == Y_train_torch.shape, "The shapes of the model output must match the target for the loss function to work"
            train_loss = loss_fn(y_pred_for_loss, Y_train_torch)  # Compute the loss
        scaler.scale(train_loss).backward()  # Backward pass to compute the gradients
        scaler.step(optimizer)  # Update the parameters
        scaler.update()

        # logg the loss
        logger.info(f"Epoch {epoch}, Train Loss: {train_loss.item()}")

    return model, sequence_length


# ---------------------------- predict with the model ----------------------------
def predict_LSTM(model, X, sequence_length) -> np.ndarray:
    """Returns the predictions of the LSTM model for the given input data.

    This function expects a fitted LSTM model and input Test data X for prediction.

    Args:
    - model (object): A fitted LSTM model.
    - X (numpy.ndarray): Input features with shape (num_samples, num_features), where num_samples is the number of samples and num_features is the number of features.

    Returns:
    - y_pred (numpy.ndarray): Predicted labels with shape (num_samples,), where each entry is the target value for the corresponding sample in X.
    """

    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess the data -----------------------------------------------------
    Y = np.array([0] * len(X))  # Dummy Y values for reshaping
    X_test_torch, _ = reshape_sequences_1d(X, Y, seq_length=sequence_length)

    # Predicting Y with the trained model -------------------------------------
    with torch.no_grad():
        y_pred = model(X_test_torch).cpu().numpy()

    return y_pred


# -------------------------- Shape the input Data --------------------------


def reshape_sequences_1d(X, Y, seq_length, device="cuda"):
    """
    Reshape the input X, Y for sequence modeling, predicting once per sequence.

    The X segments do overlap, but the Y labels are not used for the overlapping segments.
    This is to avoid data leakage whilst maintaining the total number of predictions matches the input.

    Args:
    - X (numpy.ndarray): Input features with shape (num_samples, num_features).
    - Y (numpy.ndarray): Corresponding labels with shape (num_samples, ).
    - seq_length (int): Desired sequence length for the LSTM inputs.

    - device (str): The device to use ('cpu' or 'cuda').

    Returns:
    - X_torch, Y_torch,
    """

    # Calculate the number of sequences
    num_sequences = len(X) - seq_length + 1

    # Initialize the reshaped data arrays
    X_reshaped = np.zeros((num_sequences, seq_length, X.shape[1]))
    Y_reshaped = np.zeros((num_sequences))

    # Fill the reshaped data arrays
    for i in range(num_sequences):
        X_reshaped[i] = X[i : i + seq_length]  # Select a view of shape (seq_length, num_features) where seqennces overlap
        Y_reshaped[i] = Y[i + seq_length - 1]  # Select the label corresponding to the end of the sequence

    # Convert to PyTorch tensors and add necessary dimensions
    X_torch = torch.tensor(X_reshaped, dtype=torch.float32).to(device)
    Y_torch = torch.tensor(Y_reshaped, dtype=torch.float32).to(device)

    return X_torch, Y_torch
