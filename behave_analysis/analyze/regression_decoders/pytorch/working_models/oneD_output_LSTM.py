"""

Performance estimations:
- 5 minutes to train 60k samples on 1d output LSTM with sequence length of 5 and 100 hidden units lr = 0.005
- So for 150 points this should take 12.5 hours

#TODO
-- Replace with eigensum for dimension changes
-- Figure out speed of training

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


class LSTMModel(nn.Module):
    """An RNN model for regression with a single output dimenion"""

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


def run_LSTM(X, Y):
    """Run a LSTM modelf for regression with a single output.

    Args:
    - X (numpy.ndarray): Input features with shape (num_samples, num_features).
    - Y (numpy.ndarray): Corresponding labels with shape (num_samples, )."""

    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters ---------------------------------------------------
    test_size = 0.2
    num_layers = 1
    input_dim = X.shape[1]
    epoch_saves = 50 # Save and print the loss every 50 epochs

    # Hyperparameters ----------------------------------------------
    hidden_dim = 100  # Number of LSTM cells
    num_epochs = 250 # Number of iterations to train the model
    sequence_length = 5  # Number of time steps to consider for each prediction

    # Preprocess the data -------------------------------------------
    X_train_torch, X_test_torch, Y_train_torch, Y_test_torch = reshape_sequences_1d(X, Y, seq_length=sequence_length, test_size=test_size)

    # Create the LSTM model
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005) 

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
            train_loss = loss_fn(y_pred_for_loss, Y_train_torch) # Compute the loss
        scaler.scale(train_loss).backward()  # Backward pass to compute the gradients
        scaler.step(optimizer) # Update the parameters
        scaler.update()

        # Track the losses ---------------------------------------------
        if epoch % epoch_saves == 0:
            train_losses.append(train_loss.item())
            # ---------------------Track the test loss ----------------
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Turn off the gradients
                y_pred_test = model(X_test_torch)
                y_pred_test = torch.squeeze(y_pred_test)
                test_loss = loss_fn(y_pred_test, Y_test_torch)
                test_losses.append(test_loss.item())
                print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")

    # Predicting Y with the trained model -------------------------------------
    with torch.no_grad():
        y_pred_test = model(X_test_torch).cpu().numpy()
        y_pred_train = model(X_train_torch).cpu().numpy()

    # Learning Curve (now spans two columns)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, num_epochs, epoch_saves), train_losses, label="Training Loss", color="blue")
    plt.plot(np.arange(0, num_epochs, epoch_saves), test_losses, label="Test Loss", color="orange")
    plt.title(f"Learning Curves - Min train loss: {min(train_losses):.2f}, Min test loss: {min(test_losses):.2f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Training data predictions vs actual values
    plt.subplot(2, 2, 3)
    plt.plot(y_pred_train[:500], label="Predicted Train", linewidth=2)
    plt.plot(Y_train_torch.cpu().numpy()[:500], label="Actual Train", linewidth=2)
    train_r2 = r2_score(Y_train_torch.cpu().numpy(), y_pred_train)
    plt.xlabel("Frame")
    plt.ylabel("Angle")
    plt.title(f"LSTM Model Training Set w r2: {train_r2}")
    plt.legend()

    # Test fit
    plt.subplot(2, 2, 4)
    plt.plot(y_pred_test[:500], label="Predicted Test", linewidth=2)
    plt.plot(Y_test_torch.cpu().numpy()[:500], label="Actual labels", linewidth=2)
    test_r2 = r2_score(Y_test_torch.cpu().numpy(), y_pred_test)
    plt.xlabel("Frame")
    plt.ylabel("Angle")
    plt.title(f"LSTM Model Test Set w r2: {test_r2}")
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------------- LSTM Helper Functions --------------------------


def reshape_sequences_1d(X, Y, seq_length, test_size=0.2, random_state=42, device="cuda"):
    """
    Reshape the input X, Y for sequence modeling, predicting once per sequence.

    The X segments do overlap, but the Y labels are not used for the overlapping segments.
    This is to avoid data leakage whilst maintaining the total number of predictions matches the input.

    Args:
    - X (numpy.ndarray): Input features with shape (num_samples, num_features).
    - Y (numpy.ndarray): Corresponding labels with shape (num_samples, ).
    - seq_length (int): Desired sequence length for the LSTM inputs.
    - test_size (float): Fraction of the dataset to be used as test set.
    - random_state (int): Seed for the random number generator.
    - device (str): The device to use ('cpu' or 'cuda').

    Returns:
    - X_train_torch, X_test_torch, y_train_torch, y_test_torch: Reshaped and split data, as PyTorch tensors.
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

    # Split the reshaped data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_reshaped, Y_reshaped, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors and add necessary dimensions
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32).to(device)

    return X_train_torch, X_test_torch, Y_train_torch, Y_test_torch
