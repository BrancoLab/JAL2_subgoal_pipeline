"""This script runs a working ugly code LSTM data on buzacki data.
The only thing that needs to be changed is the path to the data.
This is a sense check that I can get a working LSTM model before refactoring and making 
the model bigger and better.

- First built 1d output with sequence length of 1 no batch - worked
- Added sequence length which broke as memory error
- Now need to batch


Batch Size: Defines the number of samples for the model to work through before updating it's internal parameters
Epoch: Defines the number times that the learning algorithm will work through the entire training dataset.

Functions:
- unsqueeze: Adds an extra dimension to the tensor
- squeeze: Removes a dimension from the tensor

x = torch.rand(5, 10).shape
print(x) -> (5, 10)
print(x.unsqueeze(0).shape) -> (1, 5, 10)
print(x.unsqueeze(1).shape) -> (5, 1, 10)
print(x.unsqueeze(-1).shape) -> (5, 10, 1)

#NOTE
-- Replace with eigensum for dimension changes
-- 

"""

from matplotlib.gridspec import GridSpec

import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from behave_analysis.analyze.regression_decoders.pytorch.custom_dataset import LSTMDataset, LSTMDataset2
from torch.utils.data import Subset

# General Functions


def check_if_cuda_device_is_available():
    # Ensure GPU is used if available else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("Cuda is not available. Using CPU instead this will take forever fix.")
    if device == "cuda":
        logger.success("Using CUDA device")
    return device


def test_1d(X, Y):
    print("This is a test")
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
            # x.size(0): This represents the batch size, i.e., the number of sequences in the batch
            # #that will be processed by the LSTM. It's derived from the first dimension of the input
            # tensor x, which is the batch size.
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
            out, _ = self.lstm(x, (h0, c0))  # Shape (?, 1, hidden_dimens)
            out = self.linear(out[:, -1, :])  # Take the last sequence output
            out = torch.tanh(out) * 3.14  # scale the output to be between -pi and pi
            return out

    # Preprocess the data -------------------------------------------

    Y_reshaped = np.asarray(Y).reshape(len(Y), 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_reshaped, test_size=0.2, random_state=42)

    # Convert training and test sets to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Hyperparameters
    input_dim = X.shape[1]
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


def test_1d_with_batch(X, Y):
    logger.info("This is a test to get a dataloader working with batching")

    # Check if GPU is available
    device = check_if_cuda_device_is_available()

    # Hyperparameters
    input_dim = X.shape[1]  # Number of literal biological neurons
    hidden_dim = 100  # Number of LSTM cells
    num_layers = 1
    output_dim = 1
    batch_size = 32  # How many sequences are fed into the model each training step
    sequence_lenght = 1
    num_epochs = 1

    # Load and preprocess the data
    data = LSTMDataset(X, Y, seq_len=sequence_lenght)
    total_samples = len(data)
    train_size = int(total_samples * 0.95)  # 95% of the dataset is used for training
    train_dataset = Subset(data, range(0, train_size))
    test_dataset = Subset(data, range(train_size, total_samples))

    # Dataloader ingest a pytorch dataset class and returns an iterator for batching
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Creat the LSTM model
    model = LSTMModel1Sequence(input_dim, hidden_dim, output_dim, num_layers, device).to(device)
    logger.debug(model)  # Print the model architecture
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # -- init loss
    # train_losses = []
    # test_losses = []

    # -------------------------- Training the model ----------------------------------
    for epoch in range(num_epochs):
        model.train()
        # train_loss = 0.0
        for batch, (X_batch, Y_batch) in enumerate(training_loader):
            X, y = X_batch.to(device), Y_batch.to(device)  # Send data to GPU
            optimizer.zero_grad()  # Zero the gradients
            output = model(X)
            # y = y.unsqueeze(-1)  # Adds an extra dimension, making it (batch_size, seq_len, 1)
            assert output.shape == y.shape, "The shapes of the model output must match the target for the loss function to work"
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            # train_loss += loss.item() * X.size(0)
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(training_loader)}")

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     logger.info(f"Epoch: {epoch + 1} loss: {loss:>7f}  [{current:>5d}/{len(training_loader.dataset)}]")

        # train_loss /= len(training_loader.dataset)
        # train_losses.append(train_loss)

    # ------------------------ Testing the model -------------------------------------

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    predictions, labels = [], []

    with torch.no_grad():  # Disable gradient computation
        for X_batch, Y_batch in test_loader:
            X_test, y_test = X_batch.to(device), Y_batch.to(device)
            output_test = model(X_test)
            # y_test_squeesh = y_test.unsqueeze(-1)
            loss_test = loss_fn(output_test, y_test_squeesh)
            test_loss += loss_test.item() * X_test.size(0)  # Multiply by batch size to scale loss
    test_loss /= len(test_loader.dataset)  # Average the test loss over all test data
    test_losses.append(test_loss)
    logger.info(f"Test Loss: {test_loss:>7f}")

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Top left: Training loss over time
    # axs[0, 0].plot(train_losses, label='Training Loss')
    # axs[0, 0].set_title('Training Loss Over Time')
    # axs[0, 0].set_xlabel('Epoch')
    # axs[0, 0].set_ylabel('Loss')
    # axs[0, 0].legend()

    # # Top right: Test loss over time
    # axs[0, 1].plot(test_losses, label='Test Loss', color='orange')
    # axs[0, 1].set_title('Test Loss Over Time')
    # axs[0, 1].set_xlabel('Epoch')
    # axs[0, 1].set_ylabel('Loss')
    # axs[0, 1].legend()

    # Bottom left: Predicted Y vs. Actual Y (Training)
    # Assuming the first batch is large enough; adjust as needed.
    X_train, y_train = next(iter(training_loader))
    X_train, y_train = X_train.to(device), y_train.to(device)
    with torch.no_grad():
        y_pred_train = model(X_train).cpu().numpy()
    # y_train = y_train[:100].cpu().numpy()
    y_train = y_train.reshape(-1).cpu().numpy()
    y_pred_train = y_pred_train.reshape(-1)
    train_r2 = r2_score(y_train, y_pred_train)
    axs[1, 0].plot(y_train, label="Actual Y", marker="o")
    axs[1, 0].plot(y_pred_train, label="Predicted Y", linestyle="--", marker="x")
    axs[1, 0].set_title(f"Predicted vs. Actual Y (Training) - R2: {train_r2}")
    axs[1, 0].set_xlabel("Sample")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].legend()

    # # Bottom right: Predicted Y vs. Actual Y (Test)
    # X_test, y_test = next(iter(test_loader))
    # X_test, y_test = X_test.to(device), y_test.to(device)
    # with torch.no_grad():
    #     y_pred_test = model(X_test[:100]).cpu().numpy()
    # y_test = y_test[:100].cpu().numpy()
    # test_r2 = r2_score(y_test, y_pred_test)
    # axs[1, 1].plot(y_test, label="Actual Y", marker="o")
    # axs[1, 1].plot(y_pred_test, label="Predicted Y", linestyle="--", marker="x")
    # axs[1, 1].set_title("Predicted vs. Actual Y (Test) - R2: {test_r2}")
    # axs[1, 1].set_xlabel("Sample")
    # axs[1, 1].set_ylabel("Value")
    # axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


def reshape_sequences_1d(X, Y, seq_length, test_size=0.2, random_state=42, device="cuda"):
    """
    Reshape the input data X and labels Y for sequence modeling, predicting every time step.

    Args:
    - X (numpy.ndarray): Input features with shape (num_samples, num_features).
    - Y (numpy.ndarray): Corresponding labels with shape (num_samples, ).
    - seq_length (int): Desired sequence length for the LSTM inputs.
    - test_size (float): Fraction of the dataset to be used as test set.
    - random_state (int): Seed for the random number generator.
    - device (str): The device to use ('cpu' or 'cuda').

    Returns:
    - X_train_torch, X_test_torch, y_train_torch, y_test_torch: Reshaped and split data, as PyTorch tensors.

    To avoid data leakage one prediction is made for each sequence. I also can't figure it out another way.
    """

    # Calculate the number of sequences
    num_sequences = len(X) - seq_length + 1

    # Initialize the reshaped data arrays
    X_reshaped = np.zeros((num_sequences, seq_length, X.shape[1]))
    Y_reshaped = np.zeros((num_sequences))

    # Fill the reshaped data arrays
    for i in range(num_sequences):
        X_reshaped[i] = X[i : i + seq_length]
        Y_reshaped[i] = Y[i + seq_length - 1]  # Select the label corresponding to the end of the sequence

    # Split the reshaped data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_reshaped, Y_reshaped, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors and add necessary dimensions
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32).to(device)

    return X_train_torch, X_test_torch, Y_train_torch, Y_test_torch


def reshape_sequences_2d(X, Y, seq_length, test_size=0.2, random_state=42, device="cuda"):
    """
    Reshape the input data X and labels Y for sequence modeling, predicting every time step.

    Args:
    - X (numpy.ndarray): Input features with shape (num_samples, num_features).
    - Y (numpy.ndarray): Corresponding labels with shape (num_samples, 2).
    - seq_length (int): Desired sequence length for the LSTM inputs.
    - test_size (float): Fraction of the dataset to be used as test set.
    - random_state (int): Seed for the random number generator.
    - device (str): The device to use ('cpu' or 'cuda').

    Returns:
    - X_train_torch, X_test_torch, y_train_torch, y_test_torch: Reshaped and split data, as PyTorch tensors.

    To avoid data leakage one prediction is made for each sequence. I also can't figure it out another way.
    """

    # Calculate the number of sequences
    num_sequences = len(X) - seq_length + 1

    # Initialize the reshaped data arrays
    X_reshaped = np.zeros((num_sequences, seq_length, X.shape[1]))
    Y_reshaped = np.zeros((num_sequences, Y.shape[1]))

    # Fill the reshaped data arrays
    for i in range(num_sequences):
        X_reshaped[i] = X[i : i + seq_length]
        Y_reshaped[i] = Y[i + seq_length - 1].to_numpy().flatten()  # Select the label corresponding to the end of the sequence

    # Split the reshaped data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_reshaped, Y_reshaped, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors and add necessary dimensions
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32).to(device)

    return X_train_torch, X_test_torch, Y_train_torch, Y_test_torch


# ----------------------------------- WORKING TEST MODELS -----------------------------------------


def test_1d_no_batch_with_seq(X, Y):
    print("This is a working sequence model")
    device = check_if_cuda_device_is_available()

    # Define LSTM model ----------------------------------------------
    class LSTMModel(nn.Module):
        """_summary_

        Architecture:
        -- Linear layer: Squashes the output of the LSTM to a continuous value for regression

        Args:
            nn (_type_): _description_
        """

        def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x.size(0): This represents the batch size, i.e., the number of sequences in the batch
            # #that will be processed by the LSTM. It's derived from the first dimension of the input
            # tensor x, which is the batch size.
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # initial hidden state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # initial cell state
            out, (_, _) = self.lstm(x, (h0, c0))  # Shape (num_samples, 1, hidden_dimens)
            # Now Squash the output of (Samples, sequence_lenght, hidden_dimens) to (Samples, output_dim)
            out = self.linear(out[:, -1, :])  # Take the last sequence output
            out = torch.tanh(out) * 3.14  # scale the output to be between -pi and pi
            return out

    # Hyperparameters
    input_dim = X.shape[1]
    hidden_dim = 100  # Number of LSTM cells
    num_layers = 1
    output_dim = 1
    num_epochs = 500
    sequence_length = 7
    epoch_saves = 50

    # Preprocess the data -------------------------------------------
    X_train_torch, X_test_torch, Y_train_torch, Y_test_torch = reshape_sequences_1d(
        X, Y, seq_length=sequence_length, test_size=0.2, random_state=42, device="cuda"
    )

    # Create the LSTM model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Init --------------------------------------------------------
    train_losses = []
    test_losses = []

    # Training the model ----------------------------------------------
    logger.info("Training the model")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_for_loss = model(X_train_torch)
        y_pred_for_loss = torch.squeeze(y_pred_for_loss)
        assert y_pred_for_loss.shape == Y_train_torch.shape, "The shapes of the model output must match the target for the loss function to work"
        train_loss = loss_fn(y_pred_for_loss, Y_train_torch)
        train_loss.backward()
        optimizer.step()
        if epoch % epoch_saves == 0:
            train_losses.append(train_loss.item())
            # Compute the test loss
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test_torch)
                y_pred_test = torch.squeeze(y_pred_test)
                test_loss = loss_fn(y_pred_test, Y_test_torch)
                test_losses.append(test_loss.item())
                print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")

    # Predicting Y with the trained model -------------------------------------
    with torch.no_grad():
        y_pred_test = model(X_test_torch).cpu().numpy()
        y_pred_train = model(X_train_torch).cpu().numpy()

    fig, ax = plt.subplots(figsize=(15, 6), nrows=2, ncols=2, gridspec_kw={"height_ratios": [1, 1], "width_ratios": [1, 1]})

    # Remove the second subplot on the first row to make space for the learning curve
    fig.delaxes(ax[0][1])

    # Learning Curve (now spans two columns)
    ax[0][0].plot(np.arange(0, num_epochs, epoch_saves), train_losses, label="Training Loss", color="blue")
    ax[0][0].plot(np.arange(0, num_epochs, epoch_saves), test_losses, label="Test Loss", color="orange")
    ax[0][0].set_title(f"Learning Curves - Min train loss: {min(train_losses):.2f}, Min test loss: {min(test_losses):.2f}")
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Loss")
    ax[0][0].legend(loc="upper right")

    # Training data predictions vs actual values
    ax[1][0].plot(y_pred_train[:500], label="Predicted Train", linewidth=2)
    ax[1][0].plot(Y_train_torch.cpu().numpy()[:500], label="Actual Train", linewidth=2)
    train_r2 = r2_score(Y_train_torch.cpu().numpy(), y_pred_train)
    ax[1][0].set_title(f"LSTM Model Training Set w r2: {train_r2}")
    ax[1][0].legend()

    # Test fit
    ax[1][1].plot(y_pred_test[:500], label="Predicted Test", linewidth=2)
    ax[1][1].plot(Y_test_torch.cpu().numpy()[:500], label="Actual labels", linewidth=2)
    test_r2 = r2_score(Y_test_torch.cpu().numpy(), y_pred_test)
    ax[1][1].set_title(f"LSTM Model Test Set w r2: {test_r2}")
    ax[1][1].legend()

    plt.tight_layout()
    plt.show()


def multi_output_test(X, Y):
    """Test to see if I can get a multi output model working"""

    device = check_if_cuda_device_is_available()

    # Define LSTM model ----------------------------------------------
    class LSTMModel(nn.Module):
        """_summary_

        Architecture:
        -- Linear layer: Squashes the output of the LSTM to a continuous value for regression

        Args:
            nn (_type_): _description_
        """

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x.size(0): This represents the batch size, i.e., the number of sequences in the batch
            # #that will be processed by the LSTM. It's derived from the first dimension of the input
            # tensor x, which is the batch size.
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # initial hidden state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # initial cell state
            out, (_, _) = self.lstm(x, (h0, c0))  # Shape (num_samples, 1, hidden_dimens)
            # Now Squash the output of (Samples, sequence_lenght, hidden_dimens) to (Samples, output_dim)
            out = self.linear(out[:, -1, :])  # Take the last sequence output
            out = torch.tanh(out) * 3.14  # scale the output to be between -pi and pi
            return out

    # Hyperparameters
    input_dim = X.shape[1]
    num_features = Y.shape[1]
    hidden_dim = 100  # Number of LSTM cells
    num_layers = 1
    output_dim = num_features
    num_epochs = 500
    sequence_length = 2  # 7 works best so far but hit memory error for above that
    epoch_saves = 50

    # Preprocess the data -------------------------------------------
    X_train_torch, X_test_torch, Y_train_torch, Y_test_torch = reshape_sequences_2d(
        X, Y, seq_length=sequence_length, test_size=0.2, random_state=42, device="cuda"
    )

    # Create the LSTM model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Init --------------------------------------------------------
    train_losses = []
    test_losses = []

    # Training the model ----------------------------------------------
    logger.info("Training the model")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_for_loss = model(X_train_torch)
        # y_pred_for_loss = torch.squeeze(y_pred_for_loss)
        # IF shapes are not identical the loss function may broadcast and create memory errors
        assert y_pred_for_loss.shape == Y_train_torch.shape, "The shapes of the model output must match the target for the loss function to work"
        train_loss = loss_fn(y_pred_for_loss, Y_train_torch)
        train_loss.backward()
        optimizer.step()
        if epoch % epoch_saves == 0:
            train_losses.append(train_loss.item())
            # Compute the test loss
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test_torch)
                y_pred_test = torch.squeeze(y_pred_test)
                test_loss = loss_fn(y_pred_test, Y_test_torch)
                test_losses.append(test_loss.item())
                print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")

    # Predicting Y with the trained model -------------------------------------
    with torch.no_grad():
        y_pred_test = model(X_test_torch).cpu().numpy()
        y_pred_train = model(X_train_torch).cpu().numpy()

    height_ratios = [1] * (num_features + 1)  # Creates a list with equal height for each row

    fig, ax = plt.subplots(figsize=(15, 6), nrows=num_features + 1, ncols=2, gridspec_kw={"height_ratios": height_ratios, "width_ratios": [1, 1]})

    # Remove the second subplot on the first row to make space for the learning curve
    fig.delaxes(ax[0][1])

    # Learning Curve (now spans two columns)
    ax[0][0].plot(np.arange(0, num_epochs, epoch_saves), train_losses, label="Training Loss", color="blue")
    ax[0][0].plot(np.arange(0, num_epochs, epoch_saves), test_losses, label="Test Loss", color="orange")
    ax[0][0].set_title(f"Learning Curves - Min train loss: {min(train_losses):.2f}, Min test loss: {min(test_losses):.2f}")
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Loss")
    ax[0][0].legend(loc="upper right")

    # Training data predictions vs actual values for each dimension
    for dim in range(num_features):  # Assuming 2-dimensional y
        ax[dim + 1][0].plot(y_pred_train[:500, dim], label=f"Predicted Train Dim {dim+1}", linewidth=2)
        ax[dim + 1][0].plot(Y_train_torch.cpu().numpy()[:500, dim], label=f"Actual Train Dim {dim+1}", linewidth=2)
        train_r2 = r2_score(Y_train_torch.cpu().numpy()[:, dim], y_pred_train[:, dim])
        ax[dim + 1][0].set_title(f"LSTM Model Training Set Dim {dim+1} w/ r2: {train_r2:.2f}")
        ax[dim + 1][0].legend()

    # Test data predictions vs actual values for each dimension
    for dim in range(num_features):  # Assuming 2-dimensional y
        ax[dim + 1][1].plot(y_pred_test[:500, dim], label=f"Predicted Test Dim {dim+1}", linewidth=2)
        ax[dim + 1][1].plot(Y_test_torch.cpu().numpy()[:500, dim], label=f"Actual Test Dim {dim+1}", linewidth=2)
        test_r2 = r2_score(Y_test_torch.cpu().numpy()[:, dim], y_pred_test[:, dim])
        ax[dim + 1][1].set_title(f"LSTM Model Test Set Dim {dim+1} w/ r2: {test_r2:.2f}")
        ax[dim + 1][1].legend()

    plt.tight_layout()
    plt.show()


def test_1d_with_batch_and_with_seq(X, Y):

    print("Going to try and see how fast I can make it with mini batch")
    device = check_if_cuda_device_is_available()

    # Define LSTM model ----------------------------------------------
    class LSTMModel(nn.Module):
        """_summary_

        Architecture:
        -- Linear layer: Squashes the output of the LSTM to a continuous value for regression

        Args:
            nn (_type_): _description_
        """

        def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x.size(0): This represents the batch size, i.e., the number of sequences in the batch
            # #that will be processed by the LSTM. It's derived from the first dimension of the input
            # tensor x, which is the batch size.
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # initial hidden state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # initial cell state
            out, (_, _) = self.lstm(x, (h0, c0))  # Shape (num_samples, 1, hidden_dimens)
            # Now Squash the output of (Samples, sequence_lenght, hidden_dimens) to (Samples, output_dim)
            out = self.linear(out[:, -1, :])  # Take the last sequence output
            out = torch.tanh(out) * 3.14  # scale the output to be between -pi and pi
            return out

    # Hyperparameters
    input_dim = X.shape[1]
    hidden_dim = 100  # Number of LSTM cells
    num_layers = 1
    output_dim = 1
    num_epochs = 500
    sequence_length = 7
    epoch_saves = 50
    batch_size = int(len(X) / 2)  # The entire dataset is used for each batch

    # Preprocess the data -------------------------------------------
    data = LSTMDataset2(X, Y, seq_len=sequence_length)
    total_samples = len(data)
    train_size = int(total_samples * 0.95)  # 95% of the dataset is used for training
    train_dataset = Subset(data, range(0, train_size))
    test_dataset = Subset(data, range(train_size, total_samples))

    # Dataloader ingest a pytorch dataset class and returns an iterator for batching
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create the LSTM model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Init --------------------------------------------------------
    train_losses = []
    test_losses = []

    # Training the model ----------------------------------------------
    logger.info("Training the model")
    for epoch in range(num_epochs):
        model.train()
        for batch, (X_batch_train, Y_batch_train) in enumerate(training_loader):
            optimizer.zero_grad()
            y_pred_for_loss = model(X_batch_train)
            y_pred_for_loss = torch.squeeze(y_pred_for_loss)
            assert y_pred_for_loss.shape == Y_batch_train.shape, "The shapes of the model output must match the target for the loss function to work"
            train_loss = loss_fn(y_pred_for_loss, Y_batch_train)
            train_loss.backward()
            optimizer.step()
            
            loss, current = train_loss.item(), (batch + 1) * len(X_batch_train)
            logger.info(f"Epoch: {epoch + 1} loss: {loss:>7f}  [{current:>5d}/{len(training_loader.dataset)}]")
            
            if epoch % epoch_saves == 0:
                train_losses.append(train_loss.item())
                # ---- Validation on test set -----------------------------------------
                # Compute the test loss
                # model.eval()
                # with torch.no_grad():
                #     y_pred_test = model(X_test_torch)
                #     y_pred_test = torch.squeeze(y_pred_test)
                #     test_loss = loss_fn(y_pred_test, Y_test_torch)
                #     test_losses.append(test_loss.item())
                #     print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")

    # # Predicting Y with the trained model -------------------------------------
    # with torch.no_grad():
    #     y_pred_test = model(X_test_torch).cpu().numpy()
    #     y_pred_train = model(X_train_torch).cpu().numpy()

    # fig, ax = plt.subplots(figsize=(15, 6), nrows=2, ncols=2, gridspec_kw={"height_ratios": [1, 1], "width_ratios": [1, 1]})

    # # Remove the second subplot on the first row to make space for the learning curve
    # fig.delaxes(ax[0][1])

    # # Learning Curve (now spans two columns)
    # ax[0][0].plot(np.arange(0, num_epochs, epoch_saves), train_losses, label="Training Loss", color="blue")
    # ax[0][0].plot(np.arange(0, num_epochs, epoch_saves), test_losses, label="Test Loss", color="orange")
    # ax[0][0].set_title(f"Learning Curves - Min train loss: {min(train_losses):.2f}, Min test loss: {min(test_losses):.2f}")
    # ax[0][0].set_xlabel("Epoch")
    # ax[0][0].set_ylabel("Loss")
    # ax[0][0].legend(loc="upper right")

    # # Training data predictions vs actual values
    # ax[1][0].plot(y_pred_train[:500], label="Predicted Train", linewidth=2)
    # ax[1][0].plot(Y_train_torch.cpu().numpy()[:500], label="Actual Train", linewidth=2)
    # train_r2 = r2_score(Y_train_torch.cpu().numpy(), y_pred_train)
    # ax[1][0].set_title(f"LSTM Model Training Set w r2: {train_r2}")
    # ax[1][0].legend()

    # # Test fit
    # ax[1][1].plot(y_pred_test[:500], label="Predicted Test", linewidth=2)
    # ax[1][1].plot(Y_test_torch.cpu().numpy()[:500], label="Actual labels", linewidth=2)
    # test_r2 = r2_score(Y_test_torch.cpu().numpy(), y_pred_test)
    # ax[1][1].set_title(f"LSTM Model Test Set w r2: {test_r2}")
    # ax[1][1].legend()

    # plt.tight_layout()
    # plt.show()
