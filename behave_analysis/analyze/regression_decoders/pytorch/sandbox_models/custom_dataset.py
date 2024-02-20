"""Custom Dataset class for use with RNN based networks in PyTorch

Dataset stores the samples and their corresponding labels. It is designed to be used with 
a DataLoader in PyTorch, which will yield batches of data in the 
format `(batch_size, seq_len, n_features)`

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class LSTMDataset(Dataset):
    """
    A custom Dataset class for use with LSTM networks in PyTorch. This class
    extends the PyTorch Dataset and is tailored to provide input data for
    RNN-based architectures, particularly LSTMs. It transforms the input data X
    into sequence data using a rolling window approach.

    This dataset is designed to be used with a DataLoader in PyTorch, which
    will yield batches of data in the format `(batch_size, seq_len, n_features)`,
    where:
    - `batch_size` is the number of sequences fed into the model at once,
    - `seq_len` is the number of time steps in each sequence,
    - `n_features` is the number of features in each sequence.

    Parameters:
    - X (array-like): The input matrix of samples vs features, typically a 2D array [samples, features]
        where features are the number of neurons for e.g and samples are spikes per time point
    - Y (array-like): The target values associated with X
    - seq_len (int, optional): The length of the sequence window

    Attributes:
    - X (Tensor): The input features converted to a PyTorch tensor of shape
      `[batch_size, seq_len, features]`.
    - Y (Tensor): The target values converted to a PyTorch tensor. The shape can vary based on the specific task.
    - seq_len (int): The length of the sequence window.

    Methods:
    - __len__(): Returns the total number of possible sequences that can be generated from X given the specified `seq_len`.
    - __getitem__(index): Retrieves a single item (input-target pair) from the dataset at the specified index."""

    # NOTE - Sequnece lenght is currently 1

    def __init__(self, X, Y, seq_len: int):
        """Initialize the dataset."""
        assert len(X) == len(Y), "X and Y must have the same length."
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        """Return length of the dataset.

        This method alculates the number of complete, non-overlapping sequences of length self.seq_len
        that can be formed from a dataset X of length len(self.X). This not just len(X) because we
        are using sequences instead of individual samples like in an feed-forward network."""
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Parameters:
        - idx: The index of the sample to retrieve.

        Returns:
        - A tuple containing the sample and its corresponding label.
        """
        return (
            self.X[index : index + self.seq_len],
            self.Y[index + self.seq_len],  # For simplicity, associate the label with the last item in the sequence
        )

        # return (self.X[index], self.Y[index])

class LSTMDataset2(Dataset):
    def __init__(self, X, Y, seq_len: int, device='cuda'):
        """
        Initialize the dataset with data X and labels Y, transforming them
        into sequences for LSTM training.
        """
        assert len(X) == len(Y), "X and Y must have the same length."
        
        self.device = device
        self.seq_len = seq_len
        self.num_sequences = len(X) - seq_len + 1
        
        # Pre-compute the sequences to avoid doing it in __getitem__, for efficiency
        self.X_sequences = np.zeros((self.num_sequences, seq_len, X.shape[1]))
        self.Y_sequences = np.zeros((self.num_sequences,))
        
        for i in range(self.num_sequences):
            self.X_sequences[i] = X[i:i+seq_len]
            self.Y_sequences[i] = Y[i+seq_len-1]
        
        # Convert to tensors
        self.X_sequences = torch.tensor(self.X_sequences, dtype=torch.float32).to(device)
        self.Y_sequences = torch.tensor(self.Y_sequences, dtype=torch.float32).to(device)

    def __len__(self):
        """
        Returns the total number of sequences.
        """
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Retrieves the sequence and its corresponding label at index `idx`.
        """
        X_seq = self.X_sequences[idx]
        Y_seq = self.Y_sequences[idx]
        return X_seq, Y_seq