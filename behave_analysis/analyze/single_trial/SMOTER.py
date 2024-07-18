import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Functions from the previous response
def get_neighbors_of_Xs(X, k):
    """Returns the indices of the k nearest neighbors for each sample in X"""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def smoter_interpolate(X, y, k, size) -> tuple:
    """Generate new samples using SMOTER interpolation.

    This function generates new samples by interpolating between a sample and one of its k nearest neighbors. It first
    does this for X. And then uses the euclidean distance between the new sample and the og, and the new sample and the neighbor
    to project the y value to a new sample. This is done for size number of samples. Random noise is added to the new samples.

    Args:
        X (np.array): (Nsamples, features) - Input data
        y (np.array): (Nsamples, ) - Target data
        k (int): Number of neighbors to consider
        size (int): Number of samples to generate

    Returns:
        _type_: _description_
    """
    X = np.asarray(X)  # (Nsamples, features) - Convert to numpy array
    y = np.squeeze(np.asarray(y))  # (Nsamples, ) - Convert to numpy array and remove extra dimensions
    assert len(X) == len(y), "X and y must be of the same length."
    neighbor_indices = get_neighbors_of_Xs(X, k)  # (1000, k) Get k nearest neighbors for each sample and return those indices
    num_samples = len(y)
    sample_indices = np.random.choice(range(num_samples), size, replace=True)  # randomly select interger from 0 to len(y) with size
    X_new, y_new = [], []  # Create empty list to store new X and y values

    # For each new sample to be generated
    for i in sample_indices:
        X_case, y_case = X[i, :], y[i]  # Randomly select a sample from X and y
        neighbor = np.random.choice(neighbor_indices[i, :])  # (1, ) Randomly select a neighbor from the k nearest neighbors
        X_neighbor, y_neighbor = X[neighbor, :], y[neighbor]  # Get the X and y values of that neighbor
        rand = np.random.rand() * np.ones_like(X_case)  # (1, ) Create some random noise
        diff = (X_case - X_neighbor) * rand  # Find the diff between random neighbor and the original sample
        X_new_case = X_neighbor + diff  # Add the diff to the neighbor to get the new sample
        d1 = np.linalg.norm(X_new_case - X_case, ord=2)  # calculate euclidean distance between new sample and original sample
        d2 = np.linalg.norm(X_new_case - X_neighbor, ord=2)  # calculate euclidean distance between new sample and neighbor
        # Using the euclidean distances from the X examples, interpolate the y value
        y_new_case = (d2 * y_case + d1 * y_neighbor) / (d2 + d1 + 1e-10)  # calculate the new y value, 1e-10 is added to avoid division by zero
        X_new.append(X_new_case)
        y_new.append(y_new_case)

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    return X_new, y_new


def uniform_test_split(X, y, bins, samples_per_bin=100, random_state=None):
    """
    Bins: array of bin edges
    y: array of target values
    samples_per_bin: target number of samples per bin in the test set
    test_ratio: approximate ratio of test set size to total dataset size

    TODO: Currently makes test size dependent on the number of samples in each bin. This could be improved. So you
    could set a fixed test size percentage and interpolate further depending on the number of samples in each bin.
    """
    np.random.seed(random_state)
    y = np.squeeze(np.asarray(y))
    X = np.asarray(X)
    assert len(X) == len(y), "X and y must be of the same length."

    bin_indices = np.digitize(y, bins)
    unique_bins = np.unique(bin_indices)

    train_indices, test_indices = [], []
    train_freqs, test_freqs = [], []

    for bin_index in unique_bins:
        bin_mask = bin_indices == bin_index
        bin_samples = np.where(bin_mask)[0]

        if len(bin_samples) > samples_per_bin:
            # Undersample
            test_index = np.random.choice(bin_samples, samples_per_bin, replace=False)
            train_index = np.setdiff1d(bin_samples, test_index)
        else:
            # # Use all samples for test and interpolate
            test_index = np.random.choice(bin_samples, len(bin_samples), replace=True)
            train_index = np.setdiff1d(bin_samples, test_index)

        test_indices.extend(test_index)
        train_indices.extend(train_index)

        test_freqs.append(len(test_index))
        train_freqs.append(len(train_index))

    return train_indices, test_indices


def uniform_sample(X, y, bins, samples_per_bin=100):
    """
    Bins: array of bin edges
    y: array of target values
    samples_per_bin: target number of samples per bin in the test set
    test_ratio: approximate ratio of test set size to total dataset size

    TODO: Currently makes test size dependent on the number of samples in each bin. This could be improved. So you
    could set a fixed test size percentage and interpolate further depending on the number of samples in each bin.
    """
    y = np.squeeze(np.asarray(y))
    X = np.asarray(X)
    assert len(X) == len(y), "X and y must be of the same length."

    bin_indices = np.digitize(y, bins)
    unique_bins = np.unique(bin_indices)

    ind = []

    # for each bin, get the samples and randomly select samples_per_bin number of samples
    for bin_index in unique_bins:
        bin_mask = bin_indices == bin_index
        bin_samples = np.where(bin_mask)[0]

        # If we have enough samples, choose [samples_per_bin] samples from the binned Y values
        if len(bin_samples) > samples_per_bin:
            # Undersample
            indicies = np.random.choice(bin_samples, samples_per_bin, replace=False)
        else:
            # Use all samples for and replace them
            indicies = np.random.choice(bin_samples, samples_per_bin, replace=True)

        ind.extend(indicies)

    return ind


def smotre_and_resample(X, y, bins, noise_std=0.1, oversampling_factor=10):
    """Perform OLS regression with SMOTER interpolation and uniform sampling

    Args:
    -- X: np.array of shape (Nsamples, Nfeatures)
    -- y: np.array of shape (Nsamples,)
    -- bins: np.array of bin edges for uniform sampling
    -- noise_std: standard deviation of noise to add to interpolated data
    -- oversampling_factor: factor by which to oversample the data"""

    # Step 1: Perform SMOTER interpolation
    X_interpolated, y_interpolated = smoter_interpolate(X, y, k=5, size=len(X) * oversampling_factor)
    noise = np.random.normal(0, noise_std, X_interpolated.shape)
    X_interpolated_noisy = X_interpolated + noise  # Add noise to Xs to make it more realistic
    X = X_interpolated_noisy
    y = y_interpolated

    # Step 2: Perform uniform sampling
    train_indices, test_indices = uniform_test_split(X, y, bins)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test, X_interpolated, y_interpolated


if __name__ == "__main__":

    # Generate non-uniform test data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.rand(n_samples, 1) * 10  # (Nsamples, Nfeatures)
    y = 2 * X.squeeze() + np.random.exponential(scale=2, size=n_samples)  # (Nsamples,)
    bins = np.histogram(y, bins=50)[1]  # Define bins for uniform sampling

    # Perform regression with sampling
    X_train, y_train, X_test, y_test, X_interp, Y_interp = smotre_and_resample(X, y, bins, noise_std=0.1)

    # Plot the results of the resampling technique for comparison
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))

    # What does the original data look like?
    ax[0][0].hist(y, bins=50, color="red", alpha=0.5, label="Original Y")
    ax[0][0].set_title("Original Y data before any modification")
    ax[0][0].set_ylabel("Count")
    ax[0][0].legend()

    # What happens after SMOTER interpolation?
    ax[1][0].hist(Y_interp, bins=50, color="green", alpha=0.5, label="Smoter interpolated Y")
    ax[1][0].set_title("Output of SMOTER interpolation")
    ax[1][0].legend()
    ax[1][0].set_ylabel("Count")

    # What happens after uniform sampling?
    ax[2][0].hist(y_test, bins=50, color="blue", alpha=0.5, label="Smoter + Uniform sampled test Y")
    ax[2][0].set_xlabel("Arbitrary value")
    ax[2][0].set_title("Test Y data after sampling")
    ax[2][0].legend()
    ax[2][0].set_ylabel("Count")

    # If I do cross val on the original data what does the test r2 look like?
    X_train_og, X_test_og, y_train_og, y_test_og = train_test_split(X, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train_og, y_train_og)
    y_pred = lr.predict(X_test_og)
    r2 = r2_score(y_test_og, y_pred)
    ax[0][1].scatter(X_test_og, y_test_og, color="red", alpha=0.5, label="Original data")
    ax[0][1].plot(X_test_og, y_pred, color="black", label=f"Test R2: {r2:.2f}")
    ax[0][1].set_title("Original data with OLS fit")
    ax[0][1].legend()

    # If I use the interpolated data what does the test r2 look like?
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    ax[1][1].scatter(X_test, y_test, color="green", alpha=0.5, label="Interpolated + uniform sampling data")
    ax[1][1].plot(X_test, y_pred, color="black", label=f"Test R2: {r2:.2f}")
    ax[1][1].set_title("Interpolated data with OLS fit")
    ax[1][1].legend()

    # Clear last axs
    ax[2][1].axis("off")

    plt.show()
