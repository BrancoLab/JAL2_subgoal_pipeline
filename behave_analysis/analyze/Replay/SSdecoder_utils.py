"""A script for functions used in notebooks using the SS decoder"""

import numpy as np

def predicted_position_from_posterior(posterior, method = 'weighted_average'):
    """A function for estimating the predicted position from the posterior distribution over bins.
    INPUTS:
        posterior: a 2D array of shape (n_time, n_bins) representing the posterior distribution over bins at each time point
        method: a string specifying the method to use for estimating the predicted position. Options are 'weighted_average' (default) and 'argmax'. 
            'weighted_average' computes the predicted position as the weighted average over bins, where the weights are given by the posterior probabilities. 
            'argmax' computes the predicted position as the center of the bin with the highest posterior probability."""
    if method == 'weighted_average':
        return np.sum(posterior.T * np.arange(posterior.shape[1])[:, None], axis=0) / np.sum(posterior.T, axis=0)
    elif method == 'argmax':
        return np.argmax(posterior, axis=1)
    
def predicted_position_2d(posterior, bin_centers, method='weighted_average'):
    """A function for estimating the predicted 2D position from the posterior distribution over position bins.
    INPUTS:
        posterior: a 3D array of shape (n_time, n_x_bins, n_y_bins) representing the posterior distribution over position bins at each time point
        bin_centers: a 2D array of shape (n_bins, 2) representing the x and y coordinates of the center of each position bin
        method: a string specifying the method to use for estimating the predicted position. Options are 'weighted_average' (default) and 'argmax'. 
            'weighted_average' computes the predicted position as the weighted average of the bin centers, where the weights are given by the posterior probabilities. 
            'argmax' computes the predicted position as the center of the bin with the highest posterior probability."""
    n_time = posterior.shape[0]
    posterior_flat = posterior.reshape(n_time, -1)
    predicted_position = np.full((n_time, bin_centers.shape[1]), np.nan)
    
    # Check for valid rows (rows where at least some bins have values)
    row_sums = np.nansum(posterior_flat, axis=1)  # Sum ignoring NaN
    valid_mask = (row_sums > 0) & np.isfinite(row_sums)
    
    if not valid_mask.any():
        return predicted_position
    
    if method == 'weighted_average':
        # Replace NaN with 0 (NaN bins have 0 probability)
        posterior_valid = np.nan_to_num(posterior_flat[valid_mask], nan=0.0)
        
        # Normalize (now only over valid bins)
        posterior_normalized = posterior_valid / posterior_valid.sum(axis=1, keepdims=True)
        
        # Weighted average
        predicted_position[valid_mask] = posterior_normalized @ bin_centers
    
    elif method == 'argmax':
        # Find the non-NaN bin with maximum probability
        posterior_valid = posterior_flat[valid_mask].copy()
        posterior_valid = np.where(np.isnan(posterior_valid), -np.inf, posterior_valid)
        map_bin_indices = np.argmax(posterior_valid, axis=1)
        predicted_position[valid_mask] = bin_centers[map_bin_indices]
    
    return predicted_position

def compute_rmse(predicted, actual):
    """Compute the root mean squared error between predicted and actual positions.
    INPUTS:
        predicted: a 2D array of shape (n_time, 2) representing the predicted x and y coordinates at each time point
        actual: a 2D array of shape (n_time, 2) representing the actual x and y coordinates at each time point"""
    if predicted.ndim == 1:
        predicted = predicted[:, np.newaxis]
    if actual.ndim == 1:
        actual = actual[:, np.newaxis]
    return np.sqrt(np.mean(np.sum((predicted - actual) ** 2, axis=1)))

def gaussian_smooth_rates(spikes, sigma_ms=20, bin_ms=1.0, edge_mode="reflect", trim_sigma=3.0):
    """
    Smooth binned spikes (time x neurons) with a Gaussian kernel.
    
    Parameters
    ----------
    spikes : array (T x N)
        Binary or count matrix (1 ms bins by default).
    sigma_ms : float
        Gaussian std dev in ms.
    bin_ms : float
        Bin size in ms.
    edge_mode : str
        "reflect" or "trim".
    trim_sigma : float
        Number of sigmas to trim at each edge if edge_mode="trim". trim = trim_sigma * sigma
    
    Returns
    -------
    rates : array
        Smoothed firing rates in Hz.
    """
    spikes = np.asarray(spikes)
    sigma_bins = sigma_ms / bin_ms
    
    # Build Gaussian kernel (±3σ by default)
    half_width = int(np.ceil(3 * sigma_bins))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()

    # Convolve along time for each neuron
    if edge_mode == "reflect":
        pad = half_width
        padded = np.pad(spikes, ((pad, pad), (0, 0)), mode="reflect")
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)
    elif edge_mode == "trim":
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, spikes)
        trim = int(np.ceil(trim_sigma * sigma_bins))
        if trim > 0:
            smoothed = smoothed[trim:-trim, :]
    else:
        raise ValueError("edge_mode must be 'reflect' or 'trim'")

    # Convert to Hz
    rates = smoothed * (1000.0 / bin_ms)
    return rates