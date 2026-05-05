import numpy as np
from scipy.stats import mstats
from numba import njit

@njit
def firing_by_bin_median_numba(var, neural_activity, nbins, remove_empty=False):
    """Compute the median firing rate per bin using Numba."""
    var = np.asarray(var)
    neural_activity = np.asarray(neural_activity)

    # Step 1: Pre-allocate a large 2D array to store values
    max_entries = len(var)  # Worst case: all data points in one bin
    bin_storage = np.full((nbins, max_entries), np.nan)  # Fill with NaNs
    bin_counts = np.zeros(nbins, dtype=np.int32)  # Track how many entries in each bin

    # Step 2: Assign values to bins
    for i in range(len(var)):
        bin_idx = var[i]
        if 0 <= bin_idx < nbins:
            count = bin_counts[bin_idx]  # Current count in bin
            bin_storage[bin_idx, count] = neural_activity[i]  # Store value
            bin_counts[bin_idx] += 1  # Increment bin count

    # Step 3: Compute medians
    angles_firing = np.full(nbins, np.nan)
    for i in range(nbins):
        if bin_counts[i] > 0:
            angles_firing[i] = np.median(bin_storage[i, : bin_counts[i]])  # Take median of non-NaN values

    # Step 4: Handle empty bins
    if remove_empty:
        return angles_firing[~np.isnan(angles_firing)]  # Remove NaNs
    else:
        return angles_firing

def trial_median_firing(mat, avg):
    """This function computes the firing rate across trials
    INPUTS:
        mat: matrix of trials x bins of firing rates
        avg: is a string that tells us the method to be used for averaging across trials.
            'median' takes the median,
            'winsorized' takes the winsorized mean, ignoring the 90th and 10th perc of the data"""
    all_nan_cols = np.all(np.isnan(mat), axis=0)
    smoothed_firing_rates = np.full(mat.shape[1], np.nan)
    if np.any(~all_nan_cols):  # Ensure at least one valid column exists
        if avg == "median":
            smoothed_firing_rates[~all_nan_cols] = np.nanmedian(mat[:, ~all_nan_cols], axis=0)
        elif avg == "winsorized":
            smoothed_firing_rates[~all_nan_cols] = nan_winsorized_mean_2d(mat[:, ~all_nan_cols], limits=(0.15, 0.15), axis=0)
    return smoothed_firing_rates


def nan_winsorized_mean_2d(data, limits=(0.1, 0.1), axis=0):
    """
    Compute the winsorized mean along a given axis while ignoring NaNs.

    Parameters:
        data (ndarray): 2D input array, may contain NaNs.
        limits (tuple): Fraction to trim from (low, high) ends.
        axis (int): Axis along which to compute the winsorized mean.

    Returns:
        ndarray: Winsorized mean along the specified axis.
    """
    data = np.asarray(data)

    # Apply Winsorization along the specified axis while ignoring NaNs
    def winsorize_nan_safe(arr):
        non_nan_arr = arr[~np.isnan(arr)]  # Remove NaNs
        if len(non_nan_arr) == 0:
            return np.nan  # Return NaN if all values were NaN
        return np.mean(mstats.winsorize(non_nan_arr, limits=limits))  # Winsorized mean

    return np.apply_along_axis(winsorize_nan_safe, axis, data)

def firing_by_bin_winz_mean(var, neural_activity, nbins, remove_empty=False):
    """For each bin of a variable, calculate the median neural activity.
    remove_empty: if True remove bins with no behavioral data.
    """
    if var.ndim > 1:
        if var.shape[1] != 1:
            raise ValueError("Input variable has more than one column. Please provide a 1D array or a 2D array with a single column.")
    angles_firing = np.full(nbins, np.nan)  # Start with NaN to handle empty bins
    for i in range(nbins):
        mask = (var == i).ravel()  # Find data points in the current bin
        if np.any(mask):  # Check if the bin has any data
            arr = neural_activity[mask]
            non_nan_arr = arr[~np.isnan(arr)]  # Remove NaNs
            if len(non_nan_arr) > 0:
                angles_firing[i] = np.mean(mstats.winsorize(non_nan_arr, limits=(.15, .15)))
    if remove_empty:
        angles_firing = angles_firing[~np.isnan(angles_firing)]  # Remove empty bins

    return angles_firing

def nan_valid_check(var, neural_activity):
    """Check for NaN values in the input variable and neural activity. 
    And remove those timepoints from the binned firing.
    var: vector of length time of binned <var> (e.g. speed) np.unique(<var>) = np.shape(fr_var_t2)[1]
    neural_activity: vector of length time of firing rates of a single neuron to be binned by var"""    
    valid_mask = ~np.isnan(var).ravel()
    if not np.any(valid_mask):
        raise ValueError("Input variable contains only NaN values.")
    # if np.any(~valid_mask):
    #     print(f"Warning: {np.sum(~valid_mask)} NaN values (likely out of bounds of bin edges) found in behavioral variable. These will be ignored.")
    var = var[valid_mask].astype(int)
    if neural_activity.ndim > 1: # if neural activity is 2D (e.g. n_neur x time) we need to apply the valid mask to the second dimension
        neural_activity = neural_activity[:, valid_mask]
    else: # if neural activity is 1D (e.g. activity of 1 neuron overtime) we apply the valid mask directly
        neural_activity = neural_activity[valid_mask]
    return var, neural_activity