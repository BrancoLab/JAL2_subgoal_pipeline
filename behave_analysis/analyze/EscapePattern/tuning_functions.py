import numpy as np
from scipy.ndimage import gaussian_filter1d
from multiprocessing import shared_memory

from behave_analysis.utils.PersistentPool import PersistentPool
from behave_analysis.analyze.EscapePattern.gaussian_fitting import gaussian_fitting
from behave_analysis.analyze.EscapePattern.median_functions import firing_by_bin_median_numba, trial_median_firing, firing_by_bin_winz_mean
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import get_homings_onsets_in_filtered_time

# ------------------------------------Tuning for homing and escape periods (requires trials)------------------------------------

def compute_tuning_curves(var, escape_matrix, cond, bins, filtering_vector, n_cond, n_neur, n_trials, avg="winsorized", fitting=True, loo=False):
    """Filter (gauss or savgol) the full trace of neural activity -> take median per trial -> take median across trials -> fit gaussian
    INPUTS:
        avg: is a string that tells us the method to be used for averaging across trials.
            'median' takes the median,
            'winsorized' takes the winsorized mean, ignoring the 90th and 10th perc of the data
        loo: boolean, if True, leave one out reliability will also be computed
    RETURNS:
        y_fitted_full: matrix of conditions x neurons x n_bins of the gaussian fits of the average data across trials
        R_full: matrix of neurons x conditions of the R^2 of the gaussian fits
        fr_full: matrix of conditions x neurons x n_bins of the average data across trials
        params_full: matrix of neurons x conditions x 6 of the parameters of the gaussian fit.
            If single gaussian is best = [Amplitude, mu, sigma, nan, nan, nan];
            if double gaussian fit is best = [Amplitude1, mu1, sigma1, Aplitude2, mu2, sigma2]
        mat_num_cond: matrix of conditions x neurons x trials x n_bins of the average data per trial
        leave_one_out_reliability: matrix of conditions x neurons of the leave one out reliability score
    """
    # step 1: filter the full trace
    # Assuming the filtered trace is what is passed as an arg to extract_homing_and_escape_periods

    # step 2: extract homie periods
    # This is done outside as well

    # find the start of each homing period
    h_start = get_homings_onsets_in_filtered_time(filtering_vector)

    # initialize variables for output
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan)  # conditions x neurons x n_bins
    R_full = np.full((n_neur, n_cond), np.nan)  # neurons x conditions
    fr_full = np.full((n_cond, n_neur, bins), np.nan)  # conditions x neurons x n_bins
    params_full = np.full((n_neur, n_cond, 6), np.nan)
    mat_num_cond = np.full((n_cond, n_neur, n_trials, bins), np.nan)  # conditions x neurons x trials x bins

    reliability = np.full((n_cond, n_neur), np.nan)  # conditions x neurons

    # step 3: compute firing per bin per trial
    # iterate through conditions
    for i in range(n_cond):
        i = int(i)
        # start by condition
        cond_start = [x for x in h_start if cond[x] == i]
        cond_start = np.concatenate((cond_start,[np.sum(cond < i+1)])) # this adds the end of the last trial

        # iterate through neurons
        for j, n in enumerate(escape_matrix):
            # iterate through trials, pull out firing by bin
            for tr, _ in enumerate(cond_start[:-1]):
                neur = n[cond_start[tr] : cond_start[tr + 1]]
                v = var[cond_start[tr] : cond_start[tr + 1]]
                mat_num_cond[i, j, tr, :] = firing_by_bin_median_numba(v.astype(int), neur, bins, remove_empty=False)

            # step 4: take median across trials
            mat = mat_num_cond[i, j, :, :]
            smoothed_firing_rates = trial_median_firing(mat, avg)

            # Step 5: Gaussian fit
            distances = np.arange(len(smoothed_firing_rates))
            valid_idx = ~np.isnan(smoothed_firing_rates)
            v = np.where(valid_idx)[0]
            y_fitted = np.full(bins, np.nan)
            R, params = 0, np.full(6, np.nan)
            if np.any(valid_idx):
                if fitting:
                    R, y, params, _ = gaussian_fitting(smoothed_firing_rates[valid_idx], distances[valid_idx], verbose=False)
                    y_fitted[valid_idx] = y
                else:
                    y = gaussian_filter1d(smoothed_firing_rates[v[1:-1]], 2)
                    # WARNING: I'm not allowing the max to be at the edges
                    params[0] = np.nanmax(y)  # amplitude of the peak response
                    params[1] = v[np.argmax(y) + 1]  # location of the peak response
                    y_fitted[valid_idx] = gaussian_filter1d(smoothed_firing_rates[valid_idx], 2)

            # Step 6: Leave one out reliability
            if loo:
                reliability[i, j] = compute_leave_one_out_reliability(mat, smoothed_firing_rates, avg)

            # dump together for output
            fr_full[i, j, :] = smoothed_firing_rates
            R_full[j, i] = R
            params_full[j, i, : len(params)] = params
            y_fitted_full[i, j, :] = y_fitted

    return y_fitted_full, R_full, fr_full, params_full, mat_num_cond, reliability

def compute_tuning_curves_no_trials(var, escape_matrix, cond, bins, n_cond, n_neur, avg = 'winsorized', fitting = True):
    """Filter (gauss or savgol) the full trace -> take median across all time
    INPUTS:
        avg: is a string that tells us the method to be used for averaging across trials. 
            'median' takes the median, 
            'winsorized' takes the winsorized mean, ignoring the 90th and 10th perc of the data"""
    # step 1: filter the full trace
    # Assuming the filtered trace is what is passed as an arg to extract_homing_and_escape_periods

    # step 2: extract relevant neuron x time matrix
    # assumed to be one of the inputs

    # initialize variables for output
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan) # conditions x neurons x n_bins
    R_full = np.full((n_neur, n_cond), np.nan) # neurons x conditions
    fr_full = np.full((n_cond, n_neur, bins), np.nan) # conditions x neurons x n_bins
    params_full = np.full((n_neur, n_cond, 6), np.nan)

    # step 3: compute firing by bin across all time
    for c in range(n_cond):
        neur = escape_matrix[:,cond == c]
        var_cond = var[cond == c]
        for i, n in enumerate(neur):
            if avg == 'median':
                smoothed_firing_rates = firing_by_bin_median_numba(var_cond.astype(int), n, bins, remove_empty = False)
            elif avg == 'winsorized':
                smoothed_firing_rates = firing_by_bin_winz_mean(var_cond.astype(int), n, bins, remove_empty = False)
            
            # Gaussian fitting
            valid_idx = ~np.isnan(smoothed_firing_rates)
            R, y_fitted, params = 0, np.full_like(smoothed_firing_rates, np.nan), np.full(6, np.nan)
            valid = np.where(valid_idx)[0]
            if np.any(valid_idx):
                if fitting:
                    R, y, params, _ = gaussian_fitting(smoothed_firing_rates[valid_idx], np.arange(np.sum(valid_idx)), verbose=False)
                    y_fitted[valid_idx] = y
                else:
                    y = gaussian_filter1d(smoothed_firing_rates[valid_idx], 2)
                    params[0] = np.nanmax(y[1:-1])
                    y_fitted[valid_idx] = y
                    params[1] = valid[np.argmax(y[1:-1])+1] # WARNING: I'm not allowing the max to be at the edges

            # dump together for output
            fr_full[c, i, :] = smoothed_firing_rates
            R_full[i, c] = R
            params_full[i, c,:len(params)] = params
            y_fitted_full[c, i, :len(y_fitted)] = y_fitted

    return y_fitted_full, R_full, fr_full, params_full

# ------------------------------------Leave one out reliability computation------------------------------------

def compute_leave_one_out_reliability(mat, smoothed_firing_rates, avg):
    """This function computes the leave one out reliability score for a given neuron in each condition
    computes for each trial the correlation coefficient between that trial and the median of all other trials,
    then averages those correlation coefficients across trials, weighted by the rms of each trial
    RETURN:
        reliability:"""

    tr_corr_coeff = np.full(mat.shape[0], np.nan)  # trials
    tr_rms = np.full(mat.shape[0], np.nan)  # trials
    reliability = np.nan

    if np.sum(smoothed_firing_rates) > 0:
        for tr in range(mat.shape[0]):  # loop over trials and leave one out of the median
            loo_mat = np.delete(mat, tr, axis=0)  # matrix of trials x bins with one trial left out
            if loo_mat.shape[0] == 0:  # Prevent empty matrix issues
                continue
            loo = trial_median_firing(loo_mat, avg)  # leave one out median firing rates

            # corr coeff for each trial
            id_nans = np.logical_or(np.isnan(loo), np.isnan(mat[tr, :]))
            valid_corr_values = np.sum(~id_nans)  # Count non-NaN values
            if valid_corr_values > 1 and np.std(loo[~id_nans]) > 0 and np.std(mat[tr, ~id_nans]) > 0:
                tr_corr_coeff[tr] = np.corrcoef(loo[~id_nans], mat[tr, ~id_nans])[0, 1]
            tr_rms[tr] = np.sqrt(np.mean(mat[tr, :] ** 2))

        # average across trial
        id_nans = np.logical_or(np.isnan(tr_corr_coeff), np.isnan(tr_rms))
        if np.sum(id_nans) < len(id_nans):
            reliability = np.average(tr_corr_coeff[~id_nans], weights=tr_rms[~id_nans])

    return reliability

# ------------------------------------Tuning with shared memory and parallelization for exploration periods------------------------------------

def tuning_method_no_trials_with_pool(var, escape_matrix, cond, bins, n_cond, n_neur, fitting=True):
    """Optimized version using shared memory and parallelization for neurons.
    Filter (gauss or savgol) the full trace -> take median across all time"""

    # Create shared memory for escape_matrix
    escape_shm = shared_memory.SharedMemory(create=True, size=escape_matrix.nbytes)
    shared_escape_matrix = np.ndarray(escape_matrix.shape, dtype=escape_matrix.dtype, buffer=escape_shm.buf)
    np.copyto(shared_escape_matrix, escape_matrix)  # Copy data into shared memory

    # Initialize output arrays
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan)
    R_full = np.full((n_neur, n_cond), np.nan)
    fr_full = np.full((n_cond, n_neur, bins), np.nan)
    params_full = np.full((n_neur, n_cond, 6), np.nan)

    # Initialize persistent multiprocessing pool
    PPool = PersistentPool()

    # Step 3: Compute firing by bin across all time
    for c in range(n_cond):
        cond_indices = np.where(cond == c)[0]  # Precompute condition indices to avoid slicing overhead

        # Prepare arguments for multiprocessing
        args_list = [(escape_shm.name, escape_matrix.shape, escape_matrix.dtype, c, i, cond_indices, bins, var, fitting) for i in range(n_neur)]

        # Use multiprocessing pool to process neurons in parallel
        results = PPool.mp_pool.starmap(tuning_method_no_trials_parallel_function, args_list)

        # Store results
        for _, i, smoothed_firing_rates, R, params, y_fitted in results:
            fr_full[c, i, :] = smoothed_firing_rates
            R_full[i, c] = R
            params_full[i, c, : len(params)] = params
            y_fitted_full[c, i, : len(y_fitted)] = y_fitted

    # Cleanup shared memory
    escape_shm.close()
    escape_shm.unlink()
    PPool.close()

    return y_fitted_full, R_full, fr_full, params_full

def tuning_method_no_trials_parallel_function(shared_name, shape, dtype, i, neuron_index, cond_indices, bins, var, fitting, avg = 'winsorized'):
    """Worker function to process a single neuron using shared memory.
    Process a single neuron for a given condition."""
    
    # Reconnect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shared_name)
    escape_matrix = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Extract relevant data for this neuron
    n = escape_matrix[neuron_index, cond_indices]  # Avoid passing large slices through multiprocessing
    v = var[cond_indices]

    # Compute firing rates
    if avg == 'median':
        smoothed_firing_rates = firing_by_bin_median_numba(v.astype(int), n, bins, remove_empty = False)
    elif avg == 'winsorized':
        smoothed_firing_rates = firing_by_bin_winz_mean(v.astype(int), n, bins, remove_empty = False)

    # Gaussian fitting
    R, y_fitted, params = 0, np.full_like(smoothed_firing_rates, np.nan), np.full(6, np.nan)
    valid_idx = ~np.isnan(smoothed_firing_rates)
    v = np.where(valid_idx)[0]
    if np.any(valid_idx):
        if fitting:
            R, y, params, _ = gaussian_fitting(smoothed_firing_rates[valid_idx], np.arange(np.sum(valid_idx)), verbose=False)
            y_fitted[valid_idx] = y
        else:
            y = gaussian_filter1d(smoothed_firing_rates[v[1:-1]], 2)
            params[0] = np.nanmax(y)
            params[1] = v[np.argmax(y)+1] # WARNING: I'm not allowing the max to be at the edges
            y_fitted[valid_idx] = gaussian_filter1d(smoothed_firing_rates[valid_idx], 2)

    existing_shm.close()  # Close shared memory connection in worker

    return i, neuron_index, smoothed_firing_rates, R, params, y_fitted
