import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from multiprocessing import shared_memory

from JR_test_scripts.escape.escape_utils import firing_by_bin, firing_by_bin_median_numba
from behave_analysis.utils.PersistentPool import PersistentPool

def neuron_tuning_by_var(esc_var, escape_matrix, cond, h_start = [], epoch_method = 'trial', xval_method = 'cosinesim', n_epochs = 3, xval_thresh = .7, averaging = 'median'):
    """What is the tuning curve and peak firing bin of each neuron per condition?
    INPUTS:
        esc_var: is in time, the variable we want to compute the tuning for
        escape_matrix: is neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
    RETURNS:
        peak_firing_condition is a matrix of neurons x condition, where each entry is the bin with the peak firing for that nruon in that condition to the behavioral variable
        tuning_by_cond: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval"""
    peak_firing_condition = np.zeros((np.shape(escape_matrix)[0], len(np.unique(cond))))
    tuning_by_cond = []
    xval = np.zeros((np.shape(escape_matrix)[0], len(np.unique(cond))))
    cond_start = []
    for i in np.unique(cond):
        # start by condition
        if len(h_start) > 0:
            start = [x for x in h_start if cond[x] == i]
            cond_start = [x-start[0] for x in start]
        tuning_matrix, xval[:,int(i)] = create_xval_tuning_curve(esc_var[cond == i], 
                                                                 escape_matrix[:,cond == i], 
                                                                 start = cond_start, # resetting the timestamp of homing starts to align to the start of the condition
                                                                 bins = int(np.amax(esc_var)+1),
                                                                 epoch_method = epoch_method, 
                                                                 xval_method = xval_method,
                                                                 xval_thresh = xval_thresh, 
                                                                 n_epochs = n_epochs)
        # bins = esc_var[cond == i]
        peak_firing = np.argmax(tuning_matrix, axis = 1)
        peak_firing_condition[:,int(i)] = peak_firing # bins[peak_firing]
        tuning_by_cond.append(tuning_matrix)
    return peak_firing_condition, tuning_by_cond, xval.astype(int)

def create_xval_tuning_curve(esc_var, escape_matrix, bins, start = [], xval_thresh = .7, epoch_method = 'trial', xval_method = 'cosinesim', n_epochs = 3, normalize_tuning_curve = False, plot = False, averaging = 'median'):
    """Function that computes the tuning of each neuron for a given variable, cross vlaidates it and checks if it's a reliable cell
    several methods can be implemented for xval: separate by trial, even and odd time point, time
    NB: you can call this by condition or for all time!
    
    INPUTS:
        esc_var: is in time, the variable we want to compute the tuning for
        escape_matrix: is neurons x time
        start: if using 'trial' method for xval, this is the start time of the trials (if dividing by condition this will have been adjusted to the start of the condition)
        bins: number of bins for the tuning curve
        epoch_method: how to chunk the epochs, options are 'time', 'trial', 'alt_time'
        xval_method: how to compare the tuning curves, options are 'corr', 'mse', 'cosinesim', 'euclid'
        n_epochs: how many epochs we want to chunk into
        normalize_tuning_curve: if True normalize the tuning curve to 1, if False keep the raw values
    
    RETURNS:
        full_tuning: is neurons x bins, where each line is the tuning of that neuron for the variable
        xval_pass: is len(neurons) and is True if the neuron passed the xval test
    """
    # divide into epochs
    epochs = create_xval_epochs(len(esc_var), start, n_epochs, epoch_method)

    result = np.zeros((np.shape(escape_matrix)[0],len(np.unique(epochs))))
    bad_neurons = np.all(np.isnan(escape_matrix), axis=1)
    for i in np.unique(epochs):

        # compute tuning curve for each epoch
        test_var = esc_var[epochs == i]
        train_var = esc_var[epochs != i]
        test_mat = escape_matrix[:,epochs == i]
        train_mat = escape_matrix[:,epochs != i]
        # if bins = bins the max will be the max of all the bins, but if not all bins are visited in this condition, it's better to use for bins only up to the the max bin used
        test_tuning = creat_tuning_curve(test_var, test_mat, int(np.amax(esc_var)+1)) 
        train_tuning = creat_tuning_curve(train_var, train_mat, int(np.amax(esc_var)+1))

        # compare tuning curves, iterate over neurons and compute the similarity
        for it in np.arange(len(result)):
            if bad_neurons[it] == True: # this means it's an all nan neuron!
                continue
            curve1 = test_tuning[it,:]
            curve2 = train_tuning[it,:]
            if normalize_tuning_curve:
                curve1 = test_tuning[it,:]/np.amax(test_tuning[it,:])
                curve2 = train_tuning[it,:]/np.amax(train_tuning[it,:])
                curve1[np.logical_or(test_tuning[it,:] == 0, np.isinf(curve1))] = 0
                curve2[np.logical_or(train_tuning[it,:] == 0, np.isinf(curve2))] = 0
            if xval_method == 'corr':
                result[it,int(i)], _ = pearsonr(curve1, curve2)
            # if xval_method == 'mse': # don't like it, values not spread well
            #     result[it,int(i)] = np.mean((curve1 - curve2)** 2)
            if xval_method == 'cosinesim':
                result[it,int(i)] = cosine_similarity(curve1.reshape(1, -1),curve2.reshape(1, -1))[0,0]
            if xval_method == 'euclid':
                result[it,int(i)] = np.linalg.norm(curve1 - curve2) / np.linalg.norm(curve1 + curve2)

    # average results across epochs
    avg_result = np.mean(result, axis = 1)
    if np.logical_or(xval_method == 'corr', xval_method == 'cosinesim'):
        xval_pass = avg_result > xval_thresh
    if xval_method == 'euclid':
        xval_pass = avg_result < xval_thresh # recommend .45

    if plot:
        _, axs = plt.subplots(2,2,figsize = (3,6))
        for i in [0,1]:
            t = test_tuning[xval_pass == i,:]
            idx = np.argmax(t, axis = 1)
            isort = np.argsort(idx)
            axs[i,0].set_ylabel('neurons sorted by test tuning \nxval pass = ' + str(i), fontsize = 8)
            axs[i,0].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
            t2 = train_tuning[xval_pass == i,:]
            axs[i,1].imshow(t2[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
        axs[0,0].set_title('test tuning')
        axs[0,1].set_title('train tuning')
        plt.tight_layout()
        plt.show()

    # compute tuning curve for all time
    full_tuning = creat_tuning_curve(esc_var, escape_matrix, bins)

    return full_tuning, xval_pass

def creat_tuning_curve(esc_var, escape_matrix, nbins, averaging = 'median'):
    """This function creates a matrix of neurons x bins, where each line is the tuning of that neuron for the variable
    INPUT: 
        escape_matrix: is neurons x time
        esc_var: is in time
    
    RETURNS:
        tuning_matrix: is neurons x bins, where each line is the tuning of that neuron for the variable
    """
    tuning_matrix = np.full((np.shape(escape_matrix)[0],nbins), np.nan)
    for i, n in enumerate(escape_matrix):
        if averaging == 'median':
            tuning_matrix[i,:] = firing_by_bin_median(esc_var.astype(int), n, nbins, remove_empty = False)
        elif averaging == 'mean':
            tuning_matrix[i,:] = firing_by_bin(esc_var.astype(int), n, nbins, remove_empty = False)
    return tuning_matrix

def create_xval_epochs(time, start, n_epochs, epoch_method):
    """This function creates a vector of integers indicating the epochs for cross validation
    INPUTS:
        time: in frames usually (it's the length of time we want to chunk into epochs), gives us the length of the epochs
        n_epochs: how many epochs we want to chunk into
        epoch_method: how to chunk the epochs, options are 'time', 'trial', 'alt_time'
        start: if using 'trial' method, this is the start time of the trials
    
    RETURNS:
        epochs: a vector of integers indicating the epochs for cross validation, of length var
    """
    epochs = np.zeros(time)
    if epoch_method == 'time':
        transitions = [int(time/4),int(time/2),int((time/4)*3)]
        for i in transitions:
            epochs[i:] += 1
    if epoch_method == 'alt_time':
        values = np.arange(n_epochs)
        block_size = 40*3 # 40 frames per second, 3 seconds per epoch
        epochs = np.tile(np.repeat(values, block_size), time // (len(values) * block_size) + 1)
        epochs = epochs[:time]
    if epoch_method == 'trial':
        for st in start[1:-1]:
            epochs[st:] += 1
            epochs = np.mod(epochs, n_epochs) # these epochs are not equal in size, because the trials are all of different lengths!!
    return epochs

def single_trial_tuning(escape_matrix, var, cond, h_start, bins = None, averaging = 'median'):
    """Make a matrix for each neuron of activity per bin on each trial
    INPUT: 
        escape_matrix: is neurons x time (firing rates)
        esc_var: is in time, the behavioral variable of interest (discretized)
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        """
    mat_by_cond = []
    # iterate through conditions
    for i in np.unique(cond):
        # start by condition
        cond_start = [x for x in h_start if cond[x] == i]
        cond_start.append(np.sum(cond == i))
        # initialize variable, neurons x trials x bins
        if bins == None:
            bins = int(np.amax(var)+1)
        mat = np.full((np.shape(escape_matrix)[0],len(cond_start),bins), np.nan)
        # iterate through neurons
        for j, n in enumerate(escape_matrix):
            # iterate through trials, pull out firing by bin
            for tr, _ in enumerate(cond_start[:-1]):
                neur = n[cond_start[tr]:cond_start[tr+1]]
                v = var[cond_start[tr]:cond_start[tr+1]]
                if averaging == 'median':
                    mat[j, tr,:] = firing_by_bin_median(v.astype(int), neur, bins, remove_empty = False)
                elif averaging == 'mean':
                    mat[j, tr,:] = firing_by_bin(v.astype(int), neur, bins, remove_empty = False)
        mat_by_cond.append(mat)
    return mat_by_cond

def fit_gaussian(firing_rates, distances, initial_guess = [], constraints = [], verbose = False):
    """Fit a Gaussian to the data and return the fitted curve, R squared and parameters
    
    INPUTS:
        firing_rates: the firing rates of the neuron
        var: the bins at which the firing rate is calculated (0:1:nbins)
    
    RETURNS:
        y_fitted: the fitted Gaussian curve
        R: the R squared value of the fit
        params: the parameters of the Gaussian fit (A, mu, sigma)        
    """
    # how to pick initial params
    if len(initial_guess) == 0:
        initial_guess = [max(firing_rates), distances[np.argmax(firing_rates)], np.std(distances)]  # Initial guesses for A, mu, sigma
    # Fit Gaussian to the data
    params, _ = curve_fit(gaussian, distances, firing_rates, p0=initial_guess, bounds = constraints)

    # Extract parameters
    A, mu, sigma = params
    if verbose:
        print(f"Fitted parameters: A = {A:.2f}, mu = {mu:.2f}, sigma = {sigma:.2f}")

    # Generate points for the fitted Gaussian
    x_fitted = distances
    y_fitted = gaussian(x_fitted, A, mu, sigma)
    R = compute_r_squared(firing_rates, y_fitted)

    return y_fitted, R, params

def fit_double_gaussian(firing_rates, distances, initial_guess_double, constraints = [], verbose = False):
    """Fit a Gaussian to the data and return the fitted curve, R squared and parameters
    
    INPUTS:
        firing_rates: the firing rates of the neuron
        var: the bins at which the firing rate is calculated (0:1:nbins)
    
    RETURNS:
        y_fitted: the fitted Gaussian curve
        R: the R squared value of the fit
        params: the parameters of the Gaussian fit (A, mu, sigma)        
    """
    params_double, _ = curve_fit(double_gaussian, distances, firing_rates, p0=initial_guess_double, bounds = constraints)

    # Extract fitted parameters
    A1, mu1, sigma1, A2, mu2, sigma2 = params_double
    if verbose:
        print(f"Fitted parameters (Double Gaussian): A1 = {A1:.2f}, mu1 = {mu1:.2f}, sigma1 = {sigma1:.2f}, A2 = {A2:.2f}, mu2 = {mu2:.2f}, sigma2 = {sigma2:.2f}")

    # Generate points for the fitted Gaussian
    x_fitted = distances
    y_fitted_double = double_gaussian(x_fitted, A1, mu1, sigma1, A2, mu2, sigma2)
    R_double = compute_r_squared(firing_rates, y_fitted_double)
    
    return y_fitted_double, R_double, params_double

def compute_r_squared(y_observed, y_predicted):
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Define double Gaussian function
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    gaussian1 = A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    gaussian2 = A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    return gaussian1 + gaussian2

def gaussian_fitting(smoothed_firing_rates, distances, verbose = False):
    """For a single cell try to fit a single gaussian and a double gaussian"""

    peak_sep = [10,15] # the number of bins to exclude when looking for the peak of the second gaussian
    
    # Initialize variables
    fit_double = False
    double_wins = False
    params = np.zeros(3)
    y_fitted = np.zeros_like(smoothed_firing_rates)
    R = 0
    y_fitted_double = np.zeros_like(smoothed_firing_rates)
    R_double = 0
    prominent_peaks = []

    # some reused variables
    bounds_g2 = ([0, min(distances), 0, 0, min(distances), 0],  # Lower bounds
        [np.inf, max(distances), np.inf, np.inf, max(distances), np.inf])  # Upper bounds
    bounds_g1 = ([0, min(distances), 0], [np.inf, max(distances), np.inf])
    dist_std = np.std(distances)

    # Find peaks in firing rates
    import time
    t = time.time()
    peak_indices, _ = find_peaks(smoothed_firing_rates, height=0)  # Only positive peaks

    # Sort peaks by prominence
    if len(peak_indices) > 0:
        prominent_peaks = peak_indices[np.argmax(smoothed_firing_rates[peak_indices])]
        # fit single gaussian
        params = [smoothed_firing_rates[prominent_peaks], prominent_peaks, np.std(distances)]  # Initial guesses for A, mu, sigma
        
        try:
            y_fitted, R, params = fit_gaussian(smoothed_firing_rates, distances, initial_guess = params, constraints = bounds_g1)
        except:
            if verbose:
                print("Gaussian fit failed")
            return R, y_fitted, params, double_wins
        
        # fit double gaussian
        std = np.amin([peak_sep[1],np.amax([peak_sep[0],params[2]])])
        kept_peaks = peak_indices[np.logical_or(peak_indices < prominent_peaks - std, peak_indices > prominent_peaks + std)]
        if len(kept_peaks) > 0:
            second_peak = kept_peaks[np.argmax(smoothed_firing_rates[kept_peaks])]
            prominent_peaks = np.array([prominent_peaks, second_peak])
            params_double = [smoothed_firing_rates[prominent_peaks[0]],  # A1
                            prominent_peaks[0],  # mu1 (left peak)
                            dist_std,  # sigma1
                            smoothed_firing_rates[prominent_peaks[1]],  # A1
                            prominent_peaks[1],  # mu2 (right peak)
                            dist_std]   # sigma2
                    
            try:
                y_fitted_double, R_double, params_double = fit_double_gaussian(smoothed_firing_rates, distances, initial_guess_double = params_double, constraints = bounds_g2)
                A1, mu1, sigma1, A2, mu2, sigma2 = params_double

                # Separation of peaks
                peak_separation = abs(mu2 - mu1)
                max_sigma = max(sigma1, sigma2)
                distinct_peaks = peak_separation > 1 * max_sigma
                fit_double = True
            except:
                if verbose:
                    print("Double Gaussian fit failed")
                return R, y_fitted, params, double_wins

    if fit_double:
        if np.logical_and(R_double > R, distinct_peaks == True):
            y_fitted = y_fitted_double
            R = R_double
            params = params_double
            double_wins = True

    return R, y_fitted, params, double_wins


def leave_one_out_reliability(var, escape_matrix, cond, h_start, bins, n_cond, n_neur):
    """Computes for each cell in each condition the average correlation coefficient"""
    
    # initialize variables for output
    reliability = np.full((n_cond, n_neur), np.nan)
    fr_full = np.zeros((n_cond, n_neur, bins)) # conditions x neurons x n_bins
    c = [len([x for x in h_start if cond[x] == i]) for i in range(n_cond)]
    mat_num_cond = np.full((n_cond, n_neur, max(c),bins), np.nan) # conditions x neurons x trials x bins

    # iterate through conditions
    for i in range(n_cond):
        i = int(i)
        # start by condition
        cond_start = [x for x in h_start if cond[x] == i]
        cond_start.append(np.sum(cond == i))
        # iterate through neurons
        for j, n in enumerate(escape_matrix):
            # iterate through trials, pull out firing by bin
            for tr, _ in enumerate(cond_start[:-1]):
                neur = n[cond_start[tr]:cond_start[tr+1]]
                v = var[cond_start[tr]:cond_start[tr+1]]
                mat_num_cond[i, j, tr,:] = firing_by_bin_median_numba(v.astype(int), neur, bins, remove_empty = False)
            
            mat = mat_num_cond[i, j, :,:]
            # step 4: take median across trials
            all_nan_cols = np.all(np.isnan(mat), axis=0)
            smoothed_firing_rates = np.full(bins, np.nan)
            smoothed_firing_rates[~all_nan_cols] = np.nanmedian(mat[:,~all_nan_cols], axis = 0)
            # dump together for output
            fr_full[i, j, :] = smoothed_firing_rates

            # leave one out median
            tr_corr_coeff = np.full(mat_num_cond.shape[2], np.nan)
            tr_rms = np.full(mat_num_cond.shape[2], np.nan)
            if np.sum(smoothed_firing_rates) > 0:
                for tr in range(mat_num_cond.shape[2]):
                    loo_mat = np.delete(mat, tr, axis = 0)
                    all_nan_cols = np.all(np.isnan(loo_mat), axis=0)
                    loo = np.full(bins, np.nan)
                    loo[~all_nan_cols] = np.nanmedian(loo_mat[:,~all_nan_cols], axis = 0)
                    # corr coeff for each trial
                    id_nans = np.logical_or(np.isnan(loo),np.isnan(mat[tr,:]))
                    tr_corr_coeff[tr] = np.corrcoef(loo[~id_nans], mat[tr,~id_nans])[0, 1]
                    tr_rms[tr] = (np.sqrt(np.mean(mat[tr,:]**2)))
                # average across trial
                id_nans = np.logical_or(np.isnan(tr_corr_coeff), np.isnan(tr_rms))
                if np.sum(id_nans) < len(id_nans):
                    reliability[i,j] = np.average(tr_corr_coeff[~id_nans], weights = tr_rms[~id_nans])
    
    return fr_full, mat_num_cond, reliability

def tuning_method_no_trials(var, escape_matrix, cond, bins, n_cond, n_neur):
    """Filter (gauss or savgol) the full trace -> take median across all time"""
    # step 1: filter the full trace
    # Assuming the filtered trace is what is passed as an arg to extract_homing_and_escape_periods

    # step 2: extract relevant neuron x time matrix
    # assumed to be one of the inputs

    # initialize variables for output
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan) # conditions x neurons x n_bins
    R_full = np.zeros((n_neur, n_cond)) # neurons x conditions
    fr_full = np.zeros((n_cond, n_neur, bins)) # conditions x neurons x n_bins
    params_full = np.full((n_neur, n_cond, 6), np.nan)

    # step 3: compute firing by bin across all time
    for c in range(n_cond):
        neur = escape_matrix[:,cond == c]
        v = var[cond == c]
        for i, n in enumerate(neur):
            smoothed_firing_rates = firing_by_bin_median_numba(v.astype(int), n, bins, remove_empty = False)
            
            # step 4: gaussian fit
            R, y, params, _ = gaussian_fitting(smoothed_firing_rates[~np.isnan(smoothed_firing_rates)], np.arange(np.sum(~np.isnan(smoothed_firing_rates))), verbose = False)
            y_fitted = np.full_like(smoothed_firing_rates, np.nan)
            y_fitted[~np.isnan(smoothed_firing_rates)] = y

            # dump together for output
            fr_full[c, i, :] = smoothed_firing_rates
            R_full[i, c] = R
            params_full[i, c,:len(params)] = params
            y_fitted_full[c, i, :len(y_fitted)] = y_fitted

    return y_fitted_full, R_full, fr_full, params_full

def tuning_method(var, escape_matrix, cond, h_start, bins, n_cond, n_neur):
    """Filter (gauss or savgol) the full trace -> take median per trial -> take median across trials -> fit gaussian
    INPUTS:
    
    RETURNS:
        y_fitted_full: matrix of conditions x neurons x n_bins of the gaussian fits of the average data across trials
        R_full: matrix of neurons x conditions of the R^2 of the gaussian fits
        fr_full: matrix of conditions x neurons x n_bins of the average data across trials
        params_full: matrix of neurons x conditions x 6 of the parameters of the gaussian fit. 
            If single gaussian is best = [Amplitude, mu, sigma, nan, nan, nan]; 
            if double gaussian fit is best = [Amplitude1, mu1, sigma1, Aplitude2, mu2, sigma2]
        mat_num_cond: matrix of conditions x neurons x trials x n_bins of the average data per trial
    """
    # step 1: filter the full trace
    # Assuming the filtered trace is what is passed as an arg to extract_homing_and_escape_periods
    
    # step 2: extract homie periods
    # This is done outside as well

    # initialize variables for output
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan) # conditions x neurons x n_bins
    R_full = np.zeros((n_neur, n_cond)) # neurons x conditions
    fr_full = np.zeros((n_cond, n_neur, bins)) # conditions x neurons x n_bins
    params_full = np.full((n_neur, n_cond, 6), np.nan)
    c = [len([x for x in h_start if cond[x] == i]) for i in range(n_cond)] # what is the max number of trials across all conditions
    mat_num_cond = np.full((n_cond, n_neur, max(c),bins), np.nan) # conditions x neurons x trials x bins

    # step 3: compute firing per bin per trial
    # iterate through conditions
    for i in range(n_cond):
        i = int(i)
        # start by condition
        cond_start = [x for x in h_start if cond[x] == i]
        cond_start.append(np.sum(cond == i))

        # iterate through neurons
        for j, n in enumerate(escape_matrix):
            # iterate through trials, pull out firing by bin
            for tr, _ in enumerate(cond_start[:-1]):
                neur = n[cond_start[tr]:cond_start[tr+1]]
                v = var[cond_start[tr]:cond_start[tr+1]]
                mat_num_cond[i, j, tr,:] = firing_by_bin_median_numba(v.astype(int), neur, bins, remove_empty = False)
            
            # step 4: take median across trials
            mat = mat_num_cond[i, j, :,:]
            all_nan_cols = np.all(np.isnan(mat), axis=0)
            smoothed_firing_rates = np.full(bins, np.nan)
            smoothed_firing_rates[~all_nan_cols] = np.nanmedian(mat[:,~all_nan_cols], axis = 0)

            # step 5: gaussian fit
            R, y, params, double_wins = gaussian_fitting(smoothed_firing_rates[~np.isnan(smoothed_firing_rates)], np.arange(np.sum(~np.isnan(smoothed_firing_rates))), verbose = False)
            y_fitted = np.full_like(smoothed_firing_rates, np.nan)
            y_fitted[~np.isnan(smoothed_firing_rates)] = y

            # dump together for output
            fr_full[i, j, :] = smoothed_firing_rates
            R_full[j, i] = R
            params_full[j, i,:len(params)] = params
            y_fitted_full[i, j, :len(y_fitted)] = y_fitted
            
    return y_fitted_full, R_full, fr_full, params_full, mat_num_cond

##------------ TUNING FUNCTIONS USING PARALLEL PROCESSING
##------------ WARNING: THESE ARE SLOWER THAN THE UNPARALLEL ONES

def tuning_method_no_trials_parallel_function(shared_name, shape, dtype, i, neuron_index, cond_indices, bins, var):
    """Worker function to process a single neuron using shared memory.
    Process a single neuron for a given condition."""
    
    # Reconnect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shared_name)
    escape_matrix = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Extract relevant data for this neuron
    n = escape_matrix[neuron_index, cond_indices]  # Avoid passing large slices through multiprocessing
    v = var[cond_indices]

    # Compute firing rates
    smoothed_firing_rates = firing_by_bin_median_numba(v.astype(int), n, bins, remove_empty=False)

    # Gaussian fitting
    valid_idx = ~np.isnan(smoothed_firing_rates)
    y_fitted = np.full_like(smoothed_firing_rates, np.nan)
    if np.any(valid_idx):
        R, y, params, _ = gaussian_fitting(smoothed_firing_rates[valid_idx], np.arange(np.sum(valid_idx)), verbose=False)
        y_fitted[valid_idx] = y
    else:
        R, params = 0, np.full(6, np.nan)

    existing_shm.close()  # Close shared memory connection in worker

    return i, neuron_index, smoothed_firing_rates, R, params, y_fitted

def tuning_method_no_trials_with_pool(var, escape_matrix, cond, bins, n_cond, n_neur):
    """Optimized version using shared memory and parallelization for neurons.
    Filter (gauss or savgol) the full trace -> take median across all time"""
    
    # Create shared memory for escape_matrix
    escape_shm = shared_memory.SharedMemory(create=True, size=escape_matrix.nbytes)
    shared_escape_matrix = np.ndarray(escape_matrix.shape, dtype=escape_matrix.dtype, buffer=escape_shm.buf)
    np.copyto(shared_escape_matrix, escape_matrix)  # Copy data into shared memory

    # Initialize output arrays
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan)
    R_full = np.zeros((n_neur, n_cond))
    fr_full = np.zeros((n_cond, n_neur, bins))
    params_full = np.full((n_neur, n_cond, 6), np.nan)

    # Initialize persistent multiprocessing pool
    PPool = PersistentPool()
    print('Pool set up!')

    # Step 3: Compute firing by bin across all time
    for c in range(n_cond):
        cond_indices = np.where(cond == c)[0]  # Precompute condition indices to avoid slicing overhead

        # Prepare arguments for multiprocessing
        args_list = [(escape_shm.name, escape_matrix.shape, escape_matrix.dtype, c, i, cond_indices, bins, var) for i in range(n_neur)]

        # Use multiprocessing pool to process neurons in parallel
        results = PPool.mp_pool.starmap(tuning_method_no_trials_parallel_function, args_list)

        # Store results
        for _, i, smoothed_firing_rates, R, params, y_fitted in results:
            fr_full[c, i, :] = smoothed_firing_rates
            R_full[i, c] = R
            params_full[i, c, :len(params)] = params
            y_fitted_full[c, i, :len(y_fitted)] = y_fitted

    # Cleanup shared memory
    escape_shm.close()
    escape_shm.unlink()
    PPool.close()

    return y_fitted_full, R_full, fr_full, params_full

def tuning_method_parallel_function(shared_name, shape, dtype, i, j, neuron_index, cond_start, var, bins):
    """Worker function to process a single neuron using shared memory.
    """
    
    # Reconnect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shared_name)
    escape_matrix = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Extract relevant data for this neuron
    n = escape_matrix[neuron_index, :]  # Direct access to neuron data
    max_trials = len(cond_start) - 1  # Number of trials
    mat_num_cond_i_j = np.full((max_trials, bins), np.nan)

    # Step 3: Compute firing per bin per trial
    for tr in range(max_trials):
        neur = n[cond_start[tr]:cond_start[tr + 1]]
        v = var[cond_start[tr]:cond_start[tr + 1]]
        mat_num_cond_i_j[tr, :] = firing_by_bin_median_numba(v.astype(int), neur, bins, remove_empty=False)

    # Step 4: Take median across trials
    all_nan_cols = np.all(np.isnan(mat_num_cond_i_j), axis=0)
    smoothed_firing_rates = np.full(bins, np.nan)
    smoothed_firing_rates[~all_nan_cols] = np.nanmedian(mat_num_cond_i_j[:, ~all_nan_cols], axis=0)

    # Step 5: Gaussian fit
    valid_idx = ~np.isnan(smoothed_firing_rates)
    if np.any(valid_idx):
        R, y, params, _ = gaussian_fitting(smoothed_firing_rates[valid_idx], np.arange(np.sum(valid_idx)), verbose=False)
        y_fitted = np.full_like(smoothed_firing_rates, np.nan)
        y_fitted[valid_idx] = y
    else:
        R, y_fitted, params = 0, np.full_like(smoothed_firing_rates, np.nan), np.full(6, np.nan)

    existing_shm.close()  # Close shared memory connection in worker

    return j, smoothed_firing_rates, R, params, y_fitted, mat_num_cond_i_j


def tuning_method_with_pool(var, escape_matrix, cond, h_start, bins, n_cond, n_neur):
    """Optimized version using shared memory and parallelization for neurons.
    Filter (gauss or savgol) the full trace -> take median per trial -> take median across trials -> fit gaussian
    INPUTS:
    
    RETURNS:
        y_fitted_full: matrix of conditions x neurons x n_bins of the gaussian fits of the average data across trials
        R_full: matrix of neurons x conditions of the R^2 of the gaussian fits
        fr_full: matrix of conditions x neurons x n_bins of the average data across trials
        params_full: matrix of neurons x conditions x 6 of the parameters of the gaussian fit. 
            If single gaussian is best = [Amplitude, mu, sigma, nan, nan, nan]; 
            if double gaussian fit is best = [Amplitude1, mu1, sigma1, Aplitude2, mu2, sigma2]
        mat_num_cond: matrix of conditions x neurons x trials x n_bins of the average data per trial
    """

    # Create shared memory for escape_matrix
    escape_shm = shared_memory.SharedMemory(create=True, size=escape_matrix.nbytes)
    shared_escape_matrix = np.ndarray(escape_matrix.shape, dtype=escape_matrix.dtype, buffer=escape_shm.buf)
    np.copyto(shared_escape_matrix, escape_matrix)  # Copy data into shared memory

    # Initialize output arrays
    y_fitted_full = np.full((n_cond, n_neur, bins), np.nan)
    R_full = np.zeros((n_neur, n_cond))
    fr_full = np.zeros((n_cond, n_neur, bins))
    params_full = np.full((n_neur, n_cond, 6), np.nan)
    c = [len([x for x in h_start if cond[x] == i]) for i in range(n_cond)]
    mat_num_cond = np.full((n_cond, n_neur, max(c), bins), np.nan)

    # Initialize persistent multiprocessing pool
    PPool = PersistentPool()
    print('Pool set up!')

    # Step 3: Compute firing per bin per trial
    for i in range(n_cond):
        cond_start = [x for x in h_start if cond[x] == i]
        cond_start.append(np.sum(cond == i))

        # Prepare arguments for multiprocessing
        args_list = [(escape_shm.name, escape_matrix.shape, escape_matrix.dtype, i, j, j, cond_start, var, bins) for j in range(n_neur)]

        # Use multiprocessing pool to process neurons in parallel
        results = PPool.mp_pool.starmap(tuning_method_parallel_function, args_list)

        # Store results
        for j, smoothed_firing_rates, R, params, y_fitted, mat_num_cond_i_j in results:
            fr_full[i, j, :] = smoothed_firing_rates
            R_full[j, i] = R
            params_full[j, i, :len(params)] = params
            y_fitted_full[i, j, :len(y_fitted)] = y_fitted
            mat_num_cond[i, j, :mat_num_cond_i_j.shape[0], :] = mat_num_cond_i_j  # Store trial-wise data

    # Cleanup shared memory
    escape_shm.close()
    escape_shm.unlink()

    return y_fitted_full, R_full, fr_full, params_full, mat_num_cond