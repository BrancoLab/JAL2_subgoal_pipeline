import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def gaussian_fitting(smoothed_firing_rates, distances, verbose=False):
    """For a single cell try to fit a single gaussian and a double gaussian"""

    peak_sep = [10, 15]  # the number of bins to exclude when looking for the peak of the second gaussian

    # Initialize variables
    fit_double = False
    double_wins = False
    params = np.zeros(3)
    y_fitted = np.full(len(smoothed_firing_rates), np.nan)
    R = 0
    y_fitted_double = np.full(len(smoothed_firing_rates), np.nan)
    R_double = 0
    prominent_peaks = []

    # some reused variables
    bounds_g2 = ([0, min(distances), 0, 0, min(distances), 0], [np.inf, max(distances), np.inf, np.inf, max(distances), np.inf])  # Lower bounds  # Upper bounds
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
            y_fitted, R, params = fit_gaussian(smoothed_firing_rates, distances, initial_guess=params, constraints=bounds_g1)
        except:
            if verbose:
                print("Gaussian fit failed")
            return R, y_fitted, params, double_wins

        # fit double gaussian
        std = np.amin([peak_sep[1], np.amax([peak_sep[0], params[2]])])
        kept_peaks = peak_indices[np.logical_or(peak_indices < prominent_peaks - std, peak_indices > prominent_peaks + std)]
        if len(kept_peaks) > 0:
            second_peak = kept_peaks[np.argmax(smoothed_firing_rates[kept_peaks])]
            prominent_peaks = np.array([prominent_peaks, second_peak])
            params_double = [
                smoothed_firing_rates[prominent_peaks[0]],  # A1
                prominent_peaks[0],  # mu1 (left peak)
                dist_std,  # sigma1
                smoothed_firing_rates[prominent_peaks[1]],  # A1
                prominent_peaks[1],  # mu2 (right peak)
                dist_std,
            ]  # sigma2

            try:
                y_fitted_double, R_double, params_double = fit_double_gaussian(smoothed_firing_rates, distances, initial_guess_double=params_double, constraints=bounds_g2)
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


def fit_gaussian(firing_rates, distances, initial_guess=[], constraints=[], verbose=False):
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
    if len(constraints) == 0:
        params, _ = curve_fit(gaussian, distances, firing_rates, p0=initial_guess)
    else:
        params, _ = curve_fit(gaussian, distances, firing_rates, p0=initial_guess, bounds=constraints)

    # Extract parameters
    A, mu, sigma = params
    if verbose:
        print(f"Fitted parameters: A = {A:.2f}, mu = {mu:.2f}, sigma = {sigma:.2f}")

    # Generate points for the fitted Gaussian
    x_fitted = distances
    y_fitted = gaussian(x_fitted, A, mu, sigma)
    R = compute_r_squared(firing_rates, y_fitted)

    return y_fitted, R, params


def fit_double_gaussian(firing_rates, distances, initial_guess_double, constraints=[], verbose=False):
    """Fit a Gaussian to the data and return the fitted curve, R squared and parameters

    INPUTS:
        firing_rates: the firing rates of the neuron
        var: the bins at which the firing rate is calculated (0:1:nbins)

    RETURNS:
        y_fitted: the fitted Gaussian curve
        R: the R squared value of the fit
        params: the parameters of the Gaussian fit (A, mu, sigma)
    """
    params_double, _ = curve_fit(double_gaussian, distances, firing_rates, p0=initial_guess_double, bounds=constraints)

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
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# Define double Gaussian function
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    gaussian1 = A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
    gaussian2 = A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
    return gaussian1 + gaussian2
