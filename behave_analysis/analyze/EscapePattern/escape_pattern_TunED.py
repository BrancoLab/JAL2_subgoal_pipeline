from behave_analysis.analyze.EscapePattern.ComputeEscapeTuning import ComputeEscapeTuning
from behave_analysis.analyze.EscapePattern.TunED import TunED
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import (parse_residual_string, 
                                                                        get_homings_onsets_in_filtered_time)
from behave_analysis.analyze.EscapePattern.median_functions import firing_by_bin_median_numba, trial_median_firing, firing_by_bin_winz_mean
from behave_analysis.utils.creating_directories import make_directory

import numpy as np
import os
import dill as pickle
from loguru import logger

def escape_pattern_TunED(aefizz, variable, settings):
    """Identify driver variable using TunEd method"""

    """1. Load EscapePattern data for driver and passenger variables"""
    # parse variable string
    var1, time_period1, var2, time_period2 = parse_residual_string(variable)
    if time_period1 != time_period2:
        raise ValueError("TunED analysis requires both variables to be from the same time period")

    # load data from EscapeTuning objects
    v1_full, mu_v1_full, v1_cond, neural_matrix, tr_start = load_vars_escape_tuning(aefizz, var1, time_period1, settings)
    v2_full, mu_v2_full, v2_cond, _ = load_vars_escape_tuning(aefizz, var2, time_period2, settings)

    # check conditions are the same
    if not np.array_equal(v1_cond, v2_cond):
        raise ValueError("Conditions for the two variables do not match in TunED analysis")
    
    n_neur = np.shape(mu_v1_full)[1]
    n_cond = np.shape(mu_v1_full)[0]

    """2. Compute TunED"""
    tuned_settings = {'bin_edges': np.arange(settings.escape_tuning_bins + 1),
                      'compare_method': 'euclidean', # or 'cosine'
                      'stats': 'bootstrap',  # or 'linear_shift' (not yet implemented)
                      'stats_samples': 100}
    
    # Pre-allocate result matrices
    distance = np.full((n_cond, n_neur), np.nan)
    distance_bs = np.full((n_cond, n_neur, tuned_settings['stats_samples']), np.nan)
    v1_significant = np.zeros((n_cond, n_neur), dtype=bool)
    v2_significant = np.zeros((n_cond, n_neur), dtype=bool)

    for c in range(n_cond): # which condition do we want to look at

        cond_start = []
        if "homing" in time_period1 or "escape" in time_period1: 
            # start by condition
            cond_start = [x for x in tr_start if v1_cond[x] == int(c)]
            cond_start = np.concatenate((cond_start,[np.sum(v1_cond < int(c)+1)])) # this adds the end of the last trial
        
        # Step 1: Real Observed Tuning Curves and their variance

        # select v1 and v2 for this condition
        v1 = v1_full[v1_cond == c]
        v2 = v2_full[v1_cond == c]

        for idx in range(n_neur):
            
            # select average firing by bin tuning curve for this neuron and condition
            mu_v1 = mu_v1_full[c, idx,:]
            mu_v2 = mu_v2_full[c, idx,:]

            TunED_instance = TunED(v1, v2, mu_v1, mu_v2, tuned_settings).independence_test()

            # Store results
            distance[c, idx] = TunED_instance.distance

        # 2. Significance testing of TunED results
        if tuned_settings['stats'] == 'bootstrap':
            
            # create a matrix of resampling vectors (n = tuned_settings['stats_samples']) for this condition with replacement
            frames_vec = np.arange(np.sum(v1_cond == c))
            resampled_vector = np.random.choice(frames_vec, (tuned_settings['stats_samples'], len(frames_vec)), replace=True)
            
            # select neural activity for this condition
            escape_matrix = neural_matrix[:, v1_cond == c]

            for i, v in enumerate(resampled_vector):
                # resample behavioral variables and neural activity
                escape_matrix_bs = escape_matrix[:,v]
                v1_bs = v1[v]
                v2_bs = v2[v]

                # compute tuning curves for resampled data
                mu_v1_bs = compute_avg_firing_tuning_curve(escape_matrix_bs, v1_bs, tuned_settings['bin_edges'], trial_start=cond_start)
                mu_v2_bs = compute_avg_firing_tuning_curve(escape_matrix_bs, v2_bs, tuned_settings['bin_edges'], trial_start=cond_start)
                
                for idx in range(n_neur):
                    # compute TunED on resampled data
                    TunED_instance = TunED(v1_bs, v2_bs, mu_v1_bs[idx,:], mu_v2_bs[idx,:], tuned_settings).independence_test()

                    # Store results
                    distance_bs[c, idx, i] = TunED_instance.distance
                    
            # Determine significance
            lower_percentile, upper_percentile = np.percentile(distance_bs[c,:,:], [2.5, 97.5], axis = 0)
            v1_significant[c, :] = (lower_percentile > 0) & (upper_percentile > 0)
            v2_significant[c, :] = (lower_percentile < 0) & (upper_percentile < 0)

    """3. Save results"""
    savepath = make_directory(os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escape_tuning", time_period1))
    filename = savepath + os.sep + "TunED_" + var1 + "_vs_" + var2 + "_" + str(settings.escape_tuning_bins) + "bins.pkl"
    np.savez(filename,
             variable=variable,
             distance=distance,
             distance_bs=distance_bs,
             v1_significant=v1_significant,
             v2_significant=v2_significant,
             settings=tuned_settings)

def load_vars_escape_tuning(aefizz, var, time_period, settings):
    """Load EscapeTuning objects for both variables in the TunED analysis. If not found, compute them.
    Return behavioral variable and tuning curves."""

    path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escape_tuning", time_period)
    filename = path + os.sep + var + "_" + str(settings.escape_tuning_bins) + "bins.pkl"

    # check file exists
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            CT = pickle.load(f)
    else:
        logger.warning(f"Escape tuning to {var} in {time_period} file not found, computing now...")
        computeET = ComputeEscapeTuning(var + " in " + time_period, aefizz=aefizz)
        computeET.extract_data_and_tuning(aefizz=aefizz)
        computeET.compute_statistical_significance(aefizz=aefizz)
        computeET.save_escape_tuning()
        CT = computeET.ET

    if "homing" in time_period or "escape" in time_period:
        trial_start = get_homings_onsets_in_filtered_time(CT.homing_vector)
    else:
        trial_start = []

    return CT.discretized_var, CT.fr_full, CT.condition, CT.neural_matrix, trial_start

def compute_avg_firing_tuning_curve(neural_activity, variable, Nbins, trial_start = [], avg = "winsorized"):
    """Make a function that uses the median calculations from tuning_functions
    so it needs to check if using trials or not.
    """
    
    smoothed_firing_rates = np.full((neural_activity.shape[0], Nbins), np.nan)
    
    # if computing tuning curves on data with trials (e.g. homings or escapes)
    if len(trial_start) > 0:
        # iterate through neurons
        for j, n in enumerate(neural_activity):
            mat = np.full((len(trial_start)-1, Nbins), np.nan)  # trials x bins
            # iterate through trials, pull out firing by bin
            for tr, _ in enumerate(trial_start[:-1]):
                neur = n[trial_start[tr] : trial_start[tr + 1]]
                v = variable[trial_start[tr] : trial_start[tr + 1]]
                mat[tr, :] = firing_by_bin_median_numba(v.astype(int), neur,Nbins, remove_empty=False)

            smoothed_firing_rates[j,:] = trial_median_firing(mat, avg)

    else:
        # iterate through neurons
        for j, n in enumerate(neural_activity):
            if avg == 'median':
                smoothed_firing_rates[j,:] = firing_by_bin_median_numba(variable.astype(int), n, Nbins, remove_empty = False)
            elif avg == 'winsorized':
                smoothed_firing_rates[j,:] = firing_by_bin_winz_mean(variable.astype(int), n, Nbins, remove_empty = False)

    return smoothed_firing_rates