import numpy as np
from scipy.stats import zscore
from rastermap import Rastermap
import pandas as pd

def make_escape_bool(h_e_bool, full_e_bool, h_cond):
    """A function that takes in the homing and escape boolean and returns a list of start times for the homings and escapes."""
    # starts of homings & escapes
    starts = np.where(np.diff(h_e_bool.astype(int)) > 0)[0] + 1
    ends = np.where(np.diff(h_e_bool.astype(int)) < 0)[0] + 1
    homie_lengths = ends - starts
    counter, h_start, e_start = 0, np.full(len(homie_lengths), 0), np.full(len(homie_lengths) + 1, False)
    for i, h in enumerate(homie_lengths):
        if full_e_bool[starts[i]]:
            e_start[i] = True  # a bool that tells us which one of the h_starts are escapes
        h_start[i] = counter
        counter += h
    h_start = np.append(h_start, len(h_cond))  # a list of the start indices of homings + escapes in the homing/escape time period
    e_bool = full_e_bool[h_e_bool]  # a boolean that tells us which periods of the homing/escape are escapes

    return h_start, e_bool

def make_long_homing_bool(h_start, X, Y):
    """A functon that identifies which homings and escapes are long: defined as ones that go from threat zone all the way to shelter"""
    # where did the mouse start and end
    y_start = Y[h_start[:-1]]
    y_end = Y[h_start[1:] - 1]
    long_homie_bool = np.full(len(h_start[:-1]), False)
    homie_id = np.full(len(X), 0)  # a vector that increases with each homing
    for h, (s, e) in enumerate(zip(y_start, y_end)):
        homie_id[h_start[h] : h_start[h + 1]] = h
        if (s < 512) & (e > 700):
            long_homie_bool[h] = True

    return long_homie_bool


def pre_homing_bool(h_e_bool, time_s, long_homie_bool=[], e_start=[]):
    """A function that takes in the homing and escape boolean and returns a boolean for the time before the homing starts.
    It also returns a list of the homings that are escapes."""

    time = int(time_s * 40)  # convert seconds to frames
    # find the times before the homie starts
    homie = np.where(np.diff(h_e_bool.astype(int)) > 0)[0] + 1

    # if we're not given the long_homie_bool, we make a dummy one - this means we don't care about isolating the long homies
    if len(long_homie_bool) == 0:
        long_homie_bool = np.full(len(homie), True)
    # if we're not given the e_start, we make a dummy one - this means we don't care about isolating the escapes
    if len(e_start) == 0:
        e_start = np.full(len(homie), False)

    # make variables
    prebool = np.full(len(h_e_bool), False)
    e_long = []
    counter = 0
    for idx, i in enumerate(homie):
        if long_homie_bool[idx]:
            prebool[i-time: i] = True
            counter += 1
            if e_start[idx]:
                e_long.append(counter)

    return prebool, e_long

def run_rastermap(escape_matrix):
  # rastermap with running speed and y position

    fit_spks = zscore(escape_matrix.T, axis=1) # need neurons x time
    if np.shape(fit_spks)[0] > 200:
        model = Rastermap(n_clusters=100, # number of clusters to compute
                            n_PCs=128, # number of PCs to use
                            locality=0., # locality in sorting to find sequences (this is a value from 0-1)
                            grid_upsample=10, # default value, 10 is good for large recordings
                            ).fit(fit_spks)
    else:
        model = Rastermap(n_clusters=None, # None turns off clustering and sorts single neurons 
                        n_PCs=64, # use fewer PCs than neurons
                        locality=0.15, # some locality in sorting (this is a value from 0-1)
                        time_lag_window=40, # use future timepoints to compute correlation
                        grid_upsample=0, # 0 turns off upsampling since we're using single neurons
                        ).fit(fit_spks)
    isort = model.isort
    return isort, fit_spks

def generate_event_seq(n_window, threshold, window_samples, method = 'first_activity'):
    """INPUTS:
            n_window: 2D array of neural activity (time x neurons)
            threshold: zscore threshold for activity
            window_samples: number of samples in the window (e.g. 500ms = 200 samples at 40Hz)
            method: method for generating event sequence (default is 'first_activity')
    """
    # event_seq based on first activity as moment when it crosses threshold
    if method == 'first_activity':
        first_activity = np.zeros(n_window.shape[1])
        for n in range(n_window.shape[1]):
            if len(np.where((n_window[:,n] > threshold) == True)[0]) > 0:
                first_activity[n] = np.where((n_window[:,n] > threshold) == True)[0][0]
        event_seq = np.argsort(first_activity)

    # event seq based on weighted average of activity to find where the bump is
    if method == 'weighted_avg':
        weighted_avg = np.zeros(n_window.shape[1])
        for n in range(n_window.shape[1]):
            weighted_avg[n] = np.sum(n_window[:,n] * np.arange(1, window_samples+1)) / np.sum(n_window[:,n])
        event_seq = np.argsort(weighted_avg)

    # event seq based on rastermap sorting
    if method == 'rastermap':
        event_seq, _ = run_rastermap(n_window)

    return event_seq

def df_to_count_array(df, all_clusters, columns, fill_time = []):
    """Convert a DataFrame to a count array for each cluster and time bin.
    Args:
        df (pd.DataFrame): DataFrame containing spike data.
        all_clusters (np.ndarray): Array of all unique cluster IDs.
        columns: the name of the column to use for the time bins.
        fill_time: can be empty and then we don't fill missing bins, 
                    it can be a list of the start time and end time that we want to fill between, 
                    or it can be an int of the max time bin difference that we want to fill to."""
    
    count_matrix = pd.pivot_table(
        df,
        index='spike_clusters',    # rows: clusters
        columns=columns,        # columns: time bins
        values='aligned_spike_times',
        aggfunc='count',
        fill_value=0
    )
    # Get all unique cluster IDs from the full spike_df
    count_matrix = count_matrix.reindex(all_clusters, fill_value=0)

    if isinstance(fill_time, list):
        if len(fill_time) == 0:
            return count_matrix
        elif len(fill_time) == 2:
            all_time_bins = np.arange(fill_time[0], fill_time[1] + 1)
            # Reindex columns to include all bins, filling missing ones with zeros
            count_matrix = count_matrix.reindex(columns=all_time_bins, fill_value=0)
    elif isinstance(fill_time, int):
        # Find gaps in time_bin columns
        col_array = np.array(count_matrix.columns)
        diffs = np.diff(col_array)
        gap_mask = (diffs > 1) & (diffs < fill_time)
        missing_bins = []
        for i, is_gap in enumerate(gap_mask):
            if is_gap:
                # Add all missing bins in this gap
                missing = np.arange(col_array[i] + 1, col_array[i + 1])
                missing_bins.extend(missing)

        # Add missing columns with zeros
        if missing_bins:
            # Create a DataFrame of zeros with the same index as count_matrix and columns as missing_bins
            zeros_df = pd.DataFrame(0, index=count_matrix.index, columns=missing_bins)
            # Concatenate along columns
            count_matrix = pd.concat([count_matrix, zeros_df], axis=1)
            # Re-sort columns to maintain order
            count_matrix = count_matrix.reindex(sorted(count_matrix.columns), axis=1)
            # Optionally, defragment the DataFrame
            count_matrix = count_matrix.copy()

        # Re-sort columns to maintain order
        count_matrix = count_matrix.reindex(sorted(count_matrix.columns), axis=1)

    return count_matrix