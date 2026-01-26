import numpy as np
import pandas as pd


def prepare_state_space_decoder_data(spike_df, time_mask, bin_width):
    """Prepare data for state space decoder analysis.

    Args:
        spike_df (pd.DataFrame): DataFrame containing spike data with columns 'spike_aligned_to_frame', 'spike_clusters', and 'aligned_spike_times'.
        session (object): Session object containing camera trigger information.
        time_mask (np.ndarray): Boolean array indicating valid time points.
        sig_cells (np.ndarray): Boolean array indicating significant cells for each condition.
        preferred_tuning (np.ndarray): Array of preferred tuning values for cells.
        cond (int): Condition index to filter data.
        bin_width (float, optional): Width of time bins in seconds. Defaults to 0.1.

    Returns:
        tuple: count_array, spikes, time, time_bin_to_frame

    # TODO: the alignment is not perfect,
    # there is a difference of 1 frame between time_bin_to_frame and frame_for_bin at bin transition times
    # this is likely due to rounding errors in the binning process?!
    """
    spike_frames = spike_df["spike_aligned_to_frame"].to_numpy().astype(int) - 1
    mask = time_mask[spike_frames]
    spike_df_filt = spike_df[mask].copy()

    # Create a new column for the time bin index
    all_clusters = np.sort(spike_df["spike_clusters"].unique())
    spike_df_filt["time_bin"] = (spike_df_filt["aligned_spike_times"] // bin_width).astype(int)
    count_matrix = df_to_count_array(spike_df_filt, all_clusters, "time_bin", fill_time=10)

    count_array = count_matrix.to_numpy()
    count_array = (count_array > 0).astype(int)  # still have more than one spike in some cases...
    spikes = count_array

    # create a time vector in seconds
    time = np.arange(0, spikes.shape[1] / (1 / bin_width), bin_width)

    return spikes, time


def df_to_count_array(df, all_clusters, columns, fill_time=[]):
    """Convert a DataFrame to a count array for each cluster and time bin.
    Args:
        df (pd.DataFrame): DataFrame containing spike data.
        all_clusters (np.ndarray): Array of all unique cluster IDs.
        columns: the name of the column to use for the time bins.
        fill_time: can be empty and then we don't fill missing bins,
                    it can be a list of the start time and end time that we want to fill between,
                    or it can be an int of the max time bin difference that we want to fill to."""

    count_matrix = pd.pivot_table(df, index="spike_clusters", columns=columns, values="aligned_spike_times", aggfunc="count", fill_value=0)  # rows: clusters  # columns: time bins
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
