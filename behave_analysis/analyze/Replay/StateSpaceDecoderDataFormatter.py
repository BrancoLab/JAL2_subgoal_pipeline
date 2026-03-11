import numpy as np
import pandas as pd
from loguru import logger

sampling_rate = 30000  # Hz

def prepare_state_space_decoder_data(spike_df, clusters, time_mask, session, bin_width, validate_alignment=False):
    
    # cluster selection
    all_clusters = np.sort(spike_df["spike_clusters"].unique())
    selected_clusters = all_clusters[clusters]

    # convert spike time to bin index (see #2 below)
    eps = 1e-9
    spike_df2 = spike_df.copy()
    spike_df2["time_bin"] = np.floor((spike_df2["aligned_spike_times"].to_numpy() + eps) / bin_width).astype(np.int64)

    # build global bin range from recording duration
    frame_onsets = session.camera_trigger.frame_trigger_onsets_idx.astype(np.int64)  # samples @30kHz
    bin_samples = int(round(bin_width * sampling_rate))
    max_sample = int(frame_onsets[-1])  # or a better end-of-recording sample if available
    all_bin_starts_samples = np.arange(0, max_sample + 1, bin_samples, dtype=np.int64)
    all_bin_ids = np.arange(all_bin_starts_samples.size, dtype=np.int64)

    # map each bin -> frame
    frame_for_all_bins = np.searchsorted(frame_onsets, all_bin_starts_samples, side="right") - 1
    frame_for_all_bins = np.clip(frame_for_all_bins, 0, len(frame_onsets) - 1)

    # keep bins whose mapped frame is in mask
    keep_bins = all_bin_ids[time_mask[frame_for_all_bins]]
    frame_for_bin = frame_for_all_bins[time_mask[frame_for_all_bins]]

    # keep only spikes in selected clusters and in kept bins
    spike_df2 = spike_df2[spike_df2["spike_clusters"].isin(selected_clusters)]
    spike_df2 = spike_df2[spike_df2["time_bin"].isin(keep_bins)]

    # count matrix and force full kept-bin coverage (zeros where no spikes)
    count_matrix = pd.pivot_table(
        spike_df2, index="spike_clusters", columns="time_bin",
        values="aligned_spike_times", aggfunc="count", fill_value=0
    )
    count_matrix = count_matrix.reindex(selected_clusters, fill_value=0)
    count_matrix = count_matrix.reindex(columns=keep_bins, fill_value=0)

    spikes = (count_matrix.to_numpy() > 0).astype(int)
    # time = keep_bins * bin_width
    time = np.arange(0, spikes.shape[1] / (1 / bin_width), bin_width)

    # build a vector of length n_time_bins that tracks the segments of time from time_mask
    segments_mask = time_segment_vector(keep_bins)
    segments_in_bins = np.array([np.sum(segments_mask == s)/(1/bin_width) for s in np.unique(segments_mask)])
    segments_in_behave_time = (np.where(np.diff(time_mask.astype(int)) < 0)[0] - np.where(np.diff(time_mask.astype(int)) > 0)[0])/session.video.fps
    assert np.array_equal(segments_in_bins, segments_in_behave_time), "Segments in bins and segments in behave time don't match, check time_segment_vector function!"

    # Validate frame_for_bin against spike data
    if validate_alignment:
        validate_frame_to_bin_alignment(spike_df2, frame_for_bin)

    return spikes, time, frame_for_bin, segments_mask

def validate_frame_to_bin_alignment(spike_df_filt, frame_for_bin):
    """Helper function to validate the alignment between spike frames and searchsorted frames.
    This is important because there can be discrepancies due to rounding errors in binning."""
    spike_bin_frames = spike_df_filt.groupby("time_bin")["spike_aligned_to_frame"].apply(lambda x: x.unique())
    mismatches = []
    for bin_idx, frames in spike_bin_frames.items():
        if bin_idx < len(frame_for_bin):
            # Check if all spikes in this bin agree on frame
            frames_unique = np.unique(frames)
            if len(frames_unique) > 1:
                logger.warning(f"Bin {bin_idx}: spikes from multiple frames: {frames_unique}")
            # Compare spike frame to searchsorted frame (spike frame is 1-indexed, frame_for_bin is 0-indexed index into frame_trigger_onsets_idx)
            spike_frame_computed = frames_unique[0] - 1  # convert to 0-indexed
            searchsorted_frame = frame_for_bin[bin_idx]
            if abs(spike_frame_computed - searchsorted_frame) > 1:
                mismatches.append((bin_idx, spike_frame_computed, searchsorted_frame))
    
    if mismatches:
        logger.warning(f"Found {len(mismatches)} frame mismatches between spike data and searchsorted:")
        for bin_idx, spike_frame, search_frame in mismatches[:10]:  # show first 10
            logger.warning(f"  Bin {bin_idx}: spike frame={spike_frame}, searchsorted frame={search_frame}")
    else:
        logger.info("Validation successful: spike frames and searchsorted frames are well-aligned.")

    return True

def time_segment_vector(keep_bins):
    gap_idx = np.where(np.diff(keep_bins) > 1)[0]

    # indices into concatenated output arrays (spikes/time/frame_for_bin)
    segment_starts = np.r_[0, gap_idx + 1]
    segment_ends = np.r_[gap_idx + 1, len(keep_bins)]   # exclusive

    # optional: period label per bin in concatenated timeline
    segments_mask = np.zeros(len(keep_bins), dtype=np.int32)
    for k, (s, e) in enumerate(zip(segment_starts, segment_ends), start=1):
        segments_mask[s:e] = k

    return segments_mask    

# -------------- DEPRECATED/OLD CODE BELOW, KEEP FOR REFERENCE BUT NOT USED ANYMORE --------------

# def prepare_state_space_decoder_data(spike_df, clusters, time_mask, session, bin_width, validate_alignment=False):
#     """Prepare data for state space decoder analysis.

#     Args:
#         spike_df (pd.DataFrame): DataFrame containing spike data with columns 'spike_aligned_to_frame', 'spike_clusters', and 'aligned_spike_times'.
#         session (object): Session object containing camera trigger information.
#         clusters (np.ndarray): Array of unique cluster IDs. that we want to include
#         time_mask (np.ndarray): Boolean array indicating valid time points.
#         bin_width (float, optional): Width of time bins in seconds. Defaults to 0.1.

#     Returns:
#         spikes: np.ndarray: binary spike matrix of shape (n_neurons, n_time_bins)
#         time: np.ndarray: time vector corresponding to the time bins
#         time_bin_to_frame: dict: mapping from time bin indices to frame indices
#     """
#     logger.warning("This function has not been fully debugged! Check the alignment between time bins and frames.")
#     # 1. create a spike matrix (neurons x time bins) at the new bin_width resolution
#     spike_frames = spike_df["spike_aligned_to_frame"].to_numpy().astype(int) - 1 # vector of frame every spike was recorded on, 0-indexed
#     mask = time_mask[spike_frames] # boolean mask to filter spikes that are in the time period of interest
#     spike_df_filt = spike_df[mask].copy()

#     # filter only clusters of interest
#     all_clusters = np.sort(spike_df["spike_clusters"].unique())
#     selected_clusters = all_clusters[clusters]
#     spike_df_filt = spike_df_filt[spike_df_filt['spike_clusters'].isin(selected_clusters)]
    
#     # Create a new column for the time bin index
#     eps = 1e-9 # this should make bin index computation more robust?!
#     spike_df_filt["time_bin"] = np.floor((spike_df_filt["aligned_spike_times"] + eps) / bin_width).astype(int)
    
#     # Convert the filtered DataFrame to a count matrix of shape (num_clusters, num_time_bins)
#     count_matrix = df_to_count_array(spike_df_filt, selected_clusters, "time_bin", fill_time=10)

#     count_array = count_matrix.to_numpy()
#     count_array = (count_array > 0).astype(int)  # still have more than one spike in some cases...
#     spikes = count_array

#     # 2. create a time vector in seconds
#     time = np.arange(0, spikes.shape[1] / (1 / bin_width), bin_width)

#     # 3. create a mapping from time bin to frame index for realigning to behavior

#     # Calculate the start time (in samples) for each time_bin
#     time_bin_starts_sec = np.array(count_matrix.columns) * bin_width
#     time_bin_starts_samples = (time_bin_starts_sec * sampling_rate).astype(int)

#     # For each bin, find the corresponding frame (the last frame whose trigger is <= bin start)
#     frame_for_bin = np.searchsorted(session.camera_trigger.frame_trigger_onsets_idx, # in samples of NPX recording, the time each frame started!
#                                     time_bin_starts_samples, side='right')

#     # Validate frame_for_bin against spike data
#     if validate_alignment:
#         validate_frame_to_bin_alignment(spike_df_filt, frame_for_bin)
        
#     return spikes, time, frame_for_bin

# def df_to_count_array(df, all_clusters, columns, fill_time=[]):
#     """Convert a DataFrame to a count array for each cluster and time bin.
#     Args:
#         df (pd.DataFrame): DataFrame containing spike data.
#         all_clusters (np.ndarray): Array of all unique cluster IDs.
#         columns: the name of the column to use for the time bins.
#         fill_time: can be empty and then we don't fill missing bins,
#                     it can be a list of the start time and end time that we want to fill between,
#                     or it can be an int of the max time bin difference that we want to fill to."""

#     count_matrix = pd.pivot_table(df, index="spike_clusters", columns=columns, values="aligned_spike_times", aggfunc="count", fill_value=0)  # rows: clusters  # columns: time bins
#     # Get all unique cluster IDs from the full spike_df
#     count_matrix = count_matrix.reindex(all_clusters, fill_value=0)

#     if isinstance(fill_time, list):
#         if len(fill_time) == 0:
#             return count_matrix
#         elif len(fill_time) == 2:
#             all_time_bins = np.arange(fill_time[0], fill_time[1] + 1)
#             # Reindex columns to include all bins, filling missing ones with zeros
#             count_matrix = count_matrix.reindex(columns=all_time_bins, fill_value=0)
#     elif isinstance(fill_time, int):
#         # Find gaps in time_bin columns
#         col_array = np.array(count_matrix.columns)
#         diffs = np.diff(col_array)
#         gap_mask = (diffs > 1) & (diffs < fill_time)
#         missing_bins = []
#         for i, is_gap in enumerate(gap_mask):
#             if is_gap:
#                 # Add all missing bins in this gap
#                 missing = np.arange(col_array[i] + 1, col_array[i + 1])
#                 missing_bins.extend(missing)

#         # Add missing columns with zeros
#         if missing_bins:
#             # Create a DataFrame of zeros with the same index as count_matrix and columns as missing_bins
#             zeros_df = pd.DataFrame(0, index=count_matrix.index, columns=missing_bins)
#             # Concatenate along columns
#             count_matrix = pd.concat([count_matrix, zeros_df], axis=1)
#             # Re-sort columns to maintain order
#             count_matrix = count_matrix.reindex(sorted(count_matrix.columns), axis=1)
#             # Optionally, defragment the DataFrame
#             count_matrix = count_matrix.copy()

#         # Re-sort columns to maintain order
#         count_matrix = count_matrix.reindex(sorted(count_matrix.columns), axis=1)

#     return count_matrix