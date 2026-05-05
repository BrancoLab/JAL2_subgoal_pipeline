import numpy as np
from scipy.signal import savgol_filter
import polars as pl

from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.analyze.PlaceCells.place_cell_utils import create_centered_bins, assign_positional_bins_to_frames
from behave_analysis.analyze.filtering_data.filtering_functions import generate_bins

def compute_distance_object(df, variable, session):
    if "shelter" in variable:
        shelter_location = [np.mean([session.shelter_location[0][0], session.shelter_location[1][0]]), session.shelter_location[0][1]]
        return np.sqrt((df["mouse_x_position"] - shelter_location[0])**2 + (df["mouse_y_position"] - shelter_location[1])**2)
    if "barrier1" in variable:
        return np.sqrt((df["mouse_x_position"] - session.barrier_location[0][0])**2 + (df["mouse_y_position"] - session.barrier_location[0][1])**2)
    if "barrier2" in variable:
        return np.sqrt((df["mouse_x_position"] - session.barrier_location[1][0])**2 + (df["mouse_y_position"] - session.barrier_location[1][1])**2)

def select_xval_frames(video_df, frames, method, comparison_indices):
    """Method can be 'random split' or 'balanced_bins' (for behaviour)
    RETURNS: indices into fcm for explore_train, explore_test, homing (for a given condition c)"""

    if "random_split" in method:
        explore_indices = frames
        
        if method == "random_split":
            n_items = int(len(explore_indices) // 2)
        elif method == "random_split h_match":
            n_items = len(comparison_indices)

        train_idx = np.random.choice(explore_indices, size=n_items, replace=False)
        xval_idx = np.array([idx for idx in explore_indices if idx not in train_idx])

    elif method == "half":
        explore_indices = frames

        train_idx = explore_indices[: len(explore_indices) // 2]
        xval_idx = explore_indices[len(explore_indices) // 2 :]

    elif "match" in method:
        # how to balance A2 for similar behaviour as B
        bin_list = []
        if "pos" in method:
            pos_bins = create_centered_bins(nbins=16)
            video_df = assign_positional_bins_to_frames(video_df, pos_bins)
            bin_list.extend(["xbins", "ybins"])

        if "hdir" in method:
            bins, _ = generate_bins(13, -np.pi, np.pi)
            binned_angles = np.digitize(video_df["hdir"].to_numpy(), bins)
            video_df = video_df.with_columns(pl.Series(name="hdirbins", values=binned_angles))
            bin_list.extend(["hdirbins"])

        if "speed" in method:
            speed_bins, _ = generate_bins(10, 0, 100)
            binned_speed = np.digitize(video_df["speed"].to_numpy(), speed_bins)
            video_df = video_df.with_columns(pl.Series(name="speedbins", values=binned_speed))
            bin_list.extend(["speedbins"])

        Y = video_df[bin_list].to_numpy()

        valid_rows = np.sum(np.isnan(Y), axis=1) == 0
        cum_bins = np.full(Y.shape[0], -1)
        _, cum_bins[valid_rows] = np.unique(Y[valid_rows, :].astype(int), axis=0, return_inverse=True)
        cum_bins_train = cum_bins[frames]
        cum_bins_comparison = cum_bins[comparison_indices]
        # 1. Get counts for Homing (your template)
        bin_id_comparison, counts_comparison = np.unique(cum_bins_comparison, return_counts=True)

        # 2. Find how many of those same bins exist in Explore
        counts_train_in_comparison_bins = []
        for b_id in bin_id_comparison:
            count = np.sum(cum_bins_train == b_id)
            counts_train_in_comparison_bins.append(count)

        counts_train_in_comparison_bins = np.array(counts_train_in_comparison_bins)

        # 3. Calculate Scaling Factor
        # We look for the ratio (Explore_Available / Homing_Required)
        # Use 1 if Explore count is 0 to avoid a 0.0 multiplier for the whole dataset
        ratios = np.where(counts_train_in_comparison_bins > 0, counts_train_in_comparison_bins / counts_comparison, 1.0 / counts_comparison)

        scaling_factor = max(np.min(ratios), 1)

        # 4. Apply scaling and Subsample
        matched_train_indices = []

        for b_id, h_count, e_available in zip(bin_id_comparison, counts_comparison, counts_train_in_comparison_bins):
            # Target is the Homing count scaled down by the bottleneck factor
            target_n = int(np.floor(h_count * scaling_factor))

            # Final safety check: can't pick more than we have
            n_to_draw = min(e_available, target_n)

            if n_to_draw > 0:
                in_bin_mask = cum_bins_train == b_id
                available_indices = frames[in_bin_mask]

                selected = np.random.choice(available_indices, n_to_draw, replace=False)
                matched_train_indices.extend(selected)

        train_idx = np.sort(matched_train_indices)
        xval_idx = [x for x in frames if x not in matched_train_indices]

    return train_idx, xval_idx


def safe_corrcoef(x, y, min_n=3):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < min_n:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def get_correlation_loadings(data, scores):
    """
    Calculates the correlation between original variables (columns of data)
    and the canonical scores.

    data: (time x features) matrix
    scores: (time x components) matrix from cca.transform
    """
    n_features = data.shape[1]
    n_components = scores.shape[1]
    loadings = np.zeros((n_features, n_components))

    for i in range(n_features):
        for j in range(n_components):
            # Robust to constant vectors and NaNs that can appear in small/filtered splits.
            loadings[i, j] = safe_corrcoef(data[:, i], scores[:, j])

    return loadings
