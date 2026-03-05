import polars as pl
import numpy as np
from loguru import logger
import pandas as pd
from scipy.ndimage import gaussian_filter

def assign_positional_bins_to_frames(video_df: pl.DataFrame, bins: np.array) -> pl.DataFrame:
    """Assign x and y bins to each frame in the video DataFrame

    Args:
        video_df (pl.DataFrame): A DataFrame containing the video data
        bins (np.array): The bin edges to split the x and y positions into

    Returns:
        pl.DataFrame: The video DataFrame with x and y bins assigned to each frame
            NB: binning is right-edge inclusive! bin_edge[i-1] < x <= bin_edge[i]"""

    x_bins = pd.cut(video_df["mouse_x_position"], bins=bins, labels=False, retbins=False)
    video_df = video_df.with_columns(pl.Series(name="xbins", values=x_bins))

    y_bins = pd.cut(video_df["mouse_y_position"], bins=bins, labels=False, retbins=False)
    video_df = video_df.with_columns(pl.Series(name="ybins", values=y_bins))

    return video_df

def create_centered_bins(nbins: int = 0, bin_size: float = 0.0, arena_radius: float = 460.0, center_offset: float = 512.0) -> np.array:
    """Create bin edges for spatial binning that are centered around the center of the arena

    Args:
        nbins (int): The number of bins to create
        bin_size (float): The size of each bin in pixels
        arena_radius (float): The radius of the arena in pixels
        center_offset (float): The offset of the center of the arena in pixels (usually 512 for a 1024x1024 video)
    Returns:
        np.array: The bin edges for spatial binning
    """
    if (nbins > 0) and (bin_size == 0.0):
        bin_size = (arena_radius*2) / nbins
    elif (bin_size > 0.0) and (nbins == 0):
        nbins = int((arena_radius*2) / bin_size)
    elif (nbins > 0) and (bin_size > 0.0):
        logger.warning("Both nbins and bin_size provided, adjusting area of the arena covered by bins")
        arena_radius = (nbins * bin_size)
    elif (nbins == 0) and (bin_size == 0.0):
        logger.warning("Neither nbins nor bin_size provided, defaulting to 10pixel bins")
        bin_size = 10
    
    # set up bins!
    one_sided_bins = np.arange(0, arena_radius + bin_size, bin_size)
    bins = np.concatenate((-one_sided_bins[::-1], one_sided_bins[1:])) + center_offset
    
    return bins

def smooth_maps(map, column, sigma):
    """This function takes in a map (e.g. spike count map or occupancy map) and smooths it with a Gaussian kernel, ignoring NaNs.
    INPUTS:
        map: a 2D array or dataframe representing the map to be smoothed (e.g. spike count map or occupancy map)
        column: the name of the column in the dataframe that contains the values to be smoothed (e.g. 'spike_count' or 'occupancy_seconds')
        sigma: the standard deviation of the Gaussian kernel in number of bins"""
    if isinstance(map, pl.DataFrame):
        # convert to 2D array with NaNs for empty bins
        map = (map.pivot(index="ybins", columns="xbins", values=column)
                    .to_numpy())
        map = map[:, 1:] # remove the first column which is a list of ybins rather than the values to be smoothed
    # smooth the map with a Gaussian kernel, ignoring NaNs
    # Preserve NaN locations
    nan_mask = np.isnan(map)
    map_filled = np.nan_to_num(map, nan=0.0)

    # Smooth
    smoothed_matrix = gaussian_filter(map_filled, sigma=sigma)

    # Restore NaN
    smoothed_matrix[nan_mask] = np.nan

    return smoothed_matrix

def compute_spatial_information(rate_map, occupancy_map):
    """
    Compute the Skaggs et al. (1993) spatial information score
    for a 2D place field rate map.

    Parameters
    ----------
    rate_map : np.ndarray (2D)
        Mean firing rate in each spatial bin (Hz).
        - NaN for unreliable/invalid bins.
        - 0 for bins where the cell is silent (valid data).
    occupancy_map : np.ndarray (2D)
        Time spent in each spatial bin (in seconds).
        - NaN for unreliable/invalid bins (must match rate_map NaNs).

    Returns
    -------
    spatial_info_bps : float
        Spatial information in bits per spike.
    spatial_info_bpss : float
        Spatial information in bits per second.
    """
    rate_map = np.asarray(rate_map, dtype=float)
    occupancy_map = np.asarray(occupancy_map, dtype=float)

    if rate_map.shape != occupancy_map.shape:
        raise ValueError("rate_map and occupancy_map must have the same shape.")

    valid = ~np.isnan(rate_map) & ~np.isnan(occupancy_map)

    rate = rate_map[valid]
    occupancy = occupancy_map[valid]

    p_i = occupancy / np.sum(occupancy)
    mean_rate = np.sum(p_i * rate)

    if mean_rate == 0:
        return 0.0, 0.0

    nonzero_rate = rate > 0
    info_terms = np.zeros_like(rate)
    ratio = rate[nonzero_rate] / mean_rate
    info_terms[nonzero_rate] = p_i[nonzero_rate] * ratio * np.log2(ratio)

    spatial_info_bps = np.sum(info_terms)
    spatial_info_bpss = spatial_info_bps * mean_rate

    return spatial_info_bps, spatial_info_bpss