import pandas as pd
import numpy as np
from loguru import logger

def assign_positional_bins_to_frames(video_df: pd.DataFrame, bins: np.array) -> pd.DataFrame:
    """Assign x and y bins to each frame in the video DataFrame

    Args:
        video_df (pd.DataFrame): A DataFrame containing the video data
        bins (np.array): The bin edges to split the x and y positions into

    Returns:
        pd.DataFrame: The video DataFrame with x and y bins assigned to each frame"""

    x_bins = pd.cut(video_df["mouse_x_position"], bins=bins, labels=False)
    y_bins = pd.cut(video_df["mouse_y_position"], bins=bins, labels=False)
    video_df["x_bins"] = x_bins
    video_df["y_bins"] = y_bins

    return video_df

def create_centered_bins(nbins: int = 0, bin_size: float = 0.0, arena_radius: float = 0.0, center_offset: float = 0.0) -> np.array:
    """Create bin edges for spatial binning that are centered around the center of the arena

    Args:
        nbins (int): The number of bins to create
        bin_size (float): The size of each bin in pixels
        arena_radius (float): The radius of the arena in pixels
        center_offset (float): The offset of the center of the arena in pixels
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