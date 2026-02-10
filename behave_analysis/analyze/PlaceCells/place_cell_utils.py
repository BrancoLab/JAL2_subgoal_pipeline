import pandas as pd
import numpy as np

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
