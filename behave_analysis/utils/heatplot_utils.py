import numpy as np
import polars as pl


def remove_points_away_from_center_of_circle(x, y, session_height) -> tuple:
    """Remove all points that are outside of the arena circle and return the filtered x and y coordinates.

    TODO:
        + Make the radius of the arena a variable not hard coded
    """

    dist = np.sqrt(((x - session_height / 2) ** 2) + ((y - session_height / 2) ** 2))
    # Use the euclidean distance formula to find the distance from the center of the arena
    filt_x = x[dist < 460]  # 460 is size of arena circle radius, see register
    filt_y = y[dist < 460]
    return filt_x, filt_y

def filter_outside_arena_tracking_for_video_and_spike_data(video_and_spike_data: pl.DataFrame, session) -> pl.DataFrame:
    """Filter out all tracking data that is outside of the arena circle. This is the remove pooints away from center 
    of circle but for polars DataFrame.
    
    Args:
        video_and_spike_data (pl.DataFrame): A DataFrame containing spike counts per frame and neuron in a single session
        session (Session): A Session object containing metadata about the session
    
    Returns:
        pl.DataFrame: A DataFrame containing spike counts per frame and neuron in a single session with tracking data outside of the arena circle removed"""
    
    # Remove all rows that are outside of the arena circle
    video_and_spike_data = video_and_spike_data.with_columns(
        np.sqrt(((pl.col("mouse_x_position") - session.video.height / 2) ** 2) + ((pl.col("mouse_y_position") - session.video.height / 2) ** 2)).alias("dist")
    )
    
    # Now filter out the mouse x positions where distance is less than 460 and same for y
    video_and_spike_data = video_and_spike_data.filter(pl.col("dist") < 460)
    video_and_spike_data = video_and_spike_data.drop("dist")
    
    return video_and_spike_data


def add_features(ax, condition: str, tracking: dict, xbins: np.array, ybins: np.array) -> None:
    """Draw arena and barrier on top of heatmaps or scatterplots

    Args:
        ybins (np.array): ybins used to digitize the y coordinates e.g [96., 192., 288., 384., 480.]
        xbins (np.array): xbins used to digitize the x coordinates e.g [96., 192., 288., 384., 480.]
        tracking (dict): dictionary containing tracking data
        condition (str): condition of the plot
        ax (matplotlib.axes): axis object to draw on
        
    NOTE:
        If tracking is outside of the arena the shelter and the barrier will not be drawn consistently. Thus
        it is important to filter the tracking data before calling this function.
    """

    arena_radius = 460
    # draw shelter
    if "shelter_loc" in tracking.keys():
        shelt = [np.digitize(tracking["shelter_loc"][0], xbins), np.digitize(tracking["shelter_loc"][1], ybins)]
        for i in [0, 1]:
            ax.plot([shelt[0][0], shelt[1][0]], [shelt[i][1], shelt[i][1]], color="k")
            ax.plot([shelt[i][0], shelt[i][0]], [shelt[0][1], shelt[1][1]], color="k")

    if not np.logical_or(condition == "shelter_only", condition == "pre_shelter"):
        if len(tracking["barrier_loc"]) > 0:
            if np.logical_or(np.logical_or(condition == "barrier_present", condition == "all_time"), condition == "shelter_present"):
                # draw old two-sided barrier
                bar_loc = [tracking["barrier_loc"][0][0], tracking["barrier_loc"][1][0]]

            if condition == "barrier_pre_flip":
                # draw barrier from first point to the edge
                if tracking["barrier_loc"][0][0] < 512:
                    bar_loc = [tracking["barrier_loc"][0][0], 512 + arena_radius]
                else:
                    bar_loc = [512 - arena_radius, tracking["barrier_loc"][0][0]]

            if condition == "barrier_post_flip":
                # draw barrier from second point to the edge
                if tracking["barrier_loc"][1][0] < 512:
                    bar_loc = [tracking["barrier_loc"][1][0], 512 + arena_radius]
                else:
                    bar_loc = [512 - arena_radius, tracking["barrier_loc"][1][0]]

            bar_loc = np.digitize(bar_loc, xbins)
            ax.plot(
                [bar_loc[0], bar_loc[1]],
                [np.digitize(tracking["barrier_loc"][0][1], ybins), np.digitize(tracking["barrier_loc"][1][1], ybins)],
                color="k",
            )
