"""""A module to host filtering functions for polars dataframes"""

# import third party libaries
import numpy as np


def filter_video_dataframe(
    dataframe, condition, outofshelter=True, exclude_escape=True
):
    """
    A function that filters the video dataframe (the behavioural data) by angle of interest and object presence (whether the barrier or shelter is present or not)
    """

    filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == outofshelter))

    if exclude_escape:
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["EscapePeriod"] == False)
        )

    if condition == "pre_shelter":  # empty arena
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["shelter"] == False)
        )
        if "barrier_present" in filtered_video_df.columns:
            filtered_video_df = filtered_video_df.filter(
                (filtered_video_df["barrier_present"] == False)
            )

    elif condition == "shelter_only":  # only the shelter is present
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["shelter"] == True)
        )
        if "barrier_present" in filtered_video_df.columns:
            filtered_video_df = filtered_video_df.filter(
                (filtered_video_df["barrier_present"] == False)
            )

    elif (
        condition == "shelter_present"
    ):  # the whole time the shelter is present, but might include the barrier as well
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["shelter"] == True)
        )

    elif condition == "barrier_present":  # the hwole time the barrier is present
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["barrier_present"] == True)
        )

    elif condition == "barrier_pre_flip":  # the barrier is present, before we flip it
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["barrier_present"] == True)
            & (filtered_video_df["barrier_flipped"] == False)
        )

    elif condition == "barrier_post_flip":  # the barrier is present, after we flip it
        filtered_video_df = filtered_video_df.filter(
            (filtered_video_df["barrier_present"] == True)
            & (filtered_video_df["barrier_flipped"] == True)
        )

    return filtered_video_df


def identify_conditions(session) -> list:
    """Determine which conditions are available in this session

    e.g. shelter_only, barrier_present, barrier_pre_flip, barrier_post_flip"""

    condition = ["all_time"]

    if len(session.shelter_time) > 0:
        condition.append("shelter_present")
        if session.shelter_time[0] > 0:
            condition.append("pre_shelter")
        if len(session.barrier_time) > 0:
            condition.append("shelter_only")

    if len(session.barrier_time) > 0:
        condition.append("barrier_present")
        if session.barrier_flip_time:
            condition.append("barrier_pre_flip")
            condition.append("barrier_post_flip")

    return condition


def extract_all_or_custom_conditions(settings, session):
    """Identify all conditions to analyze or use custom conditions from settings file"""
    if settings.user_defined_conditions:
        conditions = settings.conditions
    else:
        conditions = identify_conditions(session)
    return conditions


def identify_angles(session):
    """
    A function that looks at shelter_time and barrier_time and determines what angles are interesting in this session
    """
    angles = ["hdir"]

    if len(session.shelter_time) > 0:
        angles.append("hsa")

    if len(session.barrier_time) > 0:
        angles.append("h_bar_north_a")
        angles.append("h_bar_south_a")
        angles.append("h_bar_centre_a")

    return angles


def generate_bin_angles(number_of_bins):
    bin_angles = np.linspace(-np.pi, np.pi, number_of_bins)
    bin_angle_center = np.sort(
        np.append(
            [-np.pi, np.pi], [bin_angles[:-1] + (np.mean(np.diff(bin_angles)) / 2)]
        )
    )
    return bin_angles, bin_angle_center


def generate_bin_positions(min, max, number_of_bins):
    """Bin the mouse's position in xy"""
    bin_pos = np.linspace(min, max, number_of_bins)
    bin_pos_center = bin_pos[:-1] + (np.mean(np.diff(bin_pos)) / 2)
    # bin_pos_center = np.sort(
    #     np.append([min, max], [bin_pos[:-1] + (np.mean(np.diff(bin_pos)) / 2)])
    # )
    return bin_pos, bin_pos_center
