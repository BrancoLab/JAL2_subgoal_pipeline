# these are filtering functions for polars dataframes
import polars as pl
import numpy as np

def filter_video_dataframe(dataframe, condition):
    """
    A function that filters the video dataframe (the behavioural data) by angle of interest and object presence (whether the barrier or shelter is present or not)
    """
    filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == True) & 
                                        (dataframe["EscapePeriod"] == False))

    if condition == 'pre_shelter': # empty arena
        filtered_video_df = filtered_video_df.filter((filtered_video_df["shelter_only"] == False))
        if 'barrier_present' in filtered_video_df.columns: 
            filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == False))

    elif condition == 'shelter_only': # only the shelter is present
        filtered_video_df = filtered_video_df.filter((filtered_video_df["shelter_only"] == True))
        if 'barrier_present' in filtered_video_df.columns: 
            filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == False))

    elif condition == 'shelter_present': # the whole time the shelter is present, but might include the barrier as well
        filtered_video_df = filtered_video_df.filter((filtered_video_df["shelter_only"] == True))

    elif condition == 'barrier_present': # the hwole time the barrier is present
        filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == True))

    elif condition == 'barrier_pre_flip': # the barrier is present, before we flip it
        filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == True))

    elif condition == 'barrier_post_flip': # the barrier is present, after we flip it
        filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == True))

    return filtered_video_df

def identify_conditions(session):
    """
    A function that looks at shelter_time, barrier_time and barrier_flip_time and determines what conditions are available in this session
    """
    condition = ['all_time']

    if len(session.shelter_time)>0:
        condition.append('shelter_present')
        if session.shelter_time[0] > 0:
            condition.append('pre_shelter')
        if len(session.barrier_time) > 0:
            condition.append('shelter_only')
    
    if len(session.barrier_time) > 0:
        condition.append('barrier_present')
        if session.barrier_flip_time:
            condition.append('barrier_pre_flip')
            condition.append('barrier_post_flip')

    return condition

def identify_angles(session):
    """
    A function that looks at shelter_time and barrier_time and determines what angles are interesting in this session
    """
    angles = ['hdir']

    if len(session.shelter_time)>0:
        angles.append('hsa')
    
    if len(session.barrier_time) > 0:
        angles.append('h_bar_north_a')
        angles.append('h_bar_south_a')
        angles.append('h_bar_centre_a')

    return angles

def generate_bin_angles(number_of_bins): 
    bin_angles = np.linspace(-np.pi, np.pi, number_of_bins)
    bin_angle_center = np.sort(np.append([-np.pi,np.pi], [bin_angles[:-1] + (np.mean(np.diff(bin_angles))/2)]))
    return bin_angles, bin_angle_center