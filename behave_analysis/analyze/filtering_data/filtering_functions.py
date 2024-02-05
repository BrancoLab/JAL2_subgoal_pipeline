"""""A module to host filtering functions for polars dataframes"""

# import third party libaries
import numpy as np
import polars as pl

from behave_analysis.utils.data_loading import load_or_extract_homings


def filter_video_dataframe(dataframe, condition, outofshelter=True, exclude_escape=True):
    """
    A function that filters the video dataframe (the behavioural data) by object presence (whether the barrier or shelter is present or not)
    """

    filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == outofshelter))

    if exclude_escape:
        filtered_video_df = filtered_video_df.filter((filtered_video_df["EscapePeriod"] == False))

    if condition == "pre_shelter":  # empty arena
        filtered_video_df = filtered_video_df.filter((filtered_video_df["shelter"] == False))
        if "barrier_present" in filtered_video_df.columns:
            filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == False))

    elif condition == "shelter_only":  # only the shelter is present
        filtered_video_df = filtered_video_df.filter((filtered_video_df["shelter"] == True))
        if "barrier_present" in filtered_video_df.columns:
            filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == False))

    elif condition == "shelter_present":  # the whole time the shelter is present, but might include the barrier as well
        filtered_video_df = filtered_video_df.filter((filtered_video_df["shelter"] == True))

    elif condition == "barrier_present":  # the hwole time the barrier is present
        filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == True))

    elif condition == "barrier_pre_flip":  # the barrier is present, before we flip it
        filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == True) & (filtered_video_df["barrier_flipped"] == False))

    elif condition == "barrier_post_flip":  # the barrier is present, after we flip it
        filtered_video_df = filtered_video_df.filter((filtered_video_df["barrier_present"] == True) & (filtered_video_df["barrier_flipped"] == True))

    return filtered_video_df


def filter_video_df_mouse_behaviour(dataframe, condition, session, good_homie):
    """
    A function that filters the video dataframe (the behavioural data) based on mousie's homing behaviour
    """
    # get homings
    homings = load_or_extract_homings(session)
    # single out the homings in this condition
    homies_in_condition = (homings.onset_frames > dataframe["frames"][0]) * (homings.offset_frames < dataframe["frames"][-1])
    homies_in_condition = [item for sublist in homies_in_condition for item in sublist]
    # extract the avg angle towards all targets for homings in this condition
    homie_angles = np.zeros((len(homings.homing_angles_dic.keys()),np.sum(homies_in_condition)))
    for i,angle in enumerate(homings.homing_angles_dic.keys()):
        homie_angles[i,:] = homings.homing_angles_dic[angle][homies_in_condition]
    # identify the target: object with smallest head angle
    target_of_homing = np.argmin(np.abs(homie_angles),axis=0)
    # which is the correct target for this condition
    angle_keys = [key for key in homings.homing_angles_dic.keys()]
    if np.logical_or(condition == 'shelter_only', condition == 'shelter_present'):
        target_of_homing = target_of_homing == angle_keys.index('avg_hsa')
    if condition == 'barrier_pre_flip': 
        target_of_homing = target_of_homing == angle_keys.index('avg_hdir_bar_goal1')
    if condition == 'barrier_post_flip': 
        target_of_homing = target_of_homing == angle_keys.index('avg_hdir_bar_goal2')
    
    # turn it into a vector
    # honestly this is pretty ugly, there must be a more elegant pythonic way around this
    frames = dataframe["frames"].to_numpy()
    correct_targeting = np.zeros(len(dataframe))
    onset_frames = homings.onset_frames[homies_in_condition]
    for c in np.arange(1,len(target_of_homing)): # not looking befoe first homing - uncertain times
        if np.logical_and(target_of_homing[c] == good_homie, target_of_homing[c-1] == good_homie):
            start_idx = np.where(frames == int(onset_frames[c-1]))[0]
            stop_idx = np.where(frames == int(onset_frames[c]))[0]
            correct_targeting[int(start_idx):int(stop_idx)] = 1

    # add correct targeting to dataframe
    dataframe = dataframe.hstack([pl.Series("correct_targeting", correct_targeting)])

    filtered_video_df = dataframe.filter((dataframe["EscapePeriod"] == False) & (dataframe['correct_targeting'] == True))

    # import matplotlib.pyplot as plt
    # plt.plot([dataframe["frames"][0],dataframe["frames"][-1]],[0, 0],'k',marker = '--')
    # plt.scatter(homings.onset_frames[homies_in_condition],homings.homing_angles_dic['avg_hsa'][homies_in_condition])
    # plt.scatter(homings.onset_frames[homies_in_condition],homings.homing_angles_dic['avg_hdir_bar_goal1'][homies_in_condition],c='r')
    # plt.scatter(homings.onset_frames[homies_in_condition],homings.homing_angles_dic['avg_hdir_bar_goal2'][homies_in_condition],c='g')
    return filtered_video_df

def filter_video_df_homing_number(dataframe, condition, session, good_homie):
    """
    A function that filters the video dataframe (the behavioural data) based on mousie's homing behaviour
    """
    # get homings
    homings = load_or_extract_homings(session)
    # single out the homings in this condition
    homies_in_condition = (homings.onset_frames > dataframe["frames"][0]) * (homings.offset_frames < dataframe["frames"][-1])
    homies_in_condition = [item for sublist in homies_in_condition for item in sublist]
    # extract the avg angle towards all targets for homings in this condition
    homie_angles = np.zeros((len(homings.homing_angles_dic.keys()),np.sum(homies_in_condition)))
    for i,angle in enumerate(homings.homing_angles_dic.keys()):
        homie_angles[i,:] = homings.homing_angles_dic[angle][homies_in_condition]
    # identify the target: object with smallest head angle
    target_of_homing = np.argmin(np.abs(homie_angles),axis=0)
    # which is the correct target for this condition
    # target_of_homing = boolean with 1s for homings that are targeting the correct (sub)goal for this condition
    angle_keys = [key for key in homings.homing_angles_dic.keys()]
    if np.logical_or(condition == 'shelter_only', condition == 'shelter_present'):
        target_of_homing = target_of_homing == angle_keys.index('avg_hsa')
    if condition == 'barrier_pre_flip': 
        target_of_homing = target_of_homing == angle_keys.index('avg_hdir_bar_goal1')
    if condition == 'barrier_post_flip': 
        target_of_homing = target_of_homing == angle_keys.index('avg_hdir_bar_goal2')
    
    good_homies = np.where(target_of_homing)[0]
    number_of_homings = 2
    good_behavior_threshold = homings.onset_frames[good_homies[number_of_homings]] # the frame of the nth good homing
    
    if good_homie:
        filtered_video_df = dataframe.filter((dataframe["EscapePeriod"] == False) & (dataframe['frames'] > good_behavior_threshold))
    else:
        filtered_video_df = dataframe.filter((dataframe["EscapePeriod"] == False) & (dataframe['frames'] < good_behavior_threshold))

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
    # this sets conditions based on when objects were introduced to arena
    if settings.user_defined_conditions:
        conditions = settings.conditions
    else:
        conditions = identify_conditions(session)
        # this identify conditions based on mousie's behaviour
        if settings.condition_types in ['behavioral_conditions', 'homing_number_2']:
            conditions = identify_conditions_based_on_behave(session)
    return conditions


def identify_conditions_based_on_behave(session):
    """This function subselects which conditions to look at homing behaviour in"""
    condition = []

    if len(session.shelter_time) > 0:
        if session.shelter_time[0] > 0:
            condition.append("pre_shelter")
        if len(session.barrier_time) > 0:
            condition.append("shelter_only")  # if there was a barrier put in at some point
        else:
            condition.append("shelter_present")  # if there was no barrier

    if len(session.barrier_time) > 0:
        if session.barrier_flip_time:
            condition.append("barrier_pre_flip")
            condition.append("barrier_post_flip")
        else:  # there was no flip, so we only have a barrier present time
            condition.append("barrier_present")

    return condition


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
    bin_angle_center = np.sort(np.append([-np.pi, np.pi], [bin_angles[:-1] + (np.mean(np.diff(bin_angles)) / 2)]))
    return bin_angles, bin_angle_center


def generate_bin_positions(min, max, number_of_bins):
    """Bin the mouse's position in xy"""
    bin_pos = np.linspace(min, max, number_of_bins)
    bin_pos_center = bin_pos[:-1] + (np.mean(np.diff(bin_pos)) / 2)
    return bin_pos, bin_pos_center


def generate_bin_positions_ego(min, max, number_of_bins):
    """Bin the mouse's position in xy"""
    bin_pos = np.linspace(min, max, number_of_bins)
    bin_pos_center = np.sort(np.append([min, max], [bin_pos[:-1] + (np.mean(np.diff(bin_pos)) / 2)]))
    return bin_pos, bin_pos_center
