from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions
from behave_analysis.homings.homings import cum_distance

import os
import numpy as np
import matplotlib.pyplot as plt


def initial_heading_angle(session, onsets, offsets, head_angle, all_conditions, tracking_data, exclude_south = False):
    """Finds the cosine similarity between the heading of the mouse when he starts running in the homing and the angle with the three goals
    Assigns the homing heading to the goal it is most similar to
    Doesn't differentiate between below and above the barrier so a lot of shelter targets are actually below the barrier"""

    good_starting_angle = [np.pi/4,3*(np.pi/4)]

    conditions = identify_conditions(session)

    if np.logical_and(not 'barrier_present' in conditions, 'shelter_present' in conditions):
        conditions = conditions + ['shelter_only']

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")

    heading_by_cond = {}
    heading_by_cond_correct_start = {}

    for i, con in enumerate(conditions):
        pref_heading = np.zeros(3)
        pref_heading_correct_start = np.zeros(3)
        sum_homings = 0
        for idx, (onset,offset) in enumerate(zip(onsets,offsets)):
            if np.isnan(onset):
                continue
            trial_condition = all_conditions[idx]
            if isinstance(onset,np.ndarray):
                onset = onset[0].astype(int)

            if np.logical_or(con == trial_condition, con == "all_time"):

                frame_coords = tracking_data["avg_loc"][onset:offset]
                # _, start_frame = cum_distance(onset, offset, frame_coords, session.video.pixels_per_cm, 15)
                _, start_frame = cum_distance(onset, offset, frame_coords, 10, 15)

                # calculate the preference of mouse heading for one of three targets using cosine similarity
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][0][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][0][1]
                bprea = - np.arctan2(ydist, xdist)
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][1][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][1][1]
                bposta = - np.arctan2(ydist, xdist)
                if tracking_data["bod_shelt_dir"][start_frame] < 0: bsa = tracking_data["bod_shelt_dir"][start_frame] + np.pi
                if tracking_data["bod_shelt_dir"][start_frame] > 0:  bsa = tracking_data["bod_shelt_dir"][start_frame] - np.pi

                if np.logical_and(exclude_south,
                                  np.logical_or(con == 'barrier_pre_flip', con == 'barrier_post_flip')):
                    if tracking_data["avg_loc"][onset,1] > 512:
                        continue
                cosim=[]
                for ang in [bprea,bsa, bposta]:
                    cosim = np.append(cosim,np.cos(ang-head_angle[idx]))
                pref_heading[np.argmax(cosim)] += 1
                if np.logical_and(tracking_data["hdir"][onset-1] > good_starting_angle[0], 
                                  tracking_data["hdir"][onset-1] < good_starting_angle[1]):
                    pref_heading_correct_start[np.argmax(cosim)] += 1
                sum_homings += 1

        heading_by_cond[con] = pref_heading
        heading_by_cond_correct_start[con] = pref_heading_correct_start

    return heading_by_cond, heading_by_cond_correct_start