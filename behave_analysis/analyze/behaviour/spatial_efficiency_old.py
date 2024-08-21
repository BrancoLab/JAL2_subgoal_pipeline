'''A function to calculate spatial efficiency as in Shamash et al. 2021'''

# OS Lib
import numpy as np
import matplotlib.pyplot as plt

from behave_analysis.utils.color_funcs import get_color_based_on_speed
from behave_analysis.analyze.behaviour.utils import base_plotting, identify_condition_of_trial

def spatial_efficiency(onset_frames, stimulus_durations, session, settings, video_df, tracking_data, plotting = True, save_dir = []):
    """ 
    Plot escape trajectories as well as optimal path
    """
    if plotting:
        # set up figure and number of rows and calculate number of columns
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        ntrial = len(onset_frames)
        nrows = 3
        ncols = ntrial // nrows + (ntrial % nrows > 0)
    
    # plot trajectory
    condition = []
    trajectory_length = np.empty(len(onset_frames))
    optimal_trajectory_length = np.empty(len(onset_frames))
    spatial_efficiency_value = np.empty(len(onset_frames))
    for trial_num, (onset_frame, stimulus_duration) in enumerate(
        zip(onset_frames, stimulus_durations)):
        condition.append([identify_condition_of_trial(video_df.filter(video_df["frames"] == onset_frame[0]), session)])
        # set up axes with shelt and barrier locations
        if plotting:
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            base_plotting(ax,tracking_data,condition[trial_num][0])
        else:
            ax = []
        trajectory_length[trial_num] = plot_escape_trajectories(onset_frame[0],stimulus_duration[0]*session.video.fps, tracking_data, settings, ax)
        optimal_trajectory_length[trial_num] = plot_optimal_trajectories(onset_frame[0], tracking_data, condition[trial_num][0], ax)
        spatial_efficiency_value[trial_num] = optimal_trajectory_length[trial_num]/trajectory_length[trial_num]
        if plotting:
            ax.set_title(f'spatial efficiency = {spatial_efficiency_value[trial_num]:.2f}')
    
    if plotting:
        # save figure
        filename = str(save_dir) + "/" + "SpatialEfficiency" + ".png"
        plt.savefig(filename)
        if settings.show_plots: plt.show()
        plt.close()

    return condition, trajectory_length, optimal_trajectory_length, spatial_efficiency_value

def plot_escape_trajectories(onset_frames,stimulus_durations, tracking_data, settings, ax = []):
    """ 
    Plot escape trajectories
    """
    # compute and plot each trajectory
    x_loc = tracking_data['head_loc'][onset_frames:onset_frames + int(stimulus_durations),0]
    y_loc = tracking_data['head_loc'][onset_frames:onset_frames + int(stimulus_durations),1]
    speed = tracking_data["avg_Velocity"][onset_frames:onset_frames + int(stimulus_durations)]
    trail_color = np.empty((len(speed),3))
    distance_travelled = []
    for i,stim_status in enumerate(np.arange(0,stimulus_durations)):
        if ax:
            trail_color[i,:] = get_color_based_on_speed(speed=speed[i], 
                                                        object_to_color="trail", 
                                                        stim_status=stim_status, 
                                                        stim_type=settings.stim_type)
        if i > 0:
            distance_travelled = np.append(distance_travelled,
                                            np.sqrt((x_loc[i] - x_loc[i-1])**2 + (y_loc[i] - y_loc[i-1])**2))
    if ax:
        ax.scatter(x_loc,y_loc,s=5,c=trail_color/255)
    return np.sum(distance_travelled)

def plot_optimal_trajectories(onset_frames, tracking_data, condition, ax = []):
    """ Plot optimal escape path"""
    x_loc = tracking_data['head_loc'][onset_frames,0]
    y_loc = tracking_data['head_loc'][onset_frames,1]
    c_line = [1,0,0]
    # compute and plot each optimal trajectory to barrier
    trjectory_to_barrier = 0
    if not(np.logical_or(condition == 'shelter_only', condition == 'barrier_removed')):
        if condition == 'barrier_present': # double sided barrier 
            nearest_barrier_edge = np.argmin([np.sqrt((x_loc - tracking_data["barrier_loc"][0][0])**2 +
                                                    (y_loc - tracking_data["barrier_loc"][0][1])**2),
                                            np.sqrt((x_loc - tracking_data["barrier_loc"][1][0])**2 +
                                                    (y_loc - tracking_data["barrier_loc"][1][1])**2)])
        elif condition == 'barrier_pre_flip':
            nearest_barrier_edge = 0
        elif condition == 'barrier_post_flip':
            nearest_barrier_edge = 1
        if ax:
            ax.plot([x_loc,tracking_data["barrier_loc"][nearest_barrier_edge][0]],
                        [y_loc,tracking_data["barrier_loc"][nearest_barrier_edge][1]],
                        color = c_line)
        trjectory_to_barrier = np.sqrt((x_loc - tracking_data["barrier_loc"][nearest_barrier_edge][0])**2
                                + (y_loc - tracking_data["barrier_loc"][nearest_barrier_edge][1])**2)
        x_loc = tracking_data["barrier_loc"][nearest_barrier_edge][0]
        y_loc = tracking_data["barrier_loc"][nearest_barrier_edge][1]
        
    # compute and plot each optimal trajectory to shelter
    if ax:
        ax.plot([x_loc,np.mean([tracking_data['shelter_loc'][0][0],tracking_data['shelter_loc'][1][0]])],
                [y_loc,np.mean([tracking_data['shelter_loc'][0][1],tracking_data['shelter_loc'][1][1]])],
                color = c_line)
    trajectory_to_shelter = np.sqrt((np.mean([tracking_data['shelter_loc'][0][0],tracking_data['shelter_loc'][1][0]]) - x_loc)**2 
                                    + (np.mean([tracking_data['shelter_loc'][0][1],tracking_data['shelter_loc'][1][1]]) - y_loc)**2)
    return np.sum([trjectory_to_barrier,trajectory_to_shelter])

###--------------UTILS

# def identify_condition_escape(video_df,session):
#     """Which condition did the escape happen in?"""
#     if np.logical_and(video_df['shelter'].to_numpy() == True, video_df['barrier_present'].to_numpy() == False):
#         condition = 'shelter_only'
#     elif np.logical_and(video_df['shelter'].to_numpy() == True, video_df['barrier_present'].to_numpy() == True):
#         if session.barrier_flip_time:
#             if video_df['barrier_flipped'].to_numpy() == False:
#                 condition = 'barrier_pre_flip'
#             else: condition = 'barrier_post_flip'
#         else: condition = 'barrier_present'
#     return condition

# def base_plotting(ax,tracking,condition, session = []):
#     if len(tracking) == 0:
#         tracking = open_tracking_data(session)
#     arena_radius = 460
#     # draw shelter
#     if 'shelter_loc' in tracking.keys():
#         for i in [0,1]:
#             plt.plot([tracking["shelter_loc"][0][0],tracking["shelter_loc"][1][0]],
#                     [tracking["shelter_loc"][i][1],tracking["shelter_loc"][i][1]],
#                     color = [1,0,0])
#             plt.plot([tracking["shelter_loc"][i][0],tracking["shelter_loc"][i][0]],
#                     [tracking["shelter_loc"][0][1],tracking["shelter_loc"][1][1]],
#                     color = [0,0,0])
    
#     if not np.logical_or(condition == 'shelter_only', condition == 'pre_shelter'):
#         if len(tracking['barrier_loc']) > 0:
#             if np.logical_or(np.logical_or(condition == 'barrier_present',condition == 'all_time'),condition == 'shelter_present'):
#                 # draw old two-sided barrier
#                 bar_loc = [tracking["barrier_loc"][0][0],tracking["barrier_loc"][1][0]]
            
#             if condition == 'barrier_pre_flip':
#                 # draw barrier from first point to the edge
#                 if tracking["barrier_loc"][0][0] < 512: bar_loc = [tracking["barrier_loc"][0][0],512+arena_radius]
#                 else: bar_loc = [512-arena_radius,tracking["barrier_loc"][0][0]]
            
#             if condition == 'barrier_post_flip':
#                 # draw barrier from second point to the edge
#                 if tracking["barrier_loc"][1][0] < 512: bar_loc = [tracking["barrier_loc"][1][0],512+arena_radius]
#                 else: bar_loc = [512-arena_radius,tracking["barrier_loc"][1][0]]
            
#             plt.plot([bar_loc[0],bar_loc[1]],
#                     [tracking["barrier_loc"][0][1],tracking["barrier_loc"][1][1]],
#                     color = [0,0,0])

#     # draw arena edge
#     a = 512 + ( arena_radius * np.cos( np.linspace( 0 , 2 * np.pi , 150 ) ) )
#     b = 512 + ( arena_radius * np.sin( np.linspace( 0 , 2 * np.pi , 150 ) ) )

#     ax.plot( a, b , color = [0,0,0])
#     ax.invert_yaxis()
#     ax.set_aspect('equal')
#     ax.axis('off')