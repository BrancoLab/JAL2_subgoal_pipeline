'''A function to calculate spatial efficiency as in Shamash et al. 2021'''

# OS Lib
import numpy as np
import matplotlib.pyplot as plt

from behave_analysis.utils.color_funcs import get_color_based_on_speed
from behave_analysis.visualize.visualize_utils import open_tracking_data

def spatial_efficiency(session, settings, video_df, tracking_data, save_dir):
    """ 
    Plot escape trajectories as well as optimal path
    """
    # set up figure and number of rows and calculate number of columns
    plt.figure(figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3)
    ntrial = len(session.__dict__[settings.stim_type].onset_frames)
    nrows = 3
    ncols = ntrial // nrows + (ntrial % nrows > 0)
    
    # plot trajectory
    trajectory_length = np.empty(len(session.__dict__[settings.stim_type].onset_frames))
    optimal_trajectory_length = np.empty(len(session.__dict__[settings.stim_type].onset_frames))
    spatial_efficiency_value = np.empty(len(session.__dict__[settings.stim_type].onset_frames))
    for trial_num, (onset_frames, stimulus_durations) in enumerate(
        zip(session.__dict__[settings.stim_type].onset_frames, 
            session.__dict__[settings.stim_type].stimulus_durations)):
        ax = plt.subplot(nrows, ncols, trial_num + 1)
        # set up axes with shelt and barrier locations
        condition = identify_condition_escape(video_df.filter(video_df['frames'] == onset_frames),session)
        base_plotting(ax,tracking_data,condition)
        trajectory_length[trial_num] = plot_escape_trajectories(onset_frames[0],stimulus_durations[0]*session.video.fps, tracking_data, ax, settings.stim_type)
        optimal_trajectory_length[trial_num] = plot_optimal_trajectories(onset_frames[0], tracking_data, condition, ax)
        spatial_efficiency_value[trial_num] = optimal_trajectory_length[trial_num]/trajectory_length[trial_num]
        ax.set_title('spatial efficiency = ' + str(spatial_efficiency_value[trial_num]))
    
    # save figure
    filename = str(save_dir) + "/" + "SpatialEfficiency" + ".png"
    plt.savefig(filename)
    if settings.show_plots: plt.show()
    plt.close()

def plot_escape_trajectories(onset_frames,stimulus_durations, tracking_data, ax = [], stim_type = []):
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
        if len(stim_type) > 0:
            trail_color[i,:] = get_color_based_on_speed(speed=speed[i], 
                                                        object_to_color="trail", 
                                                        stim_status=stim_status, 
                                                        stim_type=stim_type)
        if i > 0:
            distance_travelled = np.append(distance_travelled,
                                            np.sqrt((x_loc[i] - x_loc[i-1])**2 + (y_loc[i] - y_loc[i-1])**2))
    if len(stim_type) > 0:
        ax.scatter(x_loc,y_loc,s=5,c=trail_color/255)
    return np.sum(distance_travelled)

def plot_optimal_trajectories(onset_frames, tracking_data, condition, ax = []):
    """ Plot optimal escape path"""
    x_loc = tracking_data['head_loc'][onset_frames,0]
    y_loc = tracking_data['head_loc'][onset_frames,1]
    c_line = [1,0,0]
    # compute and plot each optimal trajectory to barrier
    trjectory_to_barrier = 0
    if not(condition == 'shelter_only'):
        if condition == 'barrier_present': # double sided barrier 
            nearest_barrier_edge = np.argmin([np.sqrt((x_loc - tracking_data["barrier_loc"][0][0])**2 +
                                                    (y_loc - tracking_data["barrier_loc"][0][1])**2),
                                            np.sqrt((x_loc - tracking_data["barrier_loc"][1][0])**2 +
                                                    (y_loc - tracking_data["barrier_loc"][1][1])**2)])
        elif condition == 'barrier_pre_flip':
            nearest_barrier_edge = 0
        elif condition == 'barrier_post_flip':
            nearest_barrier_edge = 1
        if not len(ax) == 0:
            ax.plot([x_loc,tracking_data["barrier_loc"][nearest_barrier_edge][0]],
                        [y_loc,tracking_data["barrier_loc"][nearest_barrier_edge][1]],
                        color = c_line)
        trjectory_to_barrier = np.sqrt((x_loc - tracking_data["barrier_loc"][nearest_barrier_edge][0])**2
                                + (y_loc - tracking_data["barrier_loc"][nearest_barrier_edge][1])**2)
        x_loc = tracking_data["barrier_loc"][nearest_barrier_edge][0]
        y_loc = tracking_data["barrier_loc"][nearest_barrier_edge][1]
        
    # compute and plot each optimal trajectory to shelter
    if not len(ax) == 0:
        ax.plot([x_loc,np.mean([tracking_data['shelter_loc'][0][0],tracking_data['shelter_loc'][1][0]])],
                [y_loc,np.mean([tracking_data['shelter_loc'][0][1],tracking_data['shelter_loc'][1][1]])],
                color = c_line)
    trajectory_to_shelter = np.sqrt((np.mean([tracking_data['shelter_loc'][0][0],tracking_data['shelter_loc'][1][0]]) - x_loc)**2 
                                    + (np.mean([tracking_data['shelter_loc'][0][1],tracking_data['shelter_loc'][1][1]]) - y_loc)**2)
    return np.sum([trjectory_to_barrier,trajectory_to_shelter])

###--------------UTILS

def identify_condition_escape(video_df,session):
    """Which condition did the escape happen in?"""
    if np.logical_and(video_df['shelter'].to_numpy() == True, video_df['barrier_present'].to_numpy() == False):
        condition = 'shelter_only'
    elif np.logical_and(video_df['shelter'].to_numpy() == True, video_df['barrier_present'].to_numpy() == True):
        if session.barrier_flip_time:
            if video_df['barrier_flipped'].to_numpy() == False:
                condition = 'barrier_pre_flip'
            else: condition = 'barrier_post_flip'
        else: condition = 'barrier_present'
    return condition

def base_plotting(ax,tracking,condition, session = []):
    if len(tracking) == 0:
        tracking = open_tracking_data(session)
    arena_radius = 460
    # draw shelter
    if 'shelter_loc' in tracking.keys():
        for i in [0,1]:
            plt.plot([tracking["shelter_loc"][0][0],tracking["shelter_loc"][1][0]],
                    [tracking["shelter_loc"][i][1],tracking["shelter_loc"][i][1]],
                    color = [1,0,0])
            plt.plot([tracking["shelter_loc"][i][0],tracking["shelter_loc"][i][0]],
                    [tracking["shelter_loc"][0][1],tracking["shelter_loc"][1][1]],
                    color = [0,0,0])
    
    if not np.logical_or(condition == 'shelter_only', condition == 'pre_shelter'):
        if len(tracking['barrier_loc']) > 0:
            if condition == 'barrier_present':
                # draw old two-sided barrier
                bar_loc = [tracking["barrier_loc"][0][0],tracking["barrier_loc"][1][0]]
            
            if condition == 'barrier_pre_flip':
                # draw barrier from first point to the edge
                if tracking["barrier_loc"][0][0] < 512: bar_loc = [tracking["barrier_loc"][0][0],512+arena_radius]
                else: bar_loc = [512-arena_radius,tracking["barrier_loc"][0][0]]
            
            if condition == 'barrier_post_flip':
                # draw barrier from second point to the edge
                if tracking["barrier_loc"][1][0] < 512: bar_loc = [tracking["barrier_loc"][1][0],512+arena_radius]
                else: bar_loc = [512-arena_radius,tracking["barrier_loc"][1][0]]
            
            plt.plot([bar_loc[0],bar_loc[1]],
                    [tracking["barrier_loc"][0][1],tracking["barrier_loc"][1][1]],
                    color = [0,0,0])
    
    # draw arena edge
    a = 512 + ( arena_radius * np.cos( np.linspace( 0 , 2 * np.pi , 150 ) ) )
    b = 512 + ( arena_radius * np.sin( np.linspace( 0 , 2 * np.pi , 150 ) ) )

    ax.plot( a, b , color = [0,0,0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')