'''A function to calculate spatial efficiency as in Shamash et al. 2021'''

# OS Lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity

from behave_analysis.utils.color_funcs import get_color_based_on_speed
from behave_analysis.analyze.behaviour.utils import base_plotting, identify_condition_of_trial

def spatial_efficiency(onset_frames, stimulus_durations, session, settings, video_df, tracking_data, trial_type, plotting = True, interp = 100, save_dir = []):
    """ 
    Plot escape trajectories as well as optimal path
    """
    ntrials = len(onset_frames)
    nrows = 4
    ncols = 5
        # plt.subplots_adjust(hspace=0.3)

    # Determine number of figures
    if ntrials > 20:
        number_of_figures = int(ntrials // 20 + (ntrials % 20 > 0))
    else:
        number_of_figures = 1

    condition = []
    spatial_efficiency_value = np.empty(len(onset_frames))
    # Plot up to 20 trials per figure
    trial_counter = 0
    for figure in range(number_of_figures):
        if plotting:
            fig = plt.figure(figsize=(20, 20))
            gs = gridspec.GridSpec(nrows, ncols, wspace=0, hspace=0)
        for row in range(nrows):
            for col in range(ncols):
                if trial_counter == ntrials:
                    break
                
                onset_frame = onset_frames[trial_counter]
                stimulus_duration = stimulus_durations[trial_counter]

                condition.append([identify_condition_of_trial(video_df.filter(video_df["frames"] == int(onset_frame)), session)])
                # set up axes with shelt and barrier locations
                if plotting:
                    ax = fig.add_subplot(gs[row, col])
                    base_plotting(ax,tracking_data,condition[trial_counter][0])
                else:
                    ax = []
                real_x, real_y = plot_escape_trajectories(int(onset_frame),int(stimulus_duration*session.video.fps), tracking_data, settings, interp, ax)
                optimal_x, optimal_y = plot_optimal_trajectories(int(onset_frame), tracking_data, condition[trial_counter][0], interp, ax)

                # cosine similarity
                cs = []
                for x,y,ox,oy in zip(real_x,real_y,optimal_x,optimal_y):
                    cs = np.append(cs,cosine_similarity(np.array([x - 512,y]).reshape(1, -1),np.array([ox - 512,oy]).reshape(1, -1)))
                spatial_efficiency_value[trial_counter] = np.mean(cs)
                if plotting:
                    ax.set_title(f'spatial efficiency = {spatial_efficiency_value[trial_counter]:.3f}')
                trial_counter += 1
    
        if plotting:
            # save figure
            filename = str(save_dir) + "/" + trial_type + f"_SpatialEfficiency_{figure}.png"
            plt.savefig(filename)
            if settings.show_plots: plt.show()
            plt.close()

    return condition, spatial_efficiency_value

def plot_escape_trajectories(onset_frames,stimulus_durations, tracking_data, settings,interp = 100, ax = []):
    """ 
    Plot escape trajectories.
    homings/escapes are cropped to when mouse enters the shelter
    for spatial efficiency calculation, the trajectories are interpolated to a uniform lengh given by interp
    """
    # compute and plot each trajectory
    x_loc = tracking_data['head_loc'][onset_frames:onset_frames + stimulus_durations,0]
    y_loc = tracking_data['head_loc'][onset_frames:onset_frames + stimulus_durations,1]
    speed = tracking_data["avg_Velocity"][onset_frames:onset_frames + stimulus_durations]
    # crop the points after the mouse has entered the shelter
    in_shelt = np.where(y_loc > tracking_data['shelter_loc'][0][1])[0]
    if len(in_shelt)>0:
        x_loc = x_loc[:in_shelt[0]]
        y_loc = y_loc[:in_shelt[0]]
        speed = speed[:in_shelt[0]]
    trail_color = np.empty((len(speed),3))
    for i,stim_status in enumerate(np.arange(len(speed))):
        if ax:
            trail_color[i,:] = get_color_based_on_speed(speed=speed[i], 
                                                        object_to_color="trail", 
                                                        stim_status=stim_status, 
                                                        stim_type=settings.stim_type)
    if ax:
        ax.scatter(x_loc,y_loc,s=5,c=trail_color/255)

    # interpolate to standard size
    x_loc = np.interp(np.arange(0,len(x_loc),len(x_loc)/interp),np.arange(len(x_loc)),x_loc)
    y_loc = np.interp(np.arange(0,len(y_loc),len(y_loc)/interp),np.arange(len(y_loc)),y_loc)  

    return x_loc, y_loc

def plot_optimal_trajectories(onset_frames, tracking_data, condition, interp = 100, ax = []):
    """ Plot optimal escape path"""
    opt_x = tracking_data['head_loc'][onset_frames,0]
    opt_y = tracking_data['head_loc'][onset_frames,1]
    opt_t = [0]
    # compute and plot each optimal trajectory to barrier
    if not(any([condition == 'shelter_only',condition == 'pre_shelter', condition == 'barrier_removed',opt_y > 512])): # if no barrier or mouse starts in shelter zone
        opt_t = np.append(opt_t,(interp-1)/2)
        if condition == 'barrier_present': # double sided barrier 
            nearest_barrier_edge = np.argmin([np.sqrt((opt_x - tracking_data["barrier_loc"][0][0])**2 +
                                                    (opt_y - tracking_data["barrier_loc"][0][1])**2),
                                            np.sqrt((opt_x - tracking_data["barrier_loc"][1][0])**2 +
                                                    (opt_y - tracking_data["barrier_loc"][1][1])**2)])
        elif condition == 'barrier_pre_flip':
            nearest_barrier_edge = 0
        elif condition == 'barrier_post_flip':
            nearest_barrier_edge = 1
        
        opt_x = np.append(opt_x,tracking_data['barrier_loc'][nearest_barrier_edge][0])
        opt_y = np.append(opt_y,tracking_data['barrier_loc'][nearest_barrier_edge][1])
    
    opt_x = np.append(opt_x,np.mean([tracking_data['shelter_loc'][0][0],tracking_data['shelter_loc'][1][0]]))
    opt_y = np.append(opt_y,tracking_data['shelter_loc'][0][1])
    opt_t = np.append(opt_t,interp-1)

    # interpolate optimal
    t_int = np.arange(interp)
    opt_xn = np.interp(t_int,opt_t,opt_x)
    opt_yn = np.interp(t_int,opt_t,opt_y)

    # compute and plot each optimal trajectory to shelter
    if ax:
        ax.scatter(opt_xn,opt_yn,s=5,c='r')

    return opt_xn, opt_yn