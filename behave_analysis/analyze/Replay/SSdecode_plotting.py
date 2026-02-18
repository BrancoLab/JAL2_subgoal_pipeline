"""Functions for plotting and visualizing decoder results"""

import numpy as np
import matplotlib.pyplot as plt

from behave_analysis.utils.arena_plotting import Arena

def heatmap_neural_data(axs, neural_data, time, time_markers = []):
    """INPUTS:
            neural_data: firing rate matrix (zscored?) neurons x time
            time: the time vector corresponding to the neural data (in seconds)
            time_markers: a list of time points to mark with vertical lines (in seconds)"""
    axs.imshow(neural_data, aspect='auto', origin='lower', cmap='gray_r', vmin = -.5, vmax = 2,
            extent=[time[0], time[-1], 0, neural_data.shape[0]], interpolation='none')
    if len(time_markers) > 0:
        for time_marker in time_markers:    
            axs.axvline(x=time_marker, color='r', linestyle='--', linewidth=1)
    axs.set_xlabel('time (s)')
    axs.set_title('Neural activity')

def plot_predicted_vs_actual(axs, posterior_dist, predicted_position, actual_position, time, rmse = '', var_name = '', time_markers = []):
    """INPUTS:
            axs: the axis to plot on
            posterior_dist: the posterior distribution over position (position_bins x time)
            predicted_position: the predicted position at each time point (time,) - this will be either the max or weighted average of posterior_dist
            actual_position: the actual position at each time point (time,)
            time: the time vector corresponding to the predicted and actual positions (in seconds)
            time_markers: a list of time points to mark with vertical lines (in seconds)"""

    axs.imshow(posterior_dist, aspect='auto', origin='lower', cmap='bone_r', vmin=0, vmax=.2,
           extent=[time[0], time[-1], 0, posterior_dist.shape[0]], interpolation='none')
    axs.plot(time, predicted_position, color="blue", linestyle="-", linewidth=1, clip_on=False,
            label = f"predicted {var_name}")
    axs.plot(time, actual_position, color="orange", linestyle="--", linewidth=1, clip_on=False,
            label = f"Actual {var_name}")
    if len(time_markers) > 0:
        for time_marker in time_markers:
            axs.axvline(x=time_marker, color='r', linestyle='--', linewidth=1)
    axs.set_ylabel(var_name)
    axs.set_yticks(np.arange(0, posterior_dist.shape[0]+1,posterior_dist.shape[0]//4))
    if 'escape' in var_name:
        axs.set_yticklabels(np.arange(0, 101, 25))
    elif 'speed' in var_name:
        axs.set_yticklabels(np.arange(0, 51, 12.5))
    axs.set_xlabel('time (s)')
    if rmse != '':
        axs.set_title(f"rmse: {rmse:.2f}")
    axs.legend(loc='lower right')

def plot_mouse_behaviour(axs, x, y, onset, offset, settings, time_markers = [], look_back = 0, look_forward = 0, title = ''):
    """ INPUTS:
            axs: the axis to plot on
            x: the x position of the mouse at each time point (time,) camera frames
            y: the y position of the mouse at each time point (time,) camera frames
            onset: the index of the onset of the homing period (camera frame)
            offset: the index of the offset of the homing period (camera frame)
            settings: a dictionary of settings that includes the arena condition and barrier location for plotting the arena
            time_markers: a list of time points to mark with scatter points (in frames relative to onset)
            look_back: how many seconds before onset to plot (default 0) i.e. what did the mouse do before this segment
            look_forward: how many seconds after offset to plot (default 0) i.e. what did the mouse do after this segment
            title: title for the plot (default '')"""
    Arena(ax=axs,
        condition=settings['replay_test_condition'],
        barrier_coordinates=settings['barrier_test_location'])
    axs.scatter(x[onset:offset], y[onset:offset], s=1, color='orange', alpha = 0.2) # to show where the mouse went during the homing period
    axs.scatter(x[onset-look_back:onset], y[onset-look_back:onset], s=1, color='k', alpha = 0.2) # to show where the mouse went before the homing period
    axs.scatter(x[offset:offset+look_forward], y[offset:offset+look_forward], s=1, color='k', alpha = 0.2) # to show where the mouse went after the homing period
    if len(time_markers) > 0:
        for time_marker in time_markers:
            axs.scatter(x[onset+int(40*time_marker)], y[onset+int(time_marker)], s=10, color='r', alpha = 0.5) # to show where the mouse was at each time marker
    axs.set_title(title)

def plot_behavioral_vars(axs, time, y, xlabel = '', ylabel = '', time_markers = [], color = 'blue', title = ''):
    """A function for plotting a behavioral variable (e.g. speed, escape route) over time during the homing period.
    INPUTS:
        axs: the axis to plot on
        time: the time vector corresponding to the variable (in seconds)
        y: the variable to plot at each time point (time,)
        title: title for the plot (default '')
        xlabel: label for the x axis (default '')
        ylabel: label for the y axis (default '')"""
    axs.plot(time, y, color=color, linewidth=1)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel, color=color)
    axs.set_xlim(0, time[-1])
    if len(time_markers) > 0:
        for time_marker in time_markers:
            axs.axvline(x=time_marker, color='r', linestyle='--', linewidth=1)