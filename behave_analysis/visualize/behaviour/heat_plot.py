# Import OS Lib

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl
import os

# Import custom lib

from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from settings.settings_visualize import defined_settings_visualize as settings

def plot_heat_map_of_position(video_data_frame, session_height, save_path, filter_out_shelter_time = True) -> None:
    """ 
    Plot a heatmap of the mouse position, behaviour only. With an option to filter out the time the mouse is in the shelter and
    focus on the time the mouse is in the arena. Plots for each of the conditions placed in the settings file.
    """
    
    if filter_out_shelter_time: 
        video_data_frame = video_data_frame.filter(pl.col("OutofshelterIdx") == True)
        
    # Adjust figsize and number of columns as per your needs
    fig, axs = plt.subplots(nrows=1, ncols=len(settings.conditions_to_plot), figsize=(24, 6), sharey=True, sharex=True)
    cbar_ax = fig.add_axes([.91, .13, .01, .75]) # The list represents [left, bottom, width, height], where all values are in fractional (0-1) coordinates.

    for idx, condition in enumerate(settings.conditions_to_plot):
        
        # Extract condition specific section of the tracking data
        video_data_frame_filtered  = filter_video_dataframe(video_data_frame, condition)
        x_coords = video_data_frame_filtered ['mouse_x_position'].to_numpy()
        y_coords = video_data_frame_filtered ['mouse_y_position'].to_numpy()
    
        # Remove all positions outside of the arena - TODO make this function global
        dist = np.sqrt(((x_coords - session_height/2)**2) + ((y_coords - session_height/2)**2)) # 
        all_posX = x_coords[dist<460] # 460 is size of arena circle radius, see register
        all_posY = y_coords[dist<460]

        # Generate heatmap
        heatmap, _, _ = np.histogram2d(all_posX, all_posY, bins=(96, 96)) # [int, int] - number of bins in x and y axis, abitrarily set
        
        # Plotting logic for the heatmap, transpose to get the right orientation and set zero to white
        axs[idx] = sns.heatmap(heatmap.T / 40, 
                               cmap="coolwarm",
                               cbar_ax = cbar_ax,
                               robust=True, 
                               ax=axs[idx], 
                               mask=(heatmap.T==0), 
                               cbar_kws={'label': 'Normalised time spent in position'},
                               norm = plt.Normalize(vmin=0, vmax=1))

        # Remove x and y tick labels and ticks
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        axs[idx].xaxis.set_ticks_position('none')
        axs[idx].yaxis.set_ticks_position('none')
        axs[idx].set_title(condition, fontsize = 20)
        axs[idx].figure.axes[-1].yaxis.label.set_size(16) # The legend is the last axis so this is a hack to change the font size of the legend

    plt.subplots_adjust(wspace=0.05, hspace=0)
    
    plt.show()
    
    if settings.show_plots: plt.show()
    plt.savefig(os.path.join(save_path, "Heat_plots_of_mouse_position_per_condition.png"))
    plt.close()
