"""
This is a script for making egocentric firing maps as in 
Alexander, A.S., Carstensen, L.C., Hinman, J.R., Raudies, F., Chapman, G.W., Hasselmo, M.E., 2020. Egocentric boundary vector tuning of the retrosplenial cortex. Sci Adv 6, eaaz2322. https://doi.org/10.1126/sciadv.aaz2322
"""

import os

import numpy as np
import polars as pl
import cv2
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, filter_video_dataframe
from settings.settings_visualize import defined_settings_visualize as settings_v

def egocentric_firing_map(spike_data,video_data,session):
    '''
    This function has been replaced by egocentric_firing_map_binned.egocentric_firing_map
    This function looks at the position and head dir of the mouse each time a given neuron fires.
    It is more biased by uneven sampling of space and hdir.
    This function sets up making a firing map for an egocentric view of features in the arena.
    For each cluster it will make a figure of egocentric firing maps in each condition'''

    # saving path
    map_path = os.path.join(session.base_path,session.processed_path, 'spatial_firing', 'egocentric_map',settings_v.cluster_type)
    if not(os.path.exists(map_path)): os.makedirs(map_path)

    # make rendered arena image, add an offset for the cropping window
    window_size = 100
    barrier_location = None
    if len(session.barrier_time) > 0:
        barrier_location = [[value + window_size for value in inner_list] for inner_list in session.barrier_location]
    rendered_arena = generate_rendered_arena([session.video.height + (2*window_size),session.video.width + (2*window_size)], 
                                             [[value + window_size for value in inner_list] for inner_list in session.shelter_location], 
                                             barrier_location)
    
    

    for counter,cluster in enumerate(spike_data["spike_clusters"].unique()):
        make_map_by_cluster(cluster,session,video_data,spike_data.filter(spike_data['spike_clusters'] == cluster),rendered_arena, window_size,map_path)

def make_map_by_cluster(cluster,session,video_data,spike_data,rendered_arena, window_size,map_path):

    # align spike data for this cluster to video_df
    spike_data = spike_data.with_column(spike_data['spike_aligned_to_frame'].cast(pl.Int64))
    merged_df = video_data.join(spike_data, left_on='frames', right_on = 'spike_aligned_to_frame', how='inner')

    # identify conditions
    if settings_v.user_defined_conditions:
        conditions = settings_v.conditions
    else:
        conditions = identify_conditions(session)
        
    conditions = 'shelter_present'
    for c in conditions:
        video_df = filter_video_dataframe(merged_df, c, outofshelter=True, exclude_escape=True)
        video_df = video_df.select(['frames','hdir','mouse_x_position','mouse_y_position'])
        hdir = video_df['hdir'].to_numpy()
        position = np.vstack([video_df['mouse_x_position'].to_numpy(),video_df['mouse_y_position'].to_numpy()]).T
        # add window_size as an offset to video_data['mouse_x_position','mouse_y_position']
        position = np.round(position+window_size).astype(int)

        summed_rotated = np.zeros(shape = (2*window_size,2*window_size))
        for f in np.arange(len(video_df)):
            crop = rendered_arena[position[f,0]-window_size:position[f,0]+window_size,position[f,1]-window_size:position[f,1]+window_size]
            image = Image.fromarray(crop.astype(np.uint8))
            rotated_image = image.rotate(np.rad2deg(hdir[f]))
            summed_rotated = summed_rotated + rotated_image
        
        plt.imshow(summed_rotated)
        plt.savefig(str(map_path) + "/cluster" + str(cluster) + "_ego_maps.png")
        if settings_v.show_plots:
            plt.show()
        plt.close()

def single_cluster_plot(all_conditions, cluster, plot_save_path):
    """ Make a figure for each cluster with polar plots for all angles in all conditions of interest"""

    # Add one index for the titles
    nrows = len(all_conditions)
    ncols = 2
    
    # Plot settings
    gs = gridspec.GridSpec(nrows, ncols, width_ratios = [1] + [3] * (ncols-1),
                            height_ratios = [1] + [3] * (nrows-1),
                            wspace=0, hspace=0.4)
    # gridspec sets ratios such titles are narrower than plots
    fig = plt.figure(figsize=(30, 30)) # width, height
    axs_fontsize = 23

    # Add subtitles for each condition in first column
    for c_counter, c in enumerate(all_conditions):
        ax = plt.subplot(gs[c_counter + 1, 0])
        ax.text(0, 0.5, c, rotation='horizontal',
                va='center', ha='center', fontsize=axs_fontsize)
        ax.set_axis_off()

    # Create actual polar plots for each condition and angle
    for c_counter, c in enumerate(all_conditions):
        counter = ((ncols)*(c_counter+1)) + 1
        ax = plt.subplot(nrows, ncols, counter)
        # TODO: actually plot stuff in here

    # Save and close the figure
    plt.tight_layout()
    plt.savefig(str(plot_save_path) + "/cluster" + str(cluster) + "_polar_plots.png")
    if settings_v.show_plots:
        plt.show()
    plt.close()

def generate_rendered_arena(size, shelter_location, barrier_location) -> object:
    rendered_arena = np.zeros(size).astype(np.uint8)
    # arena outline
    cv2.circle(rendered_arena, (int(size[0]/2), int(size[1]/2)), 460, (255,255,255), thickness = 2, lineType = 16)
    # shelter location
    cv2.rectangle(rendered_arena, (shelter_location[0]), (shelter_location[1]), 255, thickness=-1)
    # barrier location
    if barrier_location:
        cv2.line(rendered_arena, (barrier_location[0]), (barrier_location[1]), 255, thickness=2)
    return rendered_arena