# Import OS Lib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl

def plot_heat_map_of_position(video_data_frame, filter_out_shelter_time = True):
    """ 
    Plot a heatmap of the mouse position, behaviour only. With an option to filter out the time the mouse is in the shelter and
    focus on the time the mouse is in the arena.
    """
    
    if filter_out_shelter_time: 
        video_data_frame = video_data_frame.filter(pl.col("OutofshelterIdx") == True)
        
    x_coords = video_data_frame['mouse_x_position']
    y_coords = video_data_frame['mouse_y_position']

    plt.figure(figsize=(10, 8))
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(50, 50))

    # Draw heatmap
    ax = sns.heatmap(heatmap, cmap="crest", robust=True)

    # Remove x and y tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove x and y ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Plot
    plt.title('Mouse Position Heatmap')
    plt.xlabel('Mouse X Position')
    plt.ylabel('Mouse Y Position')
    plt.show()
    
    x = 10