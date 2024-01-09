# Import standard lib
import os

# Import 3rd party Lib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Import custom lib
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe

def plot_heat_map_of_position(session, settings, video_data_frame, conditions, session_height, save_path) -> None:
    """Create heatplot of mouse exploration for each condition"""

    fig, axs = plt.subplots(nrows=1, ncols=len(conditions), figsize=(24, 6), sharey=True, sharex=True)
    cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])  # The list represents [left, bottom, width, height],
    # where all values are in fractional (0-1) coordinates.
    # Where to plot the colorbar, create new axis object at these coordinates
    for idx, condition in enumerate(conditions):
        video_data_frame_filtered = filter_video_dataframe(video_data_frame, condition)
        x_coords = video_data_frame_filtered["mouse_x_position"].to_numpy()
        y_coords = video_data_frame_filtered["mouse_y_position"].to_numpy()
        all_posX, all_posY = remove_points_away_from_center_of_circle(x_coords, y_coords, session_height)

        # Generate heatmap
        heatmap, _, _ = np.histogram2d(
            all_posX, all_posY, bins=(96, 96)
        )  # [int, int] - number of bins in x and y axis, abitrarily set

        # Plotting logic for the heatmap, transpose to get the right orientation and set zero to white
        axs[idx] = sns.heatmap(
            heatmap.T / 40,
            cmap="coolwarm",
            cbar_ax=cbar_ax,
            robust=True,
            ax=axs[idx],
            mask=(heatmap.T == 0),
            cbar_kws={"label": "Normalised time spent in position"},
            norm=plt.Normalize(vmin=0, vmax=1),
        )

        # Remove x and y tick labels and ticks
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        axs[idx].xaxis.set_ticks_position("none")
        axs[idx].yaxis.set_ticks_position("none")
        axs[idx].set_title(condition, fontsize=20)
        axs[idx].figure.axes[-1].yaxis.label.set_size(
            16
        )  # The legend is the last axis so this is a hack to change the font size of the legend

    plt.subplots_adjust(wspace=0.05, hspace=0)
    if settings.show_plots:
        plt.show()
    plt.savefig(os.path.join(save_path, "Heat_plots_of_mouse_position_per_condition.png"))
    plt.close()


# Utils for heatplot
def remove_points_away_from_center_of_circle(x, y, session_height) -> tuple:
    """
    Ensures there are no positions outside of the areana by removing them from the x and y coordinates based
    on the fact that the radius of the arena is 460 pixels.

    TODO:
    + Make this function global
    + Make the radius of the arena a variable not hard coded
    """

    dist = np.sqrt(
        ((x - session_height / 2) ** 2) + ((y - session_height / 2) ** 2)
    )  # Use the euclidean distance formula to find the distance from the center of the arena
    filtX = x[dist < 460]  # 460 is size of arena circle radius, see register
    filtY = y[dist < 460]
    return filtX, filtY
