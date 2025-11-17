"""a set of functions for visualizing the tuning (e.g. spatial) of neurons"""

# set up
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import polars as pl

# import
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.visualize.behaviour.behavioral_stats import hsv_hdir_colormap

ANGLE_BIN_EDGES = np.linspace(-180, 180, 361)

path = r"Z:\Jasmine_Laurence\Experimental_Data\JAL004\004_flipppuf19sept_2023_09_19T14_10_56\processed_data\good_video_spike_count_df.parquet"

# load parquet file
data = pl.read_parquet(path)
save = r"Z:\Laurence\thesis\figures\spatial_firing_by_hdir"

print(data)
print(data.columns)


def digitize_head_direction(hdir_radians):
    """Convert radians to hsv_hdir_colormap bin indices (1-360)."""
    degrees = np.rad2deg(np.asarray(hdir_radians))
    bins = np.digitize(degrees, ANGLE_BIN_EDGES, right=False)
    return np.clip(bins, 1, 360)


def generate_color_wheel(ax):
    """Plot a polar colour wheel that matches hsv_hdir_colormap bins."""
    bin_indices = np.arange(1, 361)
    colors = hsv_hdir_colormap(bin_indices)

    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    width = (2 * np.pi) / 360
    ax.bar(theta, np.ones_like(theta), width=width, align="edge", bottom=0, color=colors, edgecolor="none")

    ax.set_rticks([])
    ax.set_xticks(np.deg2rad(np.array([0, 45, 90, 135, 180, 225, 270, 315])))
    ax.set_xticklabels(["0 deg", "45 deg", "90 deg", "135 deg", "180 deg", "-135 deg", "-90 deg", "-45 deg"], fontsize=6)
    ax.set_yticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Angle Legend", fontsize=8, pad=10)


def spatial_position_firing_hdir(video_df, save_path):
    """function that plots the position of the mouse at every AP of a given cluster and colours it by hdir"""
    logger.info("Commence making figures of spatial position firing plots coloured by hdir of all clusters")
    cc = matplotlib.cm.Reds  # could use Reds or copper

    # filter on OutofSehlterIdx == 1
    # video_df = filter_video_dataframe(video_df, condition=None, excl_stationary=False, exclude_escape=True, exclude_homings=False)

    ids = video_df["spike_clusters"].unique().to_list()
    this = ids[10]
    filt = video_df.filter((video_df["spike_clusters"] == this))
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.3)
    plot_ax = fig.add_subplot(gs[0, 0])
    wheel_ax = fig.add_subplot(gs[0, 1], projection="polar")
    arena_center_x = 512  # example value, replace with actual center x-coordinate
    arena_center_y = 512  # example value, replace with actual center y-coordinate
    arena_radius = 460

    # Plot transparent points for all mouse positions in grey
    #circular mask the video_df to only include points within the arena
    video_df = video_df.filter(
        ((video_df["mouse_x_position"] - arena_center_x) ** 2 + (video_df["mouse_y_position"] - arena_center_y) ** 2) <= (arena_radius ** 2)
    )
    
    plot_ax.scatter(video_df["mouse_x_position"].to_numpy(), video_df["mouse_y_position"].to_numpy(), s=7, c="grey", linewidths=0, marker=".", alpha=0.01)

    # Plot firing positions coloured by hdir
    spike_filt = filt.filter(filt["spike_count"] > 0)

    # filter spike_filt only out of shelter and no escapes
    spike_filt = filter_video_dataframe(spike_filt, condition=None, excl_stationary=False, exclude_escape=True, exclude_homings=False)
    
    # Remove any points that are outside the arena ciruclar bounds using the center and radius of the arena to filter
    spike_filt = spike_filt.filter(
        ((spike_filt["mouse_x_position"] - arena_center_x) ** 2 + (spike_filt["mouse_y_position"] - arena_center_y) ** 2) <= (arena_radius ** 2)
    )

    hdir = digitize_head_direction(spike_filt["hdir"])
    cc = hsv_hdir_colormap(hdir)
    plot_ax.scatter(spike_filt["mouse_x_position"].to_numpy(), spike_filt["mouse_y_position"].to_numpy(), s=7, c=cc, linewidths=0, marker=".")  # this neuron's firing coloured by hdir

    plot_ax.set_axis_off()
    plot_ax.invert_yaxis()
    plot_ax.set_aspect("equal")

    generate_color_wheel(wheel_ax)

    plt.show()

    plt.savefig(str(save_path) + "_clusters_spatial_firing_hdir_colored_clu11" + ".eps", format="eps", dpi=100)
    # plt.savefig(str(save_path) + "_clusters_spatial_firing_hdir_colored_clu11" + ".png", format="png", dpi=300)
    plt.close()


spatial_position_firing_hdir(data, save)
