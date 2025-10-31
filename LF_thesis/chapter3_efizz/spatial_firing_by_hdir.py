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

path = r"Z:\Jasmine_Laurence\Experimental_Data\JAL004\004_flipppuf19sept_2023_09_19T14_10_56\processed_data\good_video_spike_count_df.parquet"

# load parquet file
data = pl.read_parquet(path)
save = r'Z:\Laurence\thesis\figures\spatial_firing_by_hdir'

print(data)


def spatial_position_firing_hdir(data, clu_label, video_df, save_path, show_plots):
    """A function that plots the position of the mouse at every AP of a given cluster and colours it by hdir"""
    logger.info("Commence making figures of spatial position firing plots coloured by hdir of all clusters")
    cc = matplotlib.cm.Reds  # could use Reds or copper
    # set number of rows and calculate number of columns
    ncols = 10
    nrows = 5  # nclu // ncols + (nclu % ncols > 0)
    figg, axs = plt.subplots(nrows, ncols)
    figg.set_figwidth(30)
    figg.set_figheight(15)
    fnum = 1
    axs = axs.ravel()
    
    cluster_filt = 11
    
    ids = video_df["spike_clusters"].unique().to_list()
    this = ids[10]

    # what is firing rate per frame?
    for counter, cluster in enumerate(data["spike_clusters"].unique()):
        if counter >= (ncols * nrows) * fnum:
            figg, axs = plt.subplots(nrows, ncols)
            figg.set_figwidth(30)
            figg.set_figheight(15)
            fnum = fnum + 1
            axs = axs.ravel()
            
        # filter spikes by cluster
        filt = video_df.filter(video_df["spike_clusters"] == this)
        hdir = np.digitize(np.rad2deg(filt["hdir"]), np.arange(-180, 180))
        cc = hsv_hdir_colormap(hdir)
        # axs[counter - (nrows * ncols * fnum)].scatter(
        #     video_df["mouse_x_position"].to_numpy(), video_df["mouse_y_position"].to_numpy(), s=3, color=[0.7, 0.7, 0.7], linewidths=0, marker="."
        # )
        
        # all mouse positions
        axs[counter - (nrows * ncols * fnum)].scatter(filt["mouse_x_position"].to_numpy(), filt["mouse_y_position"].to_numpy(), s=7, c=cc, linewidths=0, marker=".")  # this neuron's firing coloured by hdir
        axs[counter - (nrows * ncols * fnum)].set_axis_off()
        axs[counter - (nrows * ncols * fnum)].invert_yaxis()
        axs[counter - (nrows * ncols * fnum)].set_aspect("equal")
        
        plt.savefig(str(save_path) + "_clusters_spatial_firing_hdir_colored_clu11" + str(fnum) + ".eps", format="eps")
        plt.close()
        break

spatial_position_firing_hdir(data, None, data, save, show_plots=False)