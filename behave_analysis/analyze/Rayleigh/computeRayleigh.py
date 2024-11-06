"""Create all things rayleigh vector related"""

import os

from tqdm import tqdm
from loguru import logger
import numpy as np
import polars as pl
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.use("TkAgg")

from settings.settings_analyze_efizz import Settings_ae
from behave_analysis.analyze.stats.linshit import LinearShift
from behave_analysis.analyze.LDA.LDA_preprocess import select_relevant_frames
from behave_analysis.analyze.LDA.LDA_utils import list_conditions, BuildSavingFolder
from behave_analysis.analyze.filtering_data.filtering_functions import (
    identify_angles,
    generate_bins,
)
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.utils.PersistentPool import PersistentPool

# ----------------------------- Main functions from analyze efizz -----------------------------


def compute_all_clusters_rayleigh(self, all_angles: list) -> None:
    """Compute rayleigh for all angles in all desired conditions AND if Settings_analyze_efizz.multi_cluster_plots = True,
    it also plots all clusters per angle"""
    if self.settings.linear_shift:
        pool = PersistentPool(workers = 20)
    else:
        pool = None
    # determine condition types
    self.number_of_homings, condition_types = list_conditions(self.settings)
    for cond in condition_types:  # e.g. 'experimental_conditions', 'behavioral_conditions'
        self.condition_types = cond
        for c in self.all_conditions:
            self.condition = c
            filtered_video_df = select_relevant_frames(self)
            data_path = BuildSavingFolder(self.dir, self.settings, self.cluster_type, self.condition_types, self.condition, compartment = [])
            compartment = identify_which_compartment(self, filtered_video_df)
            if np.logical_or(not check_if_rayleigh_exists(self, all_angles), self.settings.redo_compute):
                for a in all_angles:
                    this_df = filtered_video_df.select(["frames", a])
                    frames = this_df["frames"].unique().to_numpy() - 1  # -1 to match python indexing
                    X = self.frame_by_cluster_matrix
                    X = X[frames, :]
                    logger.info("Calculating Rayleigh vectors for " + str(a) + " in condition: " + str(c))
                    rayleigh_vector(self, this_df, X, a, data_path, compartment, pool)
        # plot all conditions in this condition types for each cluster
        if self.settings.single_cluster_plots:
            data_path = BuildSavingFolder(self.dir, self.settings, self.cluster_type, self.condition_types, condition = [], compartment = [])
            plot_save_path = make_directory(os.path.join(data_path, "single_cluster_plots"))
            single_cluster_plots(self, all_angles, self.all_conditions, data_path, plot_save_path)
    if self.settings.linear_shift:
        pool.close()


# -------------------------------------------------------------------------------------------------

def check_if_rayleigh_exists(self, all_angles):
    """Check if the Rayleigh vectors have already been computed and saved"""
    data_path = BuildSavingFolder(self.dir, self.settings, self.cluster_type, self.condition_types, self.condition)
    for a in all_angles:
        if os.path.isfile(data_path + "/" + str(a) + "_Rayleigh.arrow"):
            continue
        else:
            return False
    logger.info("Rayleigh vectors exist, either recompute or move on to plotting")
    return True


def single_cluster_plots(self, all_angles, all_conditions, data_path, plot_save_path):
    """Generate a polar plot per condition and angle for a single cluster

    Arguments:
        settings (dataclass): settings for the analysis
        all_angles (list): list of angles to consider, each element is a string
        all_conditions (list): list of conditions to consider, each element is a string
        base_path (str): path to the directory where the data is stored
        plot_save_path (str): path to the directory where the plots will be saved
    """

    logger.info("Making individual cluster polar plots")
    clusters = self.cluster_Ids

    # Add one index for the titles
    nrows = len(all_conditions) + 1
    ncols = len(all_angles) + 1

    for clu in clusters:
        if clu > 0:
            # Plot settings
            gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1] + [3] * (ncols - 1), height_ratios=[1] + [3] * (nrows - 1), wspace=0, hspace=0.4)
            # gridspec sets ratios such titles are narrower than plots
            fig = plt.figure(figsize=(30, 30))  # width, height
            axs_fontsize = 23
            labels = ["Shelter compartment", "Threat zone compartment"]
            col = ["cornflowerblue", "darkorchid"]
            legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(col, labels)]
            fig.legend(handles=legend_elements, loc="upper right", fontsize=axs_fontsize, handlelength=4)

            # Add subtitles for each angle in first row
            for a_counter, a in enumerate(all_angles):
                ax = plt.subplot(gs[0, a_counter + 1])
                ax.text(0.5, 0.5, a, rotation="horizontal", va="center", ha="center", fontsize=axs_fontsize)
                ax.set_axis_off()

            # Add subtitles for each condition in first column
            for c_counter, c in enumerate(all_conditions):
                ax = plt.subplot(gs[c_counter + 1, 0])
                ax.text(0, 0.5, c, rotation="horizontal", va="center", ha="center", fontsize=axs_fontsize)
                ax.set_axis_off()

            # Extract the max firing rate across all conditions and angles for this cluster
            max_firing_rate = extract_max_hz(clu, all_angles, all_conditions, data_path)

            # Create actual polar plots for each condition and angle
            for c_counter, condition in enumerate(all_conditions):
                counter = ((ncols) * (c_counter + 1)) + 1
                for a_counter, a in enumerate(all_angles):
                    counter = counter + 1
                    ax = plt.subplot(nrows, ncols, counter, projection="polar")
                    rayleigh_results = pl.read_ipc(os.path.join(data_path, condition) + "/" + str(a) + "_Rayleigh.arrow")
                    pcentile = compute_95th_percentile_rayleigh(rayleigh_results)
                    # make actual polar plot for a given angle in a given condition
                    polar_plot(
                        rayleigh_results.filter(rayleigh_results["clusterID"] == clu),
                        ax,
                        fig,
                        pcentile=pcentile,
                        cluster_title=False,
                        max_firing_rate=max_firing_rate,
                    )

            # Save and close the figure
            plt.tight_layout()
            plt.savefig(str(plot_save_path) + "/cluster" + str(clu) + "_polar_plots.png")
            if self.settings.show_plots:
                plt.show()
            plt.close()


def extract_max_hz(clu: int, all_angles: list, all_conditions: list, base_path: str) -> int:
    """Extract the max firing rate across all conditions and angles for this cluster
    such that the polar plots can be scaled accordingly.

    Args:
        clu (int): What cluster to extract the max firing rate for
        all_angles (list): a list of angles to consider, each a string
        all_conditions (list): a list of conditions to consider, each a string
        base_path (str): path to the directory where the data is stored
    """
    max_firing_rate = 0
    for _, condition in enumerate(all_conditions):
        data_path = os.path.join(base_path, condition)
        for _, angle in enumerate(all_angles):
            rayleigh_results = pl.read_ipc(data_path + "/" + str(angle) + "_Rayleigh.arrow")
            rayleigh_results = rayleigh_results.filter(rayleigh_results["clusterID"] == clu)
            max_element = np.max([np.max(sub_array) for sub_array in rayleigh_results["angle_firing_hist"].to_numpy()])
            if max_element > max_firing_rate:
                max_firing_rate = max_element
    return int(max_firing_rate)

def rayleigh_vector(
    self,
    filtered_video_df: pl.DataFrame,
    X: np.array, # (n_frames, n_clusters)
    angle_filt,
    plot_save_path,
    compartment: np.array,
    pool=None,
) -> None:
    """Calculate the rayleigh vector (amplitude and anlge) for each cluster w.r.t the angles given (e.g. HD or HSA)

    Considerations:
    -- It only considers times when the mouse was outside the shelter
    -- It also performs bootstrapping by computing the rayleigh vector at random time shifts of the spikes with respect to the angles
    -- The Rayleigh vector is significant if the amplitude is above the 95th percentile of boostrapped amplitudes
    -- Rayleigh's R close to zero = untuned, fires at all head directions
    -- Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation

    Arguments:
    -- compartment: a numpy array of ones and twos of the same length as the filtered_video_df
            where 1 is the shelter side and 2 is the threat zone side
    TODO - Add more details on what each input argument is

    
    NOTE - The first index of the multi dimensional arrays is the shelter compartment and the second is the threat zone
    Returns (Saves to the processed folder):
    -- A dataframe with the following columns:
        clusterID,
        Rayleigh, - Tis is actually split in two vectors for each compartment
        Rayleigh_theta,  - Tis is actually split in two vectors for each compartment
        Rayleigh_sig,  - Tis is actually split in two vectors for each compartment
        angle_firing_hist,  - Tis is actually split in two vectors for each compartment
        angles  - Tis is actually split in two vectors for each compartment

    TODO:
    -- Add whole arena values to the dataframe
    -- Add sig test for whole arena rayleigh values
    """

    # edges for binning firing rate at different angles
    bin_angles, bin_angle_center = generate_bins(number_of_bins=self.settings.number_of_bins, start=-np.pi, stop=np.pi)

    # Catch empty video dataframes
    assert len(filtered_video_df) > 0, "Video dataframe is empty, bug."

    # bin angles
    binned_angles = np.array(filtered_video_df[angle_filt].to_numpy())
    binned_angles = np.digitize(binned_angles, bin_angles) - 1

    # initialize variables to compute the Rayleigh vector
    # Remove cluster 0 from ks only - synthetic starts at cluster id 1
    cluster_Ids = self.cluster_Ids[self.cluster_Ids > 0]
    Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist, arena_rayleigh_theta, arena_rayleigh, arena_sig = init_rayleigh(
        cluster_Ids, len(np.unique(compartment)), bin_angle_center
    )

    # Check that np.unique returns the correct order of compartments
    assert np.array_equal(np.unique(compartment), np.array([1, 2])), "Compartments are not as expected"

    # assign spike times of each cluster to the corresponding video frame, then assign HD
    for count in tqdm(np.arange(len(cluster_Ids)), desc=f"Running Rayleigh on cluster out of  {len(cluster_Ids)}"):
        Rayleigh_cluster[count] = cluster_Ids[count]

        # ----------------------------Whole arena computations-----------------------------------------------------------
        # Check if all the spike counts across frames for this cluster are zero
        if sum(X[:, count] == 0) == len(X[:, count]):
            logger.warning(f"Cluster {count} has no spikes, skipping")
            continue

        arena_rayleigh[count], arena_rayleigh_theta[count], _ = compute_rayleigh_cluster(X=X[:, count], y=binned_angles, return_all_stats=True)

        # Significance test for whole arena
        if self.settings.rayleigh_significance == "bootstrap":
            arena_sig[count] = bootstrap_rayleigh_significance(
                binned_angles=binned_angles,
                count=count,
                rayleigh=arena_rayleigh,
                X=X[:, count],
                fps=self.session.video.fps,
                flag="whole_arena",
            )

        elif self.settings.linear_shift:
            arena_sig[count] = linearshift_rayleigh_significance(X=X[:, count], binned_angles=binned_angles, pool=pool)

        # ---------------------- Specific compartment computations ------------------------------------------------------
        # This should first loop through comparment 1 (shelter zone) and then compartment 2 (threat zone) 
        
        for c_count, comp in enumerate(np.unique(compartment)):
            # Check if all the spike counts across frames for this cluster are zero
            if sum(X[compartment == comp, count] == 0) == len(X[compartment == comp, count]):
                logger.warning(f"Cluster {count} has no spikes, skipping")
                continue

            Rayleigh[count, c_count], Rayleigh_theta[count, c_count], angle_firing_hist[count, :, c_count] = compute_rayleigh_cluster(
                X[compartment == comp, count], binned_angles[compartment == comp], nbins=self.settings.number_of_bins, return_all_stats=True
            )

            # Linear shifts performed at a random offset between 0 and 100 seconds to generate a null distribution to detect non-sense correlations
            if self.settings.linear_shift:
                Rayleigh_sig[count, c_count] = linearshift_rayleigh_significance(
                    X=X[compartment == comp, count],
                    binned_angles=binned_angles[compartment == comp],
                    pool=pool,
                )

            # alternative method with bootstrap
            elif self.settings.rayleigh_significance == "bootstrap":
                Rayleigh_sig[count, c_count] = bootstrap_rayleigh_significance(
                    binned_angles=binned_angles,
                    comp=comp,
                    count=count,
                    nbins=self.settings.number_of_bins,
                    rayleigh=Rayleigh,
                    X=X[compartment == comp, count],
                    fps=self.session.video.fps,
                    compartment=compartment,
                    flag="compartments",
                )

    # histogram of rayleighs
    plt.figure()
    plt.hist(Rayleigh, np.arange(0, 1, 0.1), alpha = .5, color = ['cornflowerblue','darkorchid'])
    plt.hist([Rayleigh[Rayleigh_sig[:,0] == 1,0],Rayleigh[Rayleigh_sig[:,1] == 1,1]], np.arange(0, 1, 0.1), alpha = .5, color = ['cornflowerblue','darkorchid'])
    # plt.hist(Rayleigh, np.arange(0, 1, 0.1), alpha = 0.5)
    # plt.hist(Rayleigh[Rayleigh_sig == 1,:], np.arange(0, 1, 0.1), alpha = 0.5)
    plt.xlabel("Rayleigh R")
    plt.ylabel("number of clusters")
    plt.savefig(plot_save_path + "/" + str(angle_filt) + "_Rayleigh_vector_hist.png")
    if self.settings.show_plots:
        plt.show()
    plt.close()

    # save rayleigh info for all cluster to csv for this angle in this condition
    if len(np.shape(angle_firing_hist)) == 3:
        angle_firing_hist = np.reshape(
            angle_firing_hist, [np.shape(angle_firing_hist)[0], np.shape(angle_firing_hist)[1] * np.shape(angle_firing_hist)[2]]
        )

    rayleigh_results = pl.DataFrame(
        {
            "clusterID": Rayleigh_cluster,
            "Rayleigh": Rayleigh,
            "Rayleigh_theta": Rayleigh_theta,
            "Rayleigh_sig": Rayleigh_sig,
            "angle_firing_hist": angle_firing_hist,
            "angles": np.tile(bin_angle_center, (len(Rayleigh), 1)),
            "arena_rayleigh_theta": arena_rayleigh_theta,
            "arena_rayleigh": arena_rayleigh,
            "arena_sig": arena_sig,
        }
    )

    # Save the rayleigh results to a file. This is the main output of this function
    rayleigh_results.write_ipc(plot_save_path + "/" + str(angle_filt) + "_Rayleigh.arrow")

    logger.info("Finished calculating Rayleigh vectors, moving on to polar plots")
    if self.settings.multi_cluster_plots:
        folder_name = os.path.join(plot_save_path, str(angle_filt) + "_cluster_tuning_plots")
        if not (os.path.exists(folder_name)):
            os.makedirs(folder_name)
        all_clusters_polar_plots(rayleigh_results, folder_name, self.settings.show_plots)


def bootstrap_rayleigh_significance(
    binned_angles, count: int, rayleigh, X, fps: int, flag: str, nbins, comp: int = None, compartment: np.array = None
) -> int:
    """A function that computes the significance of the rayleigh magnitude using bootstrapping"""
    # logger.info(f"Computing significance of Rayleigh vector using bootstrapping for neuron index {count}")
    x = 100
    shift_dist = np.empty(x)

    if flag == "compartments":
        # shuffled linear shifts performed at a random offset between 0 and 100 seconds
        for it in np.arange(x):
            shift = int(np.random.uniform(1, 100)) * fps
            ang_roll = np.roll(binned_angles, shift)
            shift_dist[it] = compute_rayleigh_cluster(X=X[compartment == comp, count], y=ang_roll[compartment == comp], nbins=nbins)

        significance = 0
        if rayleigh[count][int(comp) - 1] > np.percentile(shift_dist, 95):
            significance = 1
            # logger.success("Significant rayleigh vector found")

    if flag == "whole_arena":
        # shuffled linear shifts performed at a random offset between 0 and 100 seconds
        for it in np.arange(x):
            shift = int(np.random.uniform(1, 100)) * fps
            ang_roll = np.roll(binned_angles, shift)
            shift_dist[it] = compute_rayleigh_cluster(X, ang_roll)

        significance = 0
        if rayleigh[count] > np.percentile(shift_dist, 95):
            significance = 1

    # logger.info(f"Significance of Rayleigh vector using bootstrapping for neuron index {count} is {significance}")
    return significance


def linearshift_rayleigh_significance(X: np.array, binned_angles: np.array, pool) -> int:
    """Compute the significance of the rayleigh magnitude using linear shift

    Returns:
    -- significance: 1 if the rayleigh vector is significant, 0 if not"""
    # logger.info("starting linear shift significance test")
    LS_output = LinearShift(
        X, y=binned_angles, stat_computation_func=compute_rayleigh_cluster, size_of_central_chunk=np.round(np.shape(X)[0] / 3), PPool=pool
    )

    significance = 0
    # significance logical
    if LS_output.reject_null:
        significance = 1
    #     logger.success("Significant rayleigh vector found")
    # logger.info("finished linear shift significance test")

    return significance


def all_clusters_polar_plots(rayleigh_results, save_path, show_plots):
    """
    It makes a polar plot of firing at each angle (e.g. HD or HSA) for each cluster.
    self.tuning_dict['angles'] is a binned set of angles and self.tuning_dict[title][0] will give you the firing rates for cluster 0.
    Where the title is what the tunning was calculated for (e.g. 'HD' or 'HSA').
    """
    # ---------------------------------------------------
    # set up polar plots figure
    # set number of rows and calculate number of columns
    ncols = 10
    nrows = 5  # nclu // ncols + (nclu % ncols > 0)
    figg, axs = plt.subplots(nrows, ncols)
    figg.set_figwidth(30)
    figg.set_figheight(15)
    fnum = 1
    axs = axs.ravel()

    # assign spike times of each cluster to the corresponding video frame, then assign HD

    number_of_clusters = len(rayleigh_results)
    for counter in np.arange(number_of_clusters):
        # if you have filled a figure with polar plots, move onto next figure
        if counter >= (ncols * nrows) * fnum:
            figg, axs = plt.subplots(nrows, ncols)
            figg.set_figwidth(30)
            figg.set_figheight(15)
            fnum = fnum + 1
            axs = axs.ravel()

        ax = plt.subplot(nrows, ncols, 1 + counter - (nrows * ncols * (fnum - 1)), projection="polar")

        # polar plots!
        polar_plot(rayleigh_results.filter(np.arange(len(rayleigh_results)) == counter), ax, figg, pcentile=0)

        # save the whole figure
        if np.logical_or(counter - (nrows * ncols * (fnum - 1)) == (ncols * nrows) - 1, counter == number_of_clusters - 1):
            plt.tight_layout()
            plt.savefig(str(save_path) + "/cluster_polar_plots_" + str(fnum) + ".png")
            if show_plots:
                plt.show()
            plt.close()


## --------------------FUNCS ---------------------------------


def init_rayleigh(number_of_clusters, compartments, bin_angle_center):
    """Initializes the variables needed to compute the Rayleigh test

    Arguments:
    -- comparmentts (int): the number of compartments in the arena which is 2"""

    rayleigh_theta = np.empty([len(number_of_clusters), compartments])  # preferred angle
    rayleigh = np.empty([len(number_of_clusters), compartments])  # amplitude of Rayleigh vector
    rayleigh_sig = np.zeros([len(number_of_clusters), compartments])  # is the Ryleigh significant?
    rayleigh_cluster = np.empty([len(number_of_clusters)])  # which cluster ID is this Rayleigh value for?
    angle_firing_hist = np.empty([len(number_of_clusters), len(bin_angle_center), compartments])

    # Whole arena values
    arena_rayleigh_theta = np.empty([len(number_of_clusters)])  # preferred angle for the whole arena
    arena_rayleigh = np.empty([len(number_of_clusters)])  # amplitude of Rayleigh vector for the whole arena
    arena_sig = np.zeros([len(number_of_clusters)])  # is the Ryleigh significant for the whole arena?

    return rayleigh_theta, rayleigh, rayleigh_sig, rayleigh_cluster, angle_firing_hist, arena_rayleigh_theta, arena_rayleigh, arena_sig


def rayleigh(angles, firing) -> tuple:
    """Compute the rayleigh vector for a given set of angles and firing rates"""
    if np.sum(firing) == 0:
        theta = np.float64(0)
        r = np.float64(0)
    else:
        x = np.sum(np.cos(angles) * (firing)) / np.sum(firing)
        y = np.sum(np.sin(angles) * (firing)) / np.sum(firing)
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
    return r, theta


def firing_by_angle_bin(angles, neural_activity, nbins):
    angles_firing = np.zeros(nbins - 1)
    unique_groups, group_counts = np.unique(angles, return_counts=True)
    group_sums = np.bincount(angles, weights=neural_activity)
    angles_firing[unique_groups] = group_sums[unique_groups] / group_counts
    return angles_firing


def compute_rayleigh_cluster(X, y, nbins=Settings_ae.number_of_bins, return_all_stats=False):
    """This only works if there are no angle bins that are completely empty (angles that never occur)"""
    # compute firing in angle bins
    angle_firing_hist = firing_by_angle_bin(y, X, nbins)  # len(np.unique(y)))
    _, bin_angle_center = generate_bins(number_of_bins=nbins, start=-np.pi, stop=np.pi)
    # compute rayleigh
    Rval, Rtheta = rayleigh(bin_angle_center, angle_firing_hist)
    if return_all_stats:
        return Rval, Rtheta, angle_firing_hist
    else:
        return Rval


def compute_95th_percentile_rayleigh(rayleigh_results):
    """Compute the 95th percentile of the rayleigh distribution for each angle and condition"""
    flat_list = [item for sublist in rayleigh_results["Rayleigh"].to_list() for item in sublist]  # unpack a series of lists into a single list
    return np.percentile(flat_list, 95)


def identify_which_compartment(self, filtered_video_df: pl.DataFrame) -> np.ndarray:
    """Identify which compartment the mouse is in based on y position

    Compartments:
    -- compartment 1 (in blue) is the side where the shelter is
    -- compartment 2 (in purple) is the side wherethe threat zone is

    Returns:
    -- compartment: a numpy array of ones and twos of the same length as the filtered_video_df"""
    compartment = np.ones([len(filtered_video_df)])
    if len(self.session.barrier_time) > 0:
        compartment[filtered_video_df["mouse_y_position"].to_numpy() < 512] = 2  # threat zone
    return compartment


## ---------------------PLOTTING -----------------------------


def polar_plot(df, ax, fig, pcentile, max_firing_rate, cluster_title=True, plot_type="line") -> None:
    """Creates a polar plot for a single cluster in a single condition

    Visulises the firing rate at each angle (e.g. HD or HSA) for different
    compartments of the arena (e.g. shelter or threat zone).

    Arguments:
        df (pl.DataFrame) with columns clusterID, Rayleigh, Rayleigh_theta, Rayleigh_sig, angle_firing_hist, angles
        counter (int) the index of the cluster you want to plot
        ax (object) the axis index of the subplot
        plot_type (str) e.g line, bar, fill
    """

    # Check if the dataframe is empty
    if len(df["angle_firing_hist"].to_list()) == 0:
        return

    # Plot settings
    col = ["cornflowerblue", "darkorchid"]
    angle_firing = df["angle_firing_hist"].to_list()

    if np.shape(angle_firing)[1] > np.shape(df["angles"].to_list())[1]:
        angle_firing = np.reshape(
            angle_firing, [int(np.shape(df["angles"].to_list())[1]), int(np.shape(angle_firing)[1] / np.shape(df["angles"].to_list())[1])]
        )
        angle_firing = angle_firing.T

    for compartment in np.arange(len(df["Rayleigh"][0])):
        # Plot the rayleigh vector magnitude and angle
        ax.vlines(
            x=df["Rayleigh_theta"][0][int(compartment)],
            ymin=0,
            ymax=df["Rayleigh"][0][int(compartment)] * np.amax(angle_firing[compartment]),
            # ymax is rayleigh vector amplitude * max firing rate
            colors=col[compartment],
        )

        # Plot the firing rate at each angle this is the actual polar plot code

        # Settings for the polar plot bars
        if plot_type == "bar":
            ax.bar(
                x=df["angles"].to_list()[0],
                height=angle_firing[compartment],
                width=(2 * np.pi) / (len(angle_firing[compartment]) + 1),
                bottom=0.0,
                color=col[compartment],
                alpha=0.5,
            )

        # Testing out a different way to plot the polar plot bars with area filled in
        # Plot the firing rate at each angle this is the actual polar plot code
        else:
            angles = np.concatenate([df["angles"].to_list()[0], [df["angles"].to_list()[0][0]]])
            values = np.concatenate([angle_firing[compartment], [angle_firing[compartment][0]]])

            # Polar plot area with fill
            if plot_type == "fill":
                ax.fill(angles, values, color=col[compartment], alpha=0.5)

            # Polar plot area with no fill, just outline
            elif plot_type == "line":
                ax.plot(angles, values, color=col[compartment], linewidth=1.5)
                ax.set_ylim(bottom=0, top=max_firing_rate)  # set the y-axis limits to the max firing rate to make the plot more readable

    # Settings for the polar plot grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5, markevery=3)

    # Thin out y-ticks by taking every third tick
    current_ticks = ax.get_yticks()
    n = 3
    new_ticks = current_ticks[::n]
    ax.set_yticks(new_ticks)

    # Change number of decimal places to 1 for y-ticks
    new_tick_labels = [f"{tick:.1f}" for tick in new_ticks]
    ax.set_yticklabels(new_tick_labels)

    # Add title to the subplot
    if cluster_title:
        clutitle = "clu" + str(int(df["clusterID"][0]))
    else:
        clutitle = " "
    for compartment in np.arange(len(df["Rayleigh"][0])):
        clutitle = clutitle + "\n" + "Rayleigh = " + str(np.around(df["Rayleigh"][0][int(compartment)], 2))
        if df["Rayleigh_sig"][0][int(compartment)] == 1:
            clutitle = clutitle + " (sig.)"
        if df["Rayleigh"][0][int(compartment)] > pcentile:
            fig.suptitle("This cluster is worth a check", fontsize=30)
    ax.title.set_text(clutitle)
