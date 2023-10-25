# script for computin rayleigh vectors
from loguru import logger
import numpy as np
import polars as pl
import os
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from behave_analysis.analyze.filtering_data.filtering_functions  import filter_video_dataframe, identify_angles, generate_bin_angles, identify_conditions
import matplotlib.gridspec as gridspec

def compute_all_clusters_rayleigh(self,settings,all_angles,all_conditions,base_path):
    """ 
    This function does two things:
    1. compute rayleigh for all angles in all desired conditions
    2. if Settings_analyze_efizz.multi_cluster_plots = True, it also plots all clusters per angle
    """

    for c in all_conditions:
        data_path = os.path.join(base_path, c)
        if not(os.path.exists(data_path)): 
            os.makedirs(data_path)
        
        # filter data in this condition
        filtered_video_df = filter_video_dataframe(self.video_df, c)
        
        # which compartment of the arena was the mouse in?
        # compartment 1 (in blue) is the side where the shelter is
        # compartment 2 (in purple) is the side wherethe threat zone is
        compartment = np.ones([len(filtered_video_df)])
        if len(self.session.barrier_time) > 0:
            compartment[filtered_video_df['mouse_y_position'].to_numpy()>512] = 2

        for a in all_angles:
            if np.logical_or(not os.path.isfile(data_path + "/" + str(a) +  "_Rayleigh.arrow"),settings.redo_compute):

                # extract relevant data
                this_df = filtered_video_df.select(['frames',a])
                frames = this_df['frames'].unique().to_numpy() - 1
                X = self.postprocessObject.frame_by_cluster_matrix
                X = X[frames,:]           
                
                # compute tuning
                logger.info("Calculating Rayleigh vectors for " + str(a) + " in condition: " + str(c))
                rayleigh_vector(self, settings, this_df, X, a, data_path, compartment, settings.rayleigh_bootstrap)

def compute_single_cluster_tuning(self,settings):
    """Compute rayleigh and make polar plots for all angles in all conditions for a single cluster"""

    # Initialize variables
    all_angles = identify_angles(self.session)
    all_conditions = identify_conditions(self.session)
    # all_conditions = ['shelter_only', 'barrier_pre_flip', 'barrier_post_flip']
    base_path = os.path.join(self.dir, 'Rayleigh', self.cluster_type)
    plot_save_path = os.path.join(base_path, 'single_cluster_plots')

    # Create save path if it doesn't exist
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    # check that Rayleigh has been computed and saved for all conditions and if not compute it
    compute_all_clusters_rayleigh(self,settings,all_angles, all_conditions,base_path)

    single_cluster_plots(self,settings, all_angles, all_conditions, base_path, plot_save_path)

def single_cluster_plots(self,settings, all_angles, all_conditions, base_path, plot_save_path):
    """ Make a figure for each cluster with polar plots for all angles in all conditions of interest"""

    logger.info("Making individual cluster polar plots")
    clusters = self.postprocessObject.video_spike_count_df["spike_clusters"].unique()

    # Add one index for the titles
    nrows = len(all_conditions) + 1
    ncols = len(all_angles) + 1
    
    for clu in clusters:
        if clu > 0:
            
            # Plot settings
            gs = gridspec.GridSpec(nrows, ncols, width_ratios = [1] + [3] * (ncols-1),
                                  height_ratios = [1] + [3] * (nrows-1),
                                  wspace=0, hspace=0.4)
            # gridspec sets ratios such titles are narrower than plots
            fig = plt.figure(figsize=(30, 30)) # width, height
            axs_fontsize = 23
            labels = ['Shelter compartment', 'Threat zone compartment']
            col = ['cornflowerblue','darkorchid']
            legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(col, labels)]
            fig.legend(handles=legend_elements, loc='upper right', fontsize=axs_fontsize, handlelength=4)

            # Add subtitles for each angle in first row
            for a_counter, a in enumerate(all_angles):
                ax = plt.subplot(gs[0, a_counter + 1])
                ax.text(0.5, 0.5, a, rotation='horizontal',
                        va='center', ha='center', fontsize=axs_fontsize)
                ax.set_axis_off()

            # Add subtitles for each condition in first column
            for c_counter, c in enumerate(all_conditions):
                ax = plt.subplot(gs[c_counter + 1, 0])
                ax.text(0, 0.5, c, rotation='horizontal',
                        va='center', ha='center', fontsize=axs_fontsize)
                ax.set_axis_off()

            # Create actual polar plots for each condition and angle
            for c_counter, c in enumerate(all_conditions):
                counter = ((ncols)*(c_counter+1)) + 1
                data_path = os.path.join(base_path, c)
                for a_counter, a in enumerate(all_angles):
                    counter = counter + 1
                    ax = plt.subplot(nrows, ncols, counter, projection = 'polar')
                    rayleigh_results = pl.read_ipc(data_path + "/" + str(a) +  "_Rayleigh.arrow")
                    pcentile = compute_95th_percentile_rayleigh(rayleigh_results)
                    # make actual polar plot for a given angle in a given condition
                    polar_plot(rayleigh_results.filter(rayleigh_results['clusterID'] == clu), ax, fig, pcentile=pcentile, cluster_title = False)
            # Save and close the figure
            plt.tight_layout()
            plt.savefig(str(plot_save_path) + "/cluster" + str(clu) + "_polar_plots.png")
            if settings.show_plots:
                plt.show()
            plt.close()

def compute_95th_percentile_rayleigh(rayleigh_results):
    """Compute the 95th percentile of the rayleigh distribution for each angle and condition"""
    flat_list = [item for sublist in rayleigh_results["Rayleigh"].to_list() for item in sublist] # unpack a series of lists into a single list
    return np.percentile(flat_list, 95)

def rayleigh_vector(self, settings, filtered_video_df, X, angle_filt, plot_save_path, compartment, compute_bootstrap = False):
    """A function that calculates the Rayleigh vector (amplitude and angle) for each cluster with respect to the angles given (e.g. HD or HSA)
    It only considers times when the mouse was outside the shelter
    It also performs bootstrapping by computing the rayleigh vector at random time shifts of the spikes with respect to the angles
    The Rayleigh vector is significant if the amplitude is above the 95th percentile of boostrapped amplitudes
    Rayleigh's R close to zero = untuned, fires at all head directions
    Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation"""
    
    # edges for binning firing rate at different angles
    bin_angles, bin_angle_center = generate_bin_angles(number_of_bins = 19)
            
    # Catch empty video dataframes
    if len(filtered_video_df) == 0:
        raise ValueError("Video dataframe is empty, bug.")
    
    # bin angles
    binned_angles = np.array(filtered_video_df[angle_filt].to_numpy())
    binned_angles = np.digitize(binned_angles, bin_angles) - 1

    # initialize variables to compute the Rayleigh vector
    cluster_Ids = self.postprocessObject.video_spike_count_df["spike_clusters"].unique().to_numpy()
    # Remove cluster 0 from ks only - synthetic starts at cluster id 1
    cluster_Ids = cluster_Ids[cluster_Ids > 0]
    Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist = init_rayleigh(cluster_Ids, len(np.unique(compartment)), bin_angle_center)
    
    # assign spike times of each cluster to the corresponding video frame, then assign HD
    for counter,c in enumerate(cluster_Ids):
        Rayleigh_cluster[counter] = c
        for c_count, comp in enumerate(np.unique(compartment)):

            # compute firing in angle bins
            angles_firing = np.zeros(len(bin_angles)-1)
            unique_groups, group_counts = np.unique(binned_angles[compartment == comp], return_counts=True)
            group_sums = np.bincount(binned_angles[compartment == comp], weights = X[compartment == comp,counter])
            angles_firing[unique_groups] = group_sums[unique_groups] / group_counts

            # compute rayleigh
            Rayleigh[counter,c_count], Rayleigh_theta[counter,c_count] = rayleigh(bin_angle_center[1:-1],angles_firing)
            angle_firing_hist[counter,:,c_count] = angles_firing
            
            # TODO: bootstrap x times with variable shifts in time
            # Linear shifts performed at a random offset between 0 and 100 seconds to generate a null distribution to detect non-sense correlations 
            # if compute_bootstrap:
            #     x = 100
            #     shift_dist = np.empty(x)
                    
            #     # significance logical
            #     if Rayleigh[counter] > np.percentile(shift_dist, 95):
            #         Rayleigh_sig[counter] = 1
            #         print('yay! ' + str(c) + ' is significant')

    # histogram of rayleighs
    plt.figure()
    plt.hist(Rayleigh,np.arange(0,1,.1))
    plt.hist(Rayleigh[Rayleigh_sig == 1],np.arange(0,1,.1))
    plt.xlabel('Rayleigh R')
    plt.ylabel('number of clusters')
    plt.savefig(plot_save_path + "/" + str(angle_filt) +  "_Rayleigh_vector_hist.png")
    if settings.show_plots: plt.show()
    plt.close()

    # save rayleigh info for all cluster to csv for this angle in this condition
    if len(np.shape(angle_firing_hist)) == 3: angle_firing_hist = np.reshape(angle_firing_hist,[np.shape(angle_firing_hist)[0],np.shape(angle_firing_hist)[1]*np.shape(angle_firing_hist)[2]])
    rayleigh_results = pl.DataFrame({'clusterID': Rayleigh_cluster,
                                     'Rayleigh': Rayleigh,
                                     'Rayleigh_theta': Rayleigh_theta,
                                     'Rayleigh_sig': Rayleigh_sig,
                                     'angle_firing_hist': angle_firing_hist,
                                     'angles':np.tile(bin_angle_center[1:-1],(len(Rayleigh),1))})

    rayleigh_results.write_ipc(plot_save_path + "/" + str(angle_filt) +  "_Rayleigh.arrow")

    logger.info(f"Finished calculating Rayleigh vectors, moving on to polar plots")
    if settings.multi_cluster_plots:
        folder_name = os.path.join(plot_save_path,str(angle_filt) + "_cluster_tuning_plots")
        if not(os.path.exists(folder_name)): os.makedirs(folder_name)
        all_clusters_polar_plots(rayleigh_results,folder_name,settings.show_plots)

def all_clusters_polar_plots(rayleigh_results, save_path,show_plots):
    """
    It makes a polar plot of firing at each angle (e.g. HD or HSA) for each cluster.
    self.tuning_dict['angles'] is a binned set of angles and self.tuning_dict[title][0] will give you the firing rates for cluster 0.
    Where the title is what the tunning was calculated for (e.g. 'HD' or 'HSA').
    """
    # ---------------------------------------------------
    # set up polar plots figure
    # set number of rows and calculate number of columns
    ncols = 10
    nrows = 5 # nclu // ncols + (nclu % ncols > 0)
    figg, axs = plt.subplots(nrows,ncols)
    figg.set_figwidth(30)
    figg.set_figheight(15)
    fnum = 1
    axs = axs.ravel()

    # assign spike times of each cluster to the corresponding video frame, then assign HD
    
    number_of_clusters = len(rayleigh_results)
    for counter in np.arange(number_of_clusters):
        
        # if you have filled a figure with polar plots, move onto next figure
        if counter >= (ncols*nrows)*fnum:
            figg, axs = plt.subplots(nrows,ncols)
            figg.set_figwidth(30)
            figg.set_figheight(15)
            fnum = fnum + 1
            axs = axs.ravel()
            
        ax = plt.subplot(nrows,ncols,1+counter-(nrows*ncols*(fnum-1)),projection = 'polar')
        
        # polar plots!
        polar_plot(rayleigh_results.filter(np.arange(len(rayleigh_results)) == counter),ax)
            
        # save the whole figure
        if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1, counter == number_of_clusters-1):
            plt.tight_layout()
            plt.savefig(str(save_path) + "/cluster_polar_plots_" + str(fnum) + ".png")
            if show_plots: plt.show()  
            plt.close()

## --------------------FUNCS ---------------------------------

def init_rayleigh(number_of_clusters, compartments, bin_angle_center):
    """
    Initializes the variables needed to compute the Rayleigh test
    """
    Rayleigh_theta = np.empty([len(number_of_clusters),compartments]) # preferred angle
    Rayleigh = np.empty([len(number_of_clusters),compartments]) # amplitude of Rayleigh vector
    Rayleigh_sig = np.zeros([len(number_of_clusters),compartments]) # is the Ryleigh significant?
    Rayleigh_cluster = np.empty([len(number_of_clusters)]) # which cluster ID is this Rayleigh value for?
    angle_firing_hist = np.empty([len(number_of_clusters),len(bin_angle_center)-2,compartments])
    return Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist

def rayleigh(angles,firing) -> tuple:
    """Compute the rayleigh vector for a given set of angles and firing rates"""
    x = np.sum(np.cos(angles)*(firing))/np.sum(firing)
    y = np.sum(np.sin(angles)*(firing))/np.sum(firing)
    theta = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    return r, theta

## ---------------------PLOTTING -----------------------------

def polar_plot(df, ax, fig, pcentile, cluster_title = True, plot_type = "line") -> None:
    """Creates a polar plot for a single cluster in a single condition
    
    Visulises the firing rate at each angle (e.g. HD or HSA) for different
    compartments of the arena (e.g. shelter or threat zone).
    
    Arguments:
    df -- a dataframe with the following columns: 
        clusterID, 
        Rayleigh, 
        Rayleigh_theta, 
        Rayleigh_sig, 
        angle_firing_hist, 
        angles
    counter -- the index of the cluster you want to plot
    ax -- the axis index of the subplot
    plot_type - line, bar, fill
    """
    
    # Check if the dataframe is empty
    if len(df['angle_firing_hist'].to_list()) == 0: 
        return

    # Plot settings
    col = ['cornflowerblue','darkorchid']
    angle_firing = df['angle_firing_hist'].to_list()
    
    if np.shape(angle_firing)[1] > np.shape(df['angles'].to_list())[1]:
        angle_firing = np.reshape(angle_firing,[int(np.shape(df['angles'].to_list())[1]),int(np.shape(angle_firing)[1]/np.shape(df['angles'].to_list())[1])])
        angle_firing = angle_firing.T

    for compartment in np.arange(len(df['Rayleigh'][0])):
        # Plot the rayleigh vector magnitude and angle
        ax.vlines(x=df['Rayleigh_theta'][0][int(compartment)],
                  ymin=0,
                  ymax=df['Rayleigh'][0][int(compartment)] * np.amax(angle_firing[compartment]),
                  # ymax is rayleigh vector amplitude * max firing rate
                  colors=col[compartment])

        # Plot the firing rate at each angle this is the actual polar plot code

        # Settings for the polar plot bars
        if plot_type == "bar":
            ax.bar(x = df['angles'].to_list()[0],
                height = angle_firing[compartment],
                width = (2*np.pi) / (len(angle_firing[compartment])+1),
                bottom = 0.0,
                color = col[compartment],
                alpha = 0.5)
        
        # Testing out a different way to plot the polar plot bars with area filled in
        # Plot the firing rate at each angle this is the actual polar plot code
        else: 
            angles = np.concatenate([df['angles'].to_list()[0], [df['angles'].to_list()[0][0]]])
            values = np.concatenate([angle_firing[compartment], [angle_firing[compartment][0]]])
        
            # Polar plot area with fill
            if plot_type == "fill":
                ax.fill(angles, values, color=col[compartment], alpha=0.5)
            
            # Polar plot area with no fill, just outline
            elif plot_type == "line":
                ax.plot(angles, values, color=col[compartment], linewidth=1.5)


    # Settings for the polar plot grid
    ax.grid(True,
            linestyle='--',
            linewidth=0.5,
            color='gray',
            alpha = 0.5,
            markevery=3)

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
        clutitle = 'clu' + str(int(df['clusterID'][0])) 
    else:
        clutitle = ' '
    for compartment in np.arange(len(df['Rayleigh'][0])):
        clutitle = clutitle + '\n' + 'Rayleigh = ' + str(np.around(df['Rayleigh'][0][int(compartment)],2))
        if df['Rayleigh_sig'][0][int(compartment)] == 1: 
            clutitle = clutitle + ' (sig.)'
        if df['Rayleigh'][0][int(compartment)] > pcentile:
            fig.suptitle('This cluster is worth a check', fontsize=30)
    ax.title.set_text(clutitle)

