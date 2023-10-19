# script for computin rayleigh vectors
from loguru import logger
import numpy as np
import polars as pl
import os
import matplotlib 
import matplotlib.pyplot as plt
from behave_analysis.analyze.filtering_data.filtering_functions  import filter_video_dataframe, identify_angles, generate_bin_angles, identify_conditions
import matplotlib.gridspec as gridspec

def compute_all_clusters_rayleigh(self,settings):
    """ 
    This function does two things:
    1. compute rayleigh for all angles in all desired conditions
    2. plots all clusters per angle
    """

    all_angles = identify_angles(self.session)

    base_path = os.path.join(self.dir, 'Rayleigh', self.cluster_type)

    for c in self.all_conditions:
        plot_save_path = os.path.join(base_path, c)
        if not(os.path.exists(plot_save_path)): 
            os.makedirs(plot_save_path)
        logger.info("Commence making figures of every cluster for a single tuning curve")
        for a in all_angles:
            # compute tuning 
            filtered_video_df = filter_video_dataframe(self.video_df, c)
            filtered_video_df = filtered_video_df.select(['frames',a])
            frames = filtered_video_df['frames'].unique().to_numpy() - 1
            X = self.postprocessObject.frame_by_cluster_matrix
            X = X[frames,:]           
            logger.info("Calculating Rayleigh vectors for " + str(a) + " in condition: " + str(c))
            rayleigh_vector(self, settings, filtered_video_df, X, a, plot_save_path, settings.rayleigh_bootstrap)

def compute_single_cluster_tuning(self,settings):
    """Compute rayleigh and polar plots for all angles in all conditions for a single cluster"""
    # Initialize variables
    all_angles = identify_angles(self.session)
    all_conditions = identify_conditions(self.session)
    base_path = os.path.join(self.dir, 'Rayleigh', self.cluster_type)
    plot_save_path = os.path.join(base_path, 'single_cluster_plots')

    # Create save path if it doesn't exist
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    # For each condition check if the rayleigh vector has been computed
    for c_counter, c in enumerate(all_conditions):
        data_path = os.path.join(base_path, c)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        # For each angle within a condition, compute the rayleigh vector
        for a_counter, a in enumerate(all_angles):
            if not os.path.isfile(data_path + "/" + str(a) +  "_Rayleigh.arrow"):
                filtered_video_df = filter_video_dataframe(self.video_df, c)
                filtered_video_df = filtered_video_df.select(['frames',a])
                frames = filtered_video_df['frames'].unique().to_numpy() - 1
                X = self.postprocessObject.frame_by_cluster_matrix
                X = X[frames,:]
                logger.info("Calculating Rayleigh vectors for " + str(a) + " in condition: " + str(c))
                rayleigh_vector(self, settings, filtered_video_df, X, a, data_path, settings.rayleigh_bootstrap)
    
    # TODO: Is it possible to make above this line into another function and refactor the code
    # below into a separate function that calls the above function?
    # -----------------------------------------------------------------------------------------------------

    logger.info("Making individual cluster polar plots")
    clusters = self.postprocessObject.video_spike_count_df["spike_clusters"].unique()

    # Add one index for the titles
    nrows = len(all_conditions) + 1
    ncols = len(all_angles) + 1

    for clu in clusters:
        if clu > 0:
            gs = gridspec.GridSpec(nrows, ncols, width_ratios = [1] + [3] * (ncols-1),
                                  height_ratios = [1] + [3] * (nrows-1),
                                  wspace=0, hspace=0.4)
            # gridspec sets ratios such titles are narrower than plots
            _ = plt.figure(figsize=(30, 30)) # width, height
            axs_fontsize = 23

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
                    # make actual polar plot for a given angle in a given condition
                    clucounter = np.where(rayleigh_results['clusterID'].to_numpy() == clu)[0]
                    polar_plot(rayleigh_results, clucounter[0], ax, cluster_title = False)
            # Save and close the figure
            plt.tight_layout()
            plt.savefig(str(plot_save_path) + "/cluster" + str(clu) + "_polar_plots.png")
            if settings.show_plots:
                plt.show()
            plt.close()
 
def rayleigh_vector(self, settings, filtered_video_df, X, angle_filt, plot_save_path, compute_bootstrap = False):
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
    cluster_Ids = clusterIds[clusterIds > 0]
    Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist = init_rayleigh(cluster_Ids, bin_angle_center)
    
    # assign spike times of each cluster to the corresponding video frame, then assign HD
    for counter,c in enumerate(cluster_Ids):
        
        # Check for empty cluster dataframes
        # if spike_to_video_df.select(pl.col('spike_count').is_null().sum()).item() == len(spike_to_video_df):
        #     logger.info(f"Cluster {c} had no spikes, skipping this cluster and no Rayleigh vector will be computed for it nor will it be plotted")
        #     continue
        
        # calculate firing rates in angle bins
        # make sure that if any angles returned empty sets of spikes, they are registered as zeros and are not missing
        # angles_firing = np.zeros(len(bin_angles)-1)
        # for b in np.arange(1,len(bin_angles)-1):
        #     if len(X[binned_angles == b,counter]) > 0:
        #         if not(np.sum(np.isnan(X[binned_angles == b,counter])) == len(X[binned_angles == b,counter])):
        #             angles_firing[b-1] = np.nanmean(X[binned_angles == b,counter])

        angles_firing = np.zeros(len(bin_angles)-1)
        unique_groups, group_counts = np.unique(binned_angles, return_counts=True)
        group_sums = np.bincount(binned_angles, weights = X[:,counter])
        angles_firing[unique_groups] = group_sums[unique_groups] / group_counts

        # compute rayleigh
        Rayleigh[counter], Rayleigh_theta[counter] = rayleigh(bin_angle_center[1:-1],angles_firing)
        Rayleigh_cluster[counter] = c
        angle_firing_hist[counter,:] = angles_firing
        
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
    rayleigh_results = pl.DataFrame({'clusterID': Rayleigh_cluster,
                                     'Rayleigh': Rayleigh,
                                     'Rayleigh_theta': Rayleigh_theta,
                                     'Rayleigh_sig': Rayleigh_sig,
                                     'angle_firing_hist':angle_firing_hist,
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
        polar_plot(rayleigh_results,counter,ax)
            
        # save the whole figure
        if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1, counter == number_of_clusters-1):
            plt.tight_layout()
            plt.savefig(str(save_path) + "/cluster_polar_plots_" + str(fnum) + ".png")
            if show_plots: plt.show()  
            plt.close()

## --------------------FUNCS ---------------------------------

def init_rayleigh(number_of_clusters, bin_angle_center):
    """
    Initializes the variables needed to compute the Rayleigh test
    """
    Rayleigh_theta = np.empty([len(number_of_clusters)]) # preferred angle
    Rayleigh = np.empty([len(number_of_clusters)]) # amplitude of Rayleigh vector
    Rayleigh_sig = np.zeros([len(number_of_clusters)]) # is the Ryleigh significant?
    Rayleigh_cluster = np.empty([len(number_of_clusters)]) # which cluster ID is this Rayleigh value for?
    angle_firing_hist = np.empty([len(number_of_clusters),len(bin_angle_center)-2])
    return Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist

def rayleigh(angles,firing) -> tuple:
    """Compute the rayleigh vector for a given set of angles and firing rates"""
    x = np.sum(np.cos(angles)*(firing))/np.sum(firing)
    y = np.sum(np.sin(angles)*(firing))/np.sum(firing)
    theta = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    return r, theta

## ---------------------PLOTTING -----------------------------

def polar_plot(df, counter, ax, cluster_title = True) -> None:
    """Creates a polar plot
    
    Visulises the firing rate at each angle (e.g. HD or HSA) for a single cluster
    
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
    """
    # Helper function
    def extractor(df):
        """Given a cluster id as counter(used as a table index), return a series column as a list"""
        return df.take([counter]).to_list()[0]

    # Plot the rayleigh vector magnitude and angle
    ax.vlines(x = df['Rayleigh_theta'].take([counter])[0],
              ymin = 0,
              ymax = df['Rayleigh'].take([counter])[0] * np.amax(extractor(df['angle_firing_hist'])),
              # ymax is rayleigh vector amplitude * max firing rate
              colors='black')

    # Plot the firing rate at each angle this is the actual polar plot code
    if len(extractor(df['angle_firing_hist'])) > 0:

        # Settings for the polar plot bars
        ax.bar(x = extractor(df['angles']),
               height = extractor(df['angle_firing_hist']),
               width = (2*np.pi) / (len(extractor(df['angle_firing_hist']))+1),
               bottom = 0.0,
               color = 'cornflowerblue',
               alpha = 0.5)

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
        if df['Rayleigh_sig'].take([counter])[0] == 1:
            ax.title.set_text(str(df['clusterID'].take([counter])[0]) + ' clu ' + ' (sig.)' +
                                '\n' + 'Rayleigh = ' + str(np.around(df['Rayleigh'].take([counter])[0],2)))
        else:
            ax.title.set_text(str(df['clusterID'].take([counter])[0]) + ' clu ' +
                                '\n' + 'Rayleigh = ' + str(np.around(df['Rayleigh'].take([counter])[0],2)))
    else:
        if df['Rayleigh_sig'].take([counter])[0] == 1:
            ax.title.set_text(' (sig.)' + 'Rayleigh = ' + str(np.around(df['Rayleigh'].take([counter])[0],2)))
        else:
            ax.title.set_text('Rayleigh = ' + str(np.around(df['Rayleigh'].take([counter])[0],2)))
