# script for computin rayleigh vectors
from loguru import logger
import numpy as np
import polars as pl
import os
import matplotlib 
import matplotlib.pyplot as plt
from behave_analysis.analyze.filtering_data.filtering_functions  import filter_video_dataframe, identify_angles, generate_bin_angles, identify_conditions


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
        if not(os.path.exists(plot_save_path)): os.makedirs(plot_save_path)
        logger.info("Commence making figures of every cluster for a single tuning curve")
        filtered_video_df = filter_video_dataframe(self.data_df, c)
        for a in all_angles:
            # compute tuning            
            logger.info("Calculating Rayleigh vectors for " + str(a) + " in condition: " + str(c))
            rayleigh_vector(self, settings, filtered_video_df, a, plot_save_path, settings.rayleigh_bootstrap)

def compute_single_cluster_tuning(self,settings):
    """ 
    This function does two things:
    1. compute rayleigh for all angles in all desired conditions
    2. plots all tunings per cluster
    """
    all_angles = identify_angles(self.session)
    all_conditions = identify_conditions(self.session)
    base_path = os.path.join(self.dir, 'Rayleigh', self.cluster_type)
    plot_save_path = os.path.join(base_path, 'single_cluster_plots')
    if not(os.path.exists(plot_save_path)): os.makedirs(plot_save_path)

    # check that rayleigh has been computer for all conditions and angles
    for c_counter,c in enumerate(all_conditions):
        data_path = os.path.join(base_path, c)
        if not(os.path.exists(data_path)): os.makedirs(data_path)
        for a_counter, a in enumerate(all_angles):
            if not os.path.isfile(data_path + "/" + str(a) +  "_Rayleigh.arrow"):
                filtered_video_df = filter_video_dataframe(self.data_df, c)
                logger.info("Calculating Rayleigh vectors for " + str(a) + " in condition: " + str(c))
                rayleigh_vector(self, settings, filtered_video_df, a, data_path, settings.rayleigh_bootstrap)
    
    clusters = self.data_df["spike_clusters"].unique()
    nrows = len(all_conditions)+1
    ncols = len(all_angles)+1

    for clu in clusters:

        figg, _ = plt.subplots(nrows,ncols) # conditions are rows and angles are columns
        figg.set_figwidth(30)
        figg.set_figheight(30)

        # clear the corner
        ax = plt.subplot(nrows,ncols,1,projection = 'polar')
        ax.set_axis_off()

        # add text to indicate which angles are in each column 
        for a_counter, a in enumerate(all_angles):
            ax = plt.subplot(nrows,ncols,a_counter+2,projection = 'polar')
            ax.text(0, 0, a, rotation='horizontal', va='center', ha='center')
            ax.set_axis_off()

        # add text to indicate which condition is in each row
        for c_counter,c in enumerate(all_conditions):
            counter = ((ncols)*(c_counter+1))+1
            ax = plt.subplot(nrows,ncols,counter,projection = 'polar')
            ax.text(0, 0, c, rotation='vertical', va='center', ha='center')
            ax.set_axis_off()

        for c_counter,c in enumerate(all_conditions):
            counter = ((ncols)*(c_counter+1))+1
            data_path = os.path.join(base_path, c)
            for a_counter, a in enumerate(all_angles):
                counter = counter+1
                ax = plt.subplot(nrows,ncols,counter,projection = 'polar')
                rayleigh_results = pl.read_ipc(data_path + "/" + str(a) +  "_Rayleigh.arrow")
                # make actual polar plot for a given angle in a given condition
                polar_plot(rayleigh_results,clu,ax, cluster_title = False)
        plt.tight_layout()
        plt.savefig(str(plot_save_path) + "/cluster" + str(clu) + "_polar_plots.png")
        if settings.show_plots: plt.show() 
        plt.close()
 
def rayleigh_vector(self, settings, filtered_video_df, angle_filt, plot_save_path, compute_bootstrap = False):
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

    # initialize variables to compute the Rayleigh vector
    number_of_clusters = self.data_df["spike_clusters"].unique()
    Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist = init_rayleigh(number_of_clusters, bin_angle_center)
    
    # assign spike times of each cluster to the corresponding video frame, then assign HD
    for counter,c in enumerate(number_of_clusters):
                    
        # filter by cluster
        spike_to_video_df = filtered_video_df.filter(filtered_video_df['spike_clusters'] == c)
        
        # Convert spike count to firing rate
        spike_to_video_df = spike_to_video_df.with_columns(pl.col('spike_count')*self.session.video.fps)
        
        # Check for empoty cluster dataframes
        if spike_to_video_df.select(pl.col('spike_count').is_null().sum()).item() == len(spike_to_video_df):
            logger.info(f"Cluster {c} had no spikes, skipping this cluster and no Rayleigh vector will be computed for it nor will it be plotted")
            continue

        # calculate firing rates in angle bins
        spike_to_video_df = spike_to_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
        spike_to_video_df = spike_to_video_df.with_columns(spike_to_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
        spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
        spike_to_video_df = spike_to_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 
        angles_firing = (spike_to_video_df.groupby(by = 'binned_angles').agg(pl.col('spike_count').mean().alias('mean_firing_rate')))            
        angles_firing = angles_firing.sort('binned_angles')
        
        # make sure that if any angles returned empty sets of spikes, they are registered as zeros and are not missing
        all_angles_firing = pl.DataFrame({'all_angles': bin_angle_center[1:-1]})
        all_angles_firing = all_angles_firing.join(angles_firing, left_on="all_angles", right_on="binned_angles", how="left")
        all_angles_firing = all_angles_firing.fill_null(strategy="zero")
                    
        # compute rayleigh
        Rayleigh[counter], Rayleigh_theta[counter] = rayleigh(all_angles_firing['all_angles'].to_numpy(),all_angles_firing['mean_firing_rate'].to_numpy())
        Rayleigh_cluster[counter] = c
        angle_firing_hist[counter,:] = all_angles_firing['mean_firing_rate'].to_numpy()
        
        # bootstrap x times with variable shifts in time
        # Linear shifts performed at a random offset between 0 and 100 seconds to generate a null distribution to detect non-sense correlations 
        if compute_bootstrap:
            x = 100
            shift_dist = np.empty(x)
            for it in np.arange(len(shift_dist)): 
                
                # shuffled linear shifts performed at a random offset between 0 and 100 seconds
                shift = int(np.random.uniform(1, 100)) * self.session.video.fps # temporal shift in video frames
                angles = filtered_video_df[angle_filt].to_numpy()
                ang_roll = np.roll(angles, shift)
                rolled_filtered_video_df = filtered_video_df.select(pl.col('*'),pl.Series(name="rolled_angles", values = ang_roll))
                spike_to_video_df = rolled_filtered_video_df
                
                # calculate firing rates in angle bins
                # TODO - Update variable names to be more descriptive rather than just spike_to_video_df
                spike_to_video_df = spike_to_video_df.sort('rolled_angles') # polars can be annoying, when using cut it doesn't preserve order :/
                spike_to_video_df = spike_to_video_df.with_columns(spike_to_video_df['rolled_angles'].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
                spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
                spike_to_video_df = spike_to_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')])
                angles_firing = (spike_to_video_df.groupby(by ='binned_angles').agg(pl.col('spike_count').mean().alias('mean_firing_rate')))            
                angles_firing = angles_firing.sort('binned_angles')
                
                # make sure that if any angles returned empty sets of spikes, they are registered as zeros and are not missing
                all_angles_firing = pl.DataFrame({'all_angles': bin_angle_center[1:-1]})
                all_angles_firing = all_angles_firing.join(angles_firing, left_on="all_angles", right_on="binned_angles", how="left")
                all_angles_firing = all_angles_firing.fill_null(strategy="zero")
                
                # compute rayleigh
                shift_dist[it], _ = rayleigh(all_angles_firing['all_angles'].to_numpy(),all_angles_firing['mean_firing_rate'].to_numpy())

            # significance logical
            if Rayleigh[counter] > np.percentile(shift_dist, 95):
                Rayleigh_sig[counter] = 1
                print('yay! ' + str(c) + ' is significant')

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

##--------------------FUNCS

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
    
def rayleigh(angles,firing):
    x = np.sum(np.cos(angles)*(firing))/np.sum(firing)
    y = np.sum(np.sin(angles)*(firing))/np.sum(firing)
    theta = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    return r, theta

def polar_plot(df,counter,ax,cluster_title = True):
    
    ax.vlines(df['Rayleigh_theta'].take([counter])[0],
              0, 
              df['Rayleigh'].take([counter])[0]*np.amax(df['angle_firing_hist'].take([counter]).to_list()[0]), colors='black')

    if len(df['angle_firing_hist'].take([counter]).to_list()[0])>0:
            ax.bar(df['angles'].take([counter]).to_list()[0], 
                df['angle_firing_hist'].take([counter]).to_list()[0], 
                width=(2*np.pi)/(len(df['angle_firing_hist'].take([counter]).to_list()[0])+1), 
                bottom=0.0, 
                color='green', 
                alpha=0.5)
    
    # add title to the subplot
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
