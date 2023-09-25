# OS libaries
from loguru import logger
import numpy as np
import polars as pl
import os
import matplotlib 
import matplotlib.pyplot as plt
# matplotlib.use('Agg') # Removing agg as it doesnt allow for gui
import time

# Import custom settings
from settings.settings_visualize import defined_settings_visualize as settings_v

class Visualize_efizz:
    """
    A class for some sanity check efizz plots using kilosort clusters
    """
    
    def __init__(self,  PreProcessed_data_object, session):
       self.processed_data = PreProcessed_data_object
       self.session = session
       self.stim_resp_path = os.path.join(self.session.base_path,self.session.processed_path, 'stim_resp')
       if not(os.path.exists(self.stim_resp_path)): os.makedirs(self.stim_resp_path)
       self.spatial_path = os.path.join(self.session.base_path,self.session.processed_path, 'spatial_firing')
       if not(os.path.exists(self.spatial_path)): os.makedirs(self.spatial_path)
       logger.info("Visualize_efizz class initialized - Time to plot some efizz!")

# FUNCTIONS FOR PLOTTING STIM-TRIGGERED RESPONSE --------------------------------------------------------------------------------------------------------------------------------------

    def PSTH_all_neurons(self, stim_type):
        """
        Plot the mean firing rate of all cells to each trial. For each trial, retrieve:
        - the onset frame of that stimulus
        - the duration of that stimulus
        """
        
        # Hyperparameters
        timeBeforeStim = 5 # seconds
        stimulus_durations = np.amax(self.session.__dict__[stim_type].stimulus_durations)

        # plot a line of mean activity for each trial
        for trial_num, onset_frames in enumerate(self.session.__dict__[stim_type].onset_frames):
            time1 = (onset_frames / self.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.session.video.fps) + stimulus_durations
            
            # Mask spikes that are within the time window
            spikes_trial = self.processed_data.spike_data.filter((self.processed_data.spike_data['aligned_spike_times'] > time1) & (self.processed_data.spike_data['aligned_spike_times'] < time2))
            
            # Bin the spikes
            mult = 10 # binsize for looking at data - 1/10 of a second so 100ms bins 
            binEdges = np.arange(time1, time2, 1 / mult)
            firingrate, _ = np.histogram(spikes_trial['aligned_spike_times'].to_numpy(), binEdges)
            assert len(firingrate) == len(binEdges) - 1, "firingrate and binedges are not the same length"
            
            # Generate x values for plotting
            xValues = binEdges - time1 - timeBeforeStim
            assert xValues[0] == -timeBeforeStim, f"xValues[0] is not -{timeBeforeStim}"
            
            # Plot the PSTH
            # plt.plot(xValues[:-1], gaussian_filter1d(firingrate * mult, sigma = 1), label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second
            plt.plot(xValues[:-1], firingrate * mult, label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second

            plt.axvline(x = 0, color = 'k', linestyle = '-')
            plt.ylabel('Firing rate for all cells (Hz)')
            plt.xlabel('time (s)')
            plt.legend()
        
        plt.title('Trial by trial response PSTH for stimulus type: ' + stim_type)
        plt.savefig(str(self.stim_resp_path) + "/" + self.processed_data.select_clusters + "_clusters_PSTH_all_neurons_" + str(stim_type) + ".png")
        
        if settings_v.show_plots: 
            plt.show()
            
        plt.close()

    def PSTH_single_neurons(self, stim_type):
        """
        Plot the mean firing rate of each cluster averaged across all trials.
        """
        
        timeBeforeStim = 5
        stimulus_durations = np.amax(self.session.__dict__[stim_type].stimulus_durations) + 2
        xlim = [timeBeforeStim * -1,stimulus_durations]

        # Mask spikes that are within the time window
        for trial, onset_frames in enumerate(self.session.__dict__[stim_type].onset_frames):
            time1 = (onset_frames / self.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.session.video.fps) + stimulus_durations
            filt = self.processed_data.spike_data.filter(
                        (self.processed_data.spike_data['aligned_spike_times'] > time1) &
                        (self.processed_data.spike_data['aligned_spike_times'] < time2)
            )
            filt = filt.select([pl.col('aligned_spike_times').apply(lambda x: x -(onset_frames/self.session.video.fps)),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
            if trial == 0: spikes_trial = filt
            else: spikes_trial =spikes_trial.vstack(filt)      

        # How many plots do we need?
        number_of_clusters = self.processed_data.spike_data["spike_clusters"].unique()
        number_of_plots = len(number_of_clusters)
        max_plots_per_figure = 20
        
        # How many columns and rows should the plot have
        num_cols = int(np.ceil(np.sqrt(max_plots_per_figure)))
        num_rows = int(np.ceil(max_plots_per_figure / num_cols))
        
        # Across how many figures
        num_figures = int(np.ceil(number_of_plots / max_plots_per_figure))
        
        # Create the figures
        plot_counter = 0

        # firing rate binning
        mult = 10 # binsize for looking at data - 1/10 of a second so 100ms bins 
        binEdges = np.arange(xlim[0], xlim[1], 1 / mult)
        xValues = binEdges
        
        # For each figure
        for figure_idx in range(num_figures):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 8))
            
            # For each plot
            for rows in range(num_rows):
                for columns in range(num_cols):
                    if plot_counter < number_of_plots:
                        cluster = number_of_clusters[plot_counter]
                        spikes_trial_cluster = spikes_trial.filter(spikes_trial['spike_clusters'] == cluster)
                        firingrate, _ = np.histogram(spikes_trial_cluster['aligned_spike_times'].to_numpy(), binEdges)
                        # Plot the PSTH
                        # plt.plot(xValues[:-1], gaussian_filter1d(firingrate * mult, sigma = 1), label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second
                        axes[rows, columns].plot(xValues[:-1], firingrate * mult)
                        axes[rows, columns].set_title(f"Cluster: {cluster}")
                        axes[rows, columns].vlines(0, 0, np.amax(firingrate * mult), colors='r', linestyles='solid')
                        axes[rows, columns].set_xlim(xlim)
                        axes[rows, columns].set_ylabel('Firing rate for all cells (Hz)')
                        axes[rows, columns].set_xlabel('time (s)')
                    
                    # Remove the extra axes if there are no more plots
                    else:
                        fig.delaxes(axes[rows, columns])
                    
                    plot_counter += 1
            
            # SAVE FIGURE
            fig.tight_layout()
            plt.savefig(str(self.stim_resp_path) + "/" + str(stim_type) + "_single_" + self.processed_data.select_clusters + "_cluster_PSTH_" + str(figure_idx) + ".png")                
        
        if settings_v.show_plots: 
            plt.show()

    def rasters(self, stim_type):
        """
        A function that extracts spike times and aligns it to trials as a raster plot
        """
        
        # make a raster plot for each trial
        ntrial = len(self.session.__dict__[stim_type].onset_frames)
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)

        # set number of rows and calculate number of columns
        nrows = 3
        ncols = ntrial // nrows + (ntrial % nrows > 0)
        timeBeforeStim = 5 # in seconds
        all_stimulus_durations = np.amax(self.session.__dict__[stim_type].stimulus_durations)+2

        for trial_num, (onset_frames, stim_duration) in enumerate(zip(self.session.__dict__[stim_type].onset_frames, self.session.__dict__[stim_type].stimulus_durations)):
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            time1 = (onset_frames/self.session.video.fps) - timeBeforeStim
            time2 = (onset_frames/self.session.video.fps) + all_stimulus_durations
            spikes_trial = self.processed_data.spike_data.filter((self.processed_data.spike_data['aligned_spike_times'] > time1) & (self.processed_data.spike_data['aligned_spike_times'] < time2))
            ax.scatter(spikes_trial['aligned_spike_times'].to_numpy()-(onset_frames/self.session.video.fps),
                       spikes_trial['spike_clusters'].to_numpy(),
                       marker='|', s=5, c='k')
            ax.plot([0,0],[0, np.amax(spikes_trial['spike_clusters'].to_numpy())],'r-')
            ax.plot([stim_duration,stim_duration],[0, np.amax(spikes_trial['spike_clusters'].to_numpy())],'r-')
            ax.set_ylabel('clusters')
            ax.set_xlabel('time from stim (s)')
        plt.savefig(str(self.stim_resp_path) + "/" + self.processed_data.select_clusters + "_cluster_raster_trial_" + str(stim_type) + ".png")
        
        if settings_v.show_plots: 
            plt.show()
        
        plt.close()

    def single_cluster_raster(self, stim_type):
        """
        A function that extracts spike times for each cluster and aligns it to trials as a raster plot
        """
        
        timeBeforeStim = 5
        stimulus_durations = np.amax(self.session.__dict__[stim_type].stimulus_durations) + 2
        xlim = [timeBeforeStim * -1,stimulus_durations]

        # Mask spikes that are within the time window
        for trial, onset_frames in enumerate(self.session.__dict__[stim_type].onset_frames):
            time1 = (onset_frames / self.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.session.video.fps) + stimulus_durations
            filt = self.processed_data.spike_data.filter((self.processed_data.spike_data['aligned_spike_times'] > time1) & (self.processed_data.spike_data['aligned_spike_times'] < time2))
            filt = filt.select([pl.col('aligned_spike_times').apply(lambda x: x -(onset_frames/self.session.video.fps)),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
            if trial == 0: spikes_trial = filt
            else: spikes_trial = spikes_trial.vstack(filt)      

        # How many plots do we need?
        number_of_clusters = self.processed_data.spike_data["spike_clusters"].unique()
        number_of_plots = len(number_of_clusters)
        max_plots_per_figure = 20
        
        # How many columns and rows should the plot have
        num_cols = int(np.ceil(np.sqrt(max_plots_per_figure)))
        num_rows = int(np.ceil(max_plots_per_figure / num_cols))
        
        # Across how many figures
        num_figures = int(np.ceil(number_of_plots / max_plots_per_figure))
        
        # Create the figures
        plot_counter = 0
        
        # For each figure
        for figure_idx in range(num_figures):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 8))
            
            # For each plot
            for rows in range(num_rows):
                for columns in range(num_cols):
                    if plot_counter < number_of_plots:
                        cluster = number_of_clusters[plot_counter]
                        spikes_trial_cluster = spikes_trial.filter(spikes_trial['spike_clusters'] == cluster)
                        axes[rows, columns].scatter(spikes_trial_cluster['aligned_spike_times'].to_numpy(),
                                                    spikes_trial_cluster['trial'].to_numpy(),
                                                    marker='|', s=10, c='k')
                        axes[rows, columns].set_title(f"Cluster: {cluster}")
                        axes[rows, columns].vlines(0, 1, len(self.session.__dict__[stim_type].onset_frames), colors='r', linestyles='solid')
                        axes[rows, columns].set_xlim(xlim)
                    
                    # Remove the extra axes if there are no more plots
                    else:
                        fig.delaxes(axes[rows, columns])
                    
                    plot_counter += 1
            
            # SAVE FIGURE
            fig.tight_layout()
            plt.savefig(str(self.stim_resp_path) + "/" + self.processed_data.select_clusters + "_clusters_" + str(stim_type) + "_single_cluster_raster_" + str(figure_idx) + ".png")                
        
        if settings_v.show_plots: 
            plt.show()

# FUNCTIONS FOR PLOTTING TUNING ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def spatial_position_firing(self):
        """ 
        A function that makes maps of mousie's position in arena and show where each cluster fired
        """

        logger.info("Commence making figures of spatial position firing plots of all clusters")
        cc = matplotlib.cm.Reds # could use Reds or copper
        # set number of rows and calculate number of columns
        ncols = 10
        nrows = 5 # nclu // ncols + (nclu % ncols > 0)
        figg, axs = plt.subplots(nrows,ncols)
        figg.set_figwidth(30)
        figg.set_figheight(15)
        fnum = 1
        axs = axs.ravel()

        video_df = self.processed_data.video_df.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
        large_dataFrame = video_df.join(self.processed_data.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        large_dataFrame = large_dataFrame.select(['frames','spike_clusters','mouse_x_position','mouse_y_position','spike_count'])

        # what is firing rate per frame?
        for counter,cluster in enumerate(self.processed_data.spike_data["spike_clusters"].unique()):
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            # filter spikes by cluster
            spikes = large_dataFrame.filter(large_dataFrame['spike_clusters'] == cluster)
            spikes = spikes.fill_null(strategy="zero")
            
            axs[counter-(nrows*ncols*fnum)].scatter(spikes['mouse_x_position'].to_numpy(),
                                                    spikes['mouse_y_position'].to_numpy(),
                                                    s=5,c=cc(spikes['spike_count'].to_numpy()*50),linewidths=0,marker='.') # srate*2 increase contrast
            axs[counter-(nrows*ncols*fnum)].set_axis_off()
            axs[counter-(nrows*ncols*fnum)].invert_yaxis()
            axs[counter-(nrows*ncols*fnum)].set_aspect('equal')
            this_cluster = self.processed_data.clu_label.filter(self.processed_data.clu_label["spike_clusters"] == [cluster])
            axs[counter-(nrows*ncols*fnum)].title.set_text(str(this_cluster["cluster_group"].to_numpy()) + ' cluster ' + str(cluster))

            # save the figure
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1, counter == len(self.processed_data.spike_data["spike_clusters"].unique())-1):
                plt.tight_layout()
                plt.savefig(str(self.spatial_path) + "/" + self.processed_data.select_clusters + "_clusters_spatial_position_firing_" + str(fnum) + ".png")
                
                if settings_v.show_plots: 
                    plt.show()
                    
                plt.close()
    