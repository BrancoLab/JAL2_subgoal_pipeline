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
from behave_analysis.visualize.efizz.egocentric_firing_map_binned import egocentric_firing_map
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, filter_video_dataframe
from behave_analysis.visualize.visualize_behave import hsv_hdir_colormap

class Visualize_efizz:
    """
    A class for some sanity check efizz plots using kilosort clusters
    """
    
    def __init__(self,  PreProcessed_data_object, session):
       self.processed_data = PreProcessed_data_object
       self.session = session
       self.video_df = pl.read_csv(
            os.path.join(self.session.base_path, self.session.processed_path, "full_video_dataframe.csv")
        )
       self.stim_resp_path = os.path.join(self.session.base_path,self.session.processed_path, 'stim_resp')
       if not(os.path.exists(self.stim_resp_path)): os.makedirs(self.stim_resp_path)
       self.spatial_path = os.path.join(self.session.base_path,self.session.processed_path, 'spatial_firing')
       if not(os.path.exists(self.spatial_path)): os.makedirs(self.spatial_path)
       logger.info("Visualize_efizz class initialized - Time to plot some efizz!")

    def run_tuning_functions(self):
        """Make tuning plots"""
        logger.info(f"Starting to make some efizz tuning plots...")
        self.spatial_position_firing_hdir() # when a neuron fires coloured by hdir
        # self.spatial_position_firing() # ~ BUG - RuntimeError: main thread is not in main loop
        cluster_Ids = self.processed_data.video_spike_count_df["spike_clusters"].unique().to_numpy()
        egocentric_firing_map(self.processed_data.frame_by_cluster_matrix, 
                              self.video_df,
                              self.processed_data.clu_label,
                              self.session,
                              cluster_Ids = cluster_Ids[cluster_Ids > 0])
        logger.info(f"Finished! to make some efizz tuning plots...")

    def run_stim_resp_plotting(self):
        """Make plots of stimulus response"""
        logger.info(f"Starting to make some plots of threat stimulus responses.")
        self.rasters(settings_v.stim_type)
        self.PSTH_all_neurons(settings_v.stim_type)
        self.PSTH_single_neurons(settings_v.stim_type)
        self.single_cluster_raster(settings_v.stim_type)

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

    def spatial_position_firing_hdir(self):
        """ 
        A function that plots the position of the mouse at every AP of a given cluster and colours it by hdir
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
        save_path = os.path.join(self.spatial_path, 'spatial_firing_hdir_color',self.processed_data.select_clusters)

        video_df = filter_video_dataframe(self.video_df,'all_time')
        video_df = video_df.select(['frames','hdir','mouse_x_position','mouse_y_position'])
        spike_data = self.processed_data.spike_data.select(['spike_clusters','spike_aligned_to_frame'])

        # what is firing rate per frame?
        for counter,cluster in enumerate(self.processed_data.spike_data["spike_clusters"].unique()):
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            # filter spikes by cluster
            spikes = spike_data.filter(spike_data['spike_clusters'] == cluster)
            # align spike data for this cluster to video_df
            spikes = spikes.with_column(spikes['spike_aligned_to_frame'].cast(pl.Int64))
            merged_df = video_df.join(spikes, left_on='frames', right_on = 'spike_aligned_to_frame', how='inner')

            hdir = np.digitize(np.rad2deg(merged_df['hdir']), np.arange(-180, 180))
            cc = hsv_hdir_colormap(hdir)
            
            axs[counter-(nrows*ncols*fnum)].scatter(video_df['mouse_x_position'].to_numpy(),
                                                    video_df['mouse_y_position'].to_numpy(),
                                                    s=3,color=[.7, .7, .7],linewidths=0,marker='.') # all mouse positions
            axs[counter-(nrows*ncols*fnum)].scatter(merged_df['mouse_x_position'].to_numpy(),
                                                    merged_df['mouse_y_position'].to_numpy(),
                                                    s=7,c=cc,linewidths=0,marker='.') # this neuron's firing coloured by hdir
            axs[counter-(nrows*ncols*fnum)].set_axis_off()
            axs[counter-(nrows*ncols*fnum)].invert_yaxis()
            axs[counter-(nrows*ncols*fnum)].set_aspect('equal')
            this_cluster = self.processed_data.clu_label.filter(self.processed_data.clu_label["spike_clusters"] == [cluster])
            axs[counter-(nrows*ncols*fnum)].title.set_text(str(this_cluster["cluster_group"].to_numpy()) + ' cluster ' + str(cluster))

            # save the figure
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1, counter == len(self.processed_data.spike_data["spike_clusters"].unique())-1):
                
                plt.tight_layout()
                plt.savefig(str(save_path) + "/" + self.processed_data.select_clusters + "_clusters_spatial_firing_hdir_colored" + str(fnum) + ".png")
                
                if settings_v.show_plots: 
                    plt.show()
                    
                plt.close()

    def spatial_position_firing(self):
        """ 
        A function that makes maps of mousie's position in arena and show where each cluster fired
        """
        #TODO: rewrite this using big postprocess matrix instead
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
        save_path = os.path.join(self.spatial_path, 'spatial_firing_maps',self.processed_data.select_clusters)

        large_dataFrame = self.processed_data.video_spike_count_df.select(['frames','spike_clusters','mouse_x_position','mouse_y_position','spike_count'])

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
                plt.savefig(str(save_path) + "/" + self.processed_data.select_clusters + "_clusters_spatial_position_firing_" + str(fnum) + ".png")
                
                if settings_v.show_plots: 
                    plt.show()
                    
                plt.close()

    def spatial_position_firing_old(self):
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

        large_dataFrame = self.processed_data.video_spike_count_df.select(['frames','spike_clusters','mouse_x_position','mouse_y_position','spike_count'])

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