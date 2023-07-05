# OS libaries
from behave_analysis.database.synthetic_data.synthetic_main import synthetic_dataframe
from loguru import logger
import numpy as np
from glob import glob
import polars as pl
import os
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from loguru import logger
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

class PreProcess:
    """
    A class that loads the csv of aligned data and processes it into a dataframe that can be used for visualisation
    """
    def __init__(self,  visualize_object, run = "Production", select_clusters = "good", user_wants_to_regenerate_spike_by_frame_count = False):
        logger.info("Preprocessing started")
        self.Visualize = visualize_object
        self.select_clusters = select_clusters
        if run == "Test": self.select_clusters = "synthetic"
        self.run_type = run
        
        self.load_spike_data()
        self.filter_spike_data()
        self.track_to_polars()
        self.spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames(user_wants_to_regenerate_spike_by_frame_count)
        self.clean_behavioural_data = self.behaviourally_pure_tracking_data()
        
    def load_spike_data(self):
        """
        Loads the csv of aligned data
        """
        if self.run_type == "Production":
            self.csv_path = glob(os.path.join(self.Visualize.session.processed_path, "Processed_efizz_data"))[0]
        
        elif self.run_type == "Test":
            self.csv_path = os.path.join(self.Visualize.session.processed_path, "synthetic_efizz_data.csv")
            if not os.path.exists(self.csv_path):
                logger.warning("Synethic spike data doesn't exist and will now be generated")
                tuning = ['hdir']
                if len(self.Visualize.session.shelter_time) > 0: tuning.append('hsa')
                if len(self.Visualize.session.barrier_time) > 0: tuning.append('h_bar_north_a','h_bar_south_a')
                synth_df = synthetic_dataframe(tuning)
                synth_df.write_csv(self.csv_path)
            else:
                logger.info("Synethic spike data is being used when visualizing efizz - Real positional data is used from databank")
    
        else: 
            raise ValueError("Run type not recognised")

    def filter_spike_data(self):
        """
        Filter the spike data to only include good neurons or good + MUA
        """
        # NOTE - This will break if user says yes to both mua and good - too tired to fix 
        
        dataFrame = pl.read_csv(self.csv_path)
        
        if self.run_type == "Production":
            if self.select_clusters == 'all':
                self.spikedataframe = dataFrame.filter((dataFrame['cluster_group'] == "good")
                                                    | (dataFrame['cluster_group'] == "mua"))
                logger.info("Loaded good and multi unit clusters")
            else:
                self.spikedataframe = dataFrame.filter(dataFrame['cluster_group'] == self.select_clusters)
                numneurons = len(self.spikedataframe['spike_clusters'].unique())
                logger.info(f"Loaded {numneurons} {self.select_clusters} clusters")
        elif self.run_type == "Test":
            self.spikedataframe = dataFrame
            logger.info("Loaded all clusters")
        
        self.clu_label = self.spikedataframe.groupby(["spike_clusters"]).first()
        self.clu_label = self.clu_label.drop(["spike_aligned_to_frame", "spike_times", "aligned_spike_times", "aligned_spike_times_in_samples"])

    def track_to_polars(self):
        """
        Adds all the behavioral variables from track to the polars sike dataframe
        """
        OutofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0],
            self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]),
            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1],
            self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1])))
         # is there a time with shelter only?
        if len(self.Visualize.session.shelter_time) > 0:
            if not(np.logical_and(self.Visualize.session.shelter_time[0] == 0, self.Visualize.session.shelter_time[1] == -1)):
                if self.Visualize.session.shelter_time[1] == -1: # shelter only until the end of the session
                    shelteronly = np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps)
                else:
                    shelteronly = np.logical_and(np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps),
                                                np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) < (self.Visualize.sheltertime[1]*self.Visualize.session.video.fps))
            else:
                shelteronly = np.zeros(len(OutofShelterIdx)) == 0
                print('shelter always present')
        else:
            shelteronly = np.zeros(len(OutofShelterIdx)) == 0
            print('no shelter in this session')
         # what period in the recording was there a barrier?
        if len(self.Visualize.session.barrier_time) > 0:
            if self.Visualize.session.barrier_time[1] == -1: # shelter only until the end of the session
                barrier_present = np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.barriertime[0]*self.Visualize.session.video.fps)
            else:
                barrier_present = np.logical_and(np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.barriertime[0]*self.Visualize.session.video.fps),
                                             np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) < (self.Visualize.barriertime[1]*self.Visualize.session.video.fps))
        else:
            barrier_present = np.zeros(len(OutofShelterIdx)) == 1
            print('no barrier in this session')
        # find the escape periods
        EscapePeriod = np.zeros_like(OutofShelterIdx)
        for onsets in self.Visualize.session.audio.onset_frames:
            EscapePeriod[(onsets[0]-self.Visualize.session.video.fps):(onsets[0]+(10*self.Visualize.session.video.fps))] = 1
        # make a video dataframe where for each video frame:
        self.Video_df = pl.DataFrame(
                {"frames": np.arange(1,len(self.Visualize.tracking_data['hdir'])+1).astype(np.int64),
                "hdir": self.Visualize.tracking_data['hdir'],
                "hsa": self.Visualize.tracking_data['hdir_shelt'],
                "h_bar_north_a": self.Visualize.tracking_data['hdir_barrier'][:,0],
                "h_bar_south_a": self.Visualize.tracking_data['hdir_barrier'][:,1],
                "mouse_x_position": self.Visualize.tracking_data['avg_loc'][:,0],
                "mouse_y_position": self.Visualize.tracking_data['avg_loc'][:,1],
                "OutofshelterIdx": OutofShelterIdx, # was the mouse in the shelter?
                "EscapePeriod": EscapePeriod == 1, # frames from 1 second before to 10 seconds after escape
                "shelter_only": shelteronly, # was this in a shelter only period? or was there a barrier?
                "barrier_present": barrier_present,}) # was this in a barrier period? or was there a barrier?

    def behaviourally_pure_tracking_data(self):
        """
        Filter out all the data where the mouse is in the shelter for example
        """
        filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) 
                                                 & (self.Video_df["EscapePeriod"] == False))        
 
        assert len(filtered_video_df) > 0, "No data left after filtering"
        
        return filtered_video_df

    def count_spikes_and_units_to_frames(self, user_wants_to_regenerate_spike_by_frame_count = False):
        """
        Testing the query format of polars. In theory by using the query formation we can speed up the computation of the spike count outside of a loop
        by using the lazy() function, which means that computations are not immediately executed. This allows the computer to plan the operations before
        proceeding. Additionally the computational power is not linear, and thread operations are at play. This means outside of a loop should be faster. 
        """
        
        # NOTE - THis will create an arror if the filteer on the cells changes e.g good vs mua as the dataframe will not update
        # TODO - Fix this
        
        if user_wants_to_regenerate_spike_by_frame_count == False:
            try:
                with open(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_clusters +"cluster.csv", "rb") as file:
                    spikecountbyframe_neuron = pl.read_csv(file.read())
                logger.success("Found spike count by frame and cluster dataframe, loading it now")
                return spikecountbyframe_neuron
                    
            except FileNotFoundError:
                logger.info("Could not find spike count by frame and cluster dataframe, creating it now")
                logger.info("Commencing long computation to count spikes for each cluster for each frame")
                query = (self.spikedataframe.lazy().groupby(["spike_aligned_to_frame", "spike_clusters"]).agg([pl.count("spike_aligned_to_frame").alias("spike_count")])) # Lazy query to plan computation
                start_time = time.time() # Collect lazy query and time it for user as this is the longest computation in the pipeline
                spikecountbyframe_neuron = query.collect()
                print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
                spikecountbyframe_neuron.write_csv(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_clusters +"cluster.csv")
                return spikecountbyframe_neuron
        
        elif user_wants_to_regenerate_spike_by_frame_count == True:
            logger.info("recreating the spike count by frame and unit dataframe as requested by user, likely because of changing the filter on cluster type, creating it now")
            logger.info("Commencing long computation to count spikes for each cluster for each frame")
            query = (self.spikedataframe.lazy().groupby(["spike_aligned_to_frame", "spike_clusters"]).agg([pl.count("spike_aligned_to_frame").alias("spike_count")])) # Lazy query to plan computation
            start_time = time.time() # Collect lazy query and time it for user as this is the longest computation in the pipeline
            spikecountbyframe_neuron = query.collect()
            print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
            spikecountbyframe_neuron.write_csv(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_clusters +"cluster.csv")
            return spikecountbyframe_neuron
        
class Visualize_efizz:
    """
    A class for some sanity check efizz plots using kilosort clusters
    """
    def __init__(self,  PreProcessed_data_object):
       logger.info("Visualize_efizz class initialized - Time to plot some efizz!")
       self.processed_data = PreProcessed_data_object

# FUNCTIONS FOR PLOTTING STIM-TRIGGERED RESPONSE --------------------------------------------------------------------------------------------------------------------------------------

    def PSTH_all_neurons(self, stim_type):
        """
        Plot the mean firing rate of all cells to each trial. For each trial, retrieve:
        - the onset frame of that stimulus
        - the duration of that stimulus
        """
        # Hyperparameters
        timeBeforeStim = 5 # seconds
        stimulus_durations = np.amax(self.processed_data.Visualize.session.__dict__[stim_type].stimulus_durations)

        # plot a line of mean activity for each trial
        for trial_num, onset_frames in enumerate(self.processed_data.Visualize.session.__dict__[stim_type].onset_frames):
            time1 = (onset_frames / self.processed_data.Visualize.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.processed_data.Visualize.session.video.fps) + stimulus_durations
            
            # Mask spikes that are within the time window
            spikes_trial = self.processed_data.spikedataframe.filter((self.processed_data.spikedataframe['aligned_spike_times'] > time1) & (self.processed_data.spikedataframe['aligned_spike_times'] < time2))
            
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
        plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/PSTH_all_neurons_" + str(stim_type) + ".png")
        if self.processed_data.Visualize.settings.show_plots: plt.show()
        plt.close()

    def PSTH_single_neurons(self, stim_type):
        """
        Plot the mean firing rate of each cluster averaged across all trials.
        """
        timeBeforeStim = 5
        stimulus_durations = np.amax(self.processed_data.Visualize.session.__dict__[stim_type].stimulus_durations) + 2
        xlim = [timeBeforeStim * -1,stimulus_durations]

        # Mask spikes that are within the time window
        for trial, onset_frames in enumerate(self.processed_data.Visualize.session.__dict__[stim_type].onset_frames):
            time1 = (onset_frames / self.processed_data.Visualize.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.processed_data.Visualize.session.video.fps) + stimulus_durations
            filt = self.processed_data.spikedataframe.filter((self.processed_data.spikedataframe['aligned_spike_times'] > time1)
                                            & (self.processed_data.spikedataframe['aligned_spike_times'] < time2))
            filt = filt.select([pl.col('aligned_spike_times').apply(lambda x: x -(onset_frames/self.processed_data.Visualize.session.video.fps)),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
            if trial == 0: spikes_trial = filt
            else: spikes_trial =spikes_trial.vstack(filt)      

        # How many plots do we need?
        number_of_clusters = self.processed_data.spikedataframe["spike_clusters"].unique()
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
            plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + str(stim_type) + "_single_cluster_PSTH_" + str(figure_idx) + ".png")                
        
        if self.processed_data.Visualize.settings.show_plots: 
            plt.show()

    def rasters(self, stim_type):
        """
        A function that extracts spike times and aligns it to trials as a raster plot
        """
        # make a raster plot for each trial
        ntrial = len(self.processed_data.Visualize.session.__dict__[stim_type].onset_frames)
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)

        # set number of rows and calculate number of columns
        nrows = 3
        ncols = ntrial // nrows + (ntrial % nrows > 0)
        timeBeforeStim = 5 # in seconds
        all_stimulus_durations = np.amax(self.processed_data.Visualize.session.__dict__[stim_type].stimulus_durations)+2

        for trial_num, (onset_frames, stim_duration) in enumerate(zip(self.processed_data.Visualize.session.__dict__[stim_type].onset_frames, self.processed_data.Visualize.session.__dict__[stim_type].stimulus_durations)):
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            time1 = (onset_frames/self.processed_data.Visualize.session.video.fps) - timeBeforeStim
            time2 = (onset_frames/self.processed_data.Visualize.session.video.fps) + all_stimulus_durations
            spikes_trial = self.processed_data.spikedataframe.filter((self.processed_data.spikedataframe['aligned_spike_times'] > time1) & (self.processed_data.spikedataframe['aligned_spike_times'] < time2))
            ax.scatter(spikes_trial['aligned_spike_times'].to_numpy()-(onset_frames/self.processed_data.Visualize.session.video.fps),
                       spikes_trial['spike_clusters'].to_numpy(),
                       marker='|', s=5, c='k')
            ax.plot([0,0],[0, np.amax(spikes_trial['spike_clusters'].to_numpy())],'r-')
            ax.plot([stim_duration,stim_duration],[0, np.amax(spikes_trial['spike_clusters'].to_numpy())],'r-')
            ax.set_ylabel('clusters')
            ax.set_xlabel('time from stim (s)')
        plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + "all_cluster_raster_trial_" + str(stim_type) + ".png")
        if self.processed_data.Visualize.settings.show_plots: plt.show()
        plt.close()

    def single_cluster_raster(self, stim_type):
        """
        A function that extracts spike times for each cluster and aligns it to trials as a raster plot
        """
        timeBeforeStim = 5
        stimulus_durations = np.amax(self.processed_data.Visualize.session.__dict__[stim_type].stimulus_durations) + 2
        xlim = [timeBeforeStim * -1,stimulus_durations]

        # Mask spikes that are within the time window
        for trial, onset_frames in enumerate(self.processed_data.Visualize.session.__dict__[stim_type].onset_frames):
            time1 = (onset_frames / self.processed_data.Visualize.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.processed_data.Visualize.session.video.fps) + stimulus_durations
            filt = self.processed_data.spikedataframe.filter((self.processed_data.spikedataframe['aligned_spike_times'] > time1) & (self.processed_data.spikedataframe['aligned_spike_times'] < time2))
            filt = filt.select([pl.col('aligned_spike_times').apply(lambda x: x -(onset_frames/self.processed_data.Visualize.session.video.fps)),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
            if trial == 0: spikes_trial = filt
            else: spikes_trial = spikes_trial.vstack(filt)      

        # How many plots do we need?
        number_of_clusters = self.processed_data.spikedataframe["spike_clusters"].unique()
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
                        axes[rows, columns].vlines(0, 1, len(self.processed_data.Visualize.session.__dict__[stim_type].onset_frames), colors='r', linestyles='solid')
                        axes[rows, columns].set_xlim(xlim)
                    
                    # Remove the extra axes if there are no more plots
                    else:
                        fig.delaxes(axes[rows, columns])
                    
                    plot_counter += 1
            
            # SAVE FIGURE
            fig.tight_layout()
            plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + str(stim_type) + "_single_cluster_raster_" + str(figure_idx) + ".png")                
        
        if self.processed_data.Visualize.settings.show_plots: 
            plt.show()

# FUNCTIONS FOR PLOTTING TUNING ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def compute_all_tunings_for_each_cell(self, compute_bootstrap = False):
        """
        A function that computes every single tuning curve for each cell
        """
        logger.info("Commence making figures of each individual cluster and all its respective tuning plots")
        run = {'hdir': True,'hsa':True, 'pre_hsa': True, 'h_bar_south_a': True, 'pre_h_bar_south_a': True, 'h_bar_north_a': True, 'pre_h_bar_north_a': True}
        
        # If the tuning dictionary exsits, then run the keys of the above dtionary, and if the key is int the tuning directory, then set it to false
        # What?
        if self.tuning_dict:
            for key in run.keys():
                if key in self.tuning_dict: 
                    run[key] = False

        # head direction
        if run['hdir']:
            filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'hdir')
            self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
            
        # head shelter angle
        if len(self.processed_data.Visualize.session.shelter_time) > 0:
            if run['hsa']:
                filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'head_shelter_angle', object_present = True)
                self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
                if not(np.logical_and(self.processed_data.Visualize.session.shelter_time[0] == 0, self.processed_data.Visualize.session.shelter_time[1] == -1)):
                    if run['pre_hsa']:
                        filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'head_shelter_angle',object_present = False)
                        self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
                        
        # head barrier angle
        if len(self.processed_data.Visualize.session.barrier_time) > 0:
            if run['h_bar_south_a']:
                filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'head_south_barrier_angle',object_present = True)
                self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
                
            if run['h_bar_north_a']:
                filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'head_north_barrier_angle',object_present = True)
                self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
                
            if not(np.logical_and(self.processed_data.Visualize.session.barrier_time[0] == 0, self.processed_data.Visualize.session.barrier_time[1] == -1)):
                if run['pre_h_bar_south_a']:
                    filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'head_south_barrier_angle',object_present = False)
                    self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
                    
                if run['pre_h_bar_north_a']:
                    filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, 'head_north_barrier_angle',object_present = False)
                    self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)

        # individual figures for each cluster with all polar plots
        number_of_clusters = self.processed_data.spikedataframe["spike_clusters"].unique()
        for cluster in number_of_clusters:
            self.single_cluster_polar_plots(cluster)
            
        logger.success("Finished making figures of each individual cluster and all its respective tuning plots")
        
    def compute_a_single_tuning_for_all_cells(self, which_angle, compute_bootstrap = False, object_present = True):
        """
        Calculates tuning for individual clusters. Has two modes:
        1. input an angle (e.g. 'hdir') and it computes Rayleigh R and makes polar plots for that angle
        2. all_tuning_by_cluster goes through each cluster and computes rayleigh R for all possible angles ('hdir', 'hsa' and 'hba') 
        and plots the polar plots in one figure for each cluster
        """
        # subselect frames of interest:
        # 1. mouse has to be outside shelter
        # 2. for hdir take all time, for hsa take times when only a shelter was present in the arena, for hba take times when barrier was present
        # 3. exclude threat stimuli times and the escape
        filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, which_angle, object_present)

        logger.info("Commence making figures of every cluster for a single tuning curve")
        
        # compute tuning
        logger.info("Calculating Rayleigh vectors for condition: " + str(title) + ", is object present? " + str(object_present))
        self.rayleigh_vector(filtered_video_df, angle_filt, title, compute_bootstrap)
        
        logger.info(f"Finished calculating Rayleigh vectors, moving on to polar plots")
        self.all_clusters_polar_plots(title) 
        
        logger.success("Finished making figures of every cluster for a single tuning curve")

    def rayleigh_vector(self, filtered_video_df, angle_filt, title, compute_bootstrap = False):
        """A function that calculates the Rayleigh vector (amplitude and angle) for each cluster with respect to the angles given (e.g. HD or HSA)
        It only considers times when the mouse was outside the shelter
        It also performs bootstrapping by computing the rayleigh vector at random time shifts of the spikes with respect to the angles
        The Rayleigh vector is significant if the amplitude is above the 95th percentile of boostrapped amplitudes
        Rayleigh's R close to zero = untuned, fires at all head directions
        Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation"""
        
        # edges for binning firing rate at different angles
        bin_angles, bin_angle_center = generate_bin_angles(number_of_bins = 19)
        
        if not hasattr(self, 'tuning_dict'):
            self.tuning_dict = {'angles': bin_angle_center[1:-1]}
            
        # Catch empty video dataframes
        if len(filtered_video_df) == 0:
            raise ValueError("Video dataframe is empty, bug.")

        # initialize variables to compute the Rayleigh vector
        number_of_clusters = self.processed_data.spikedataframe["spike_clusters"].unique()
        Rayleigh_theta, Rayleigh, Rayleigh_sig, Rayleigh_cluster, angle_firing_hist = init_rayleigh(number_of_clusters, bin_angle_center)
        
        # assign spike times of each cluster to the corresponding video frame, then assign HD
        for counter,c in enumerate(number_of_clusters):
                        
            # filter by cluster
            spikes = self.processed_data.spikeCountByFrameAndCluster.filter(self.processed_data.spikeCountByFrameAndCluster['spike_clusters'] == c)
            
            # Convert spike count to firing rate
            spikes = spikes.with_columns(pl.col('spike_count')*self.processed_data.Visualize.session.video.fps)
                                 
            # Cast frames to float to permit join and remove old frames column with wrong type 
            filtered_video_df = filtered_video_df.select([pl.col('frames').apply(float), pl.exclude('frames')])
            
            # align spike dataframe to video dataframe
            spike_to_video_df = filtered_video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="left")
            
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
            Rayleigh[counter], Rayleigh_theta[counter] = compute_rayleigh(all_angles_firing['all_angles'].to_numpy(),all_angles_firing['mean_firing_rate'].to_numpy())
            Rayleigh_cluster[counter] = c
            angle_firing_hist[counter,:] = all_angles_firing['mean_firing_rate'].to_numpy()
            
            # bootstrap x times with variable shifts in time
            if compute_bootstrap:
                x = 100
                shift_dist = np.empty(x)
                for it in np.arange(len(shift_dist)): 
                    # shuffled shifts performed at a random offset between 0 and 100 seconds
                    shift = int(np.random.uniform(1,100))*self.processed_data.Visualize.session.video.fps # temporal shift in video frames
                    angles = filtered_video_df[angle_filt].to_numpy()
                    ang_roll = np.roll(angles,shift)
                    rolled_filtered_video_df = filtered_video_df.select(pl.col('*'),pl.Series(name="rolled_angles", values = ang_roll))
                    # align spike dataframe to video dataframe
                    spike_to_video_df = rolled_filtered_video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="left")
                    # calculate firing rates in angle bins
                    spike_to_video_df = spike_to_video_df.sort('rolled_angles') # polars can be annoying, when using cut it doesn't preserve order :/
                    spike_to_video_df = spike_to_video_df.with_columns(spike_to_video_df['rolled_angles'].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
                    spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
                    spike_to_video_df = spike_to_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) # TODO add this line to rayleigh v function
                    angles_firing = (spike_to_video_df.groupby(by ='binned_angles').agg(pl.col('spike_count').mean().alias('mean_firing_rate')))            
                    angles_firing = angles_firing.sort('binned_angles')
                    # make sure that if any angles returned empty sets of spikes, they are registered as zeros and are not missing
                    all_angles_firing = pl.DataFrame({'all_angles': bin_angle_center[1:-1]})
                    all_angles_firing = all_angles_firing.join(angles_firing, left_on="all_angles", right_on="binned_angles", how="left")
                    all_angles_firing = all_angles_firing.fill_null(strategy="zero")
                    # compute rayleigh
                    shift_dist[it], _ = compute_rayleigh(all_angles_firing['all_angles'].to_numpy(),all_angles_firing['mean_firing_rate'].to_numpy())

                # significance logical
                if Rayleigh[counter] > np.percentile(shift_dist,95):
                    Rayleigh_sig[counter] = 1
                    print('yay!')

        # histogram of rayleighs
        plt.figure()
        plt.hist(Rayleigh,np.arange(0,1,.1))
        plt.hist(Rayleigh[Rayleigh_sig == 1],np.arange(0,1,.1))
        plt.xlabel('Rayleigh R')
        plt.ylabel('number of clusters')
        plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + str(title)  + "_" + self.processed_data.select_clusters +  "_Rayleigh_vector_hist.png")
        if self.processed_data.Visualize.settings.show_plots: plt.show()

        # save all to dict
        self.tuning_dict[title] = angle_firing_hist

        if not hasattr(self, 'Rayleigh'):
            self.Rayleigh = {title: Rayleigh}
            self.Rayleigh_theta = {title: Rayleigh_theta}
            self.Rayleigh_sig = {title: Rayleigh_sig}
            self.Rayleigh_cluster = {title: Rayleigh_cluster}
        else:
            self.Rayleigh[title] = Rayleigh
            self.Rayleigh_theta[title] = Rayleigh_theta
            self.Rayleigh_sig[title] = Rayleigh_sig
            self.Rayleigh_cluster[title] = Rayleigh_cluster

    def all_clusters_polar_plots(self, title):
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
        
        number_of_clusters = self.processed_data.spikedataframe["spike_clusters"].unique()
        logger.info("About to generate plots for {} clusters".format(len(number_of_clusters)))
        for counter,c in enumerate(number_of_clusters):
            
            # if you have filled a figure with polar plots, move onto next figure
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
                
            ax = plt.subplot(nrows,ncols,1+counter-(nrows*ncols*(fnum-1)),projection = 'polar')
            
            # polar plots!
            if self.tuning_dict['angles'].size > 0 and len(self.tuning_dict[title][counter,:]) > 0:
                ax.bar(self.tuning_dict['angles'], 
                    self.tuning_dict[title][counter,:], 
                    width=(2*np.pi)/(len(self.tuning_dict['angles'])+1), 
                    bottom=0.0, 
                    color='green', 
                    alpha=0.5)
                
            else:
                logger.warning(f"Empty array for cluster {c}. Skipping plot.")
                continue # skip this cluster
            
            ax.vlines(self.Rayleigh_theta[title][counter], 
                      0, 
                      self.Rayleigh[title][counter]*np.amax(self.tuning_dict[title][counter,:]), colors='black')
            
            # add title to the subplot
            this_cluster = self.processed_data.clu_label.filter(self.processed_data.clu_label["spike_clusters"] == [c])
            if self.Rayleigh_sig[title][counter] == 1:
                ax.title.set_text(str(this_cluster["cluster_group"].to_numpy()) + ' clu ' + str(c) + ' (sig.)' + 
                                    '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[title][counter],2)))
                
            else:
                ax.title.set_text(str(this_cluster["cluster_group"].to_numpy()) + ' clu ' + str(c) + 
                                    '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[title][counter],2)))
                
            # save the whole figure
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1, counter == len(number_of_clusters)-1):
                plt.tight_layout()
                plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + str(title) + "_" + self.processed_data.select_clusters + "_cluster_polar_plots_" + str(fnum) + ".png")
                if self.processed_data.Visualize.settings.show_plots: 
                    plt.show()  

    def single_cluster_polar_plots(self, cluster):
        """Plots all polar plots for 1 cluster in 1 figure"""

        tuning_angles = ['hdir', 'hsa', 'h_bar_south_a', 'h_bar_north_a']
        figg, _ = plt.subplots(1, len(tuning_angles))
        figg.set_figwidth(30)

        for subp, angle in enumerate(tuning_angles):
            ax = plt.subplot(1,4,subp+1,projection = 'polar')
            
            if str('pre_' + angle) in self.tuning_dict:
                counter = np.where(self.Rayleigh_cluster[str('pre_' + angle)] == cluster)[0]
                
                if len(counter) > 0:
                    ax.bar(self.tuning_dict['angles'], self.tuning_dict[str('pre_' + angle)][counter,:][0], width=(2*np.pi)/(len(self.tuning_dict['angles'])+1), bottom=0.0, color='red', alpha=0.5)
                    ax.vlines(self.Rayleigh_theta[str('pre_' + angle)][counter][0], 0, self.Rayleigh[str('pre_' + angle)][counter][0]*np.amax(self.tuning_dict[str('pre_' + angle)][counter,:][0]), colors='red')
                    ax.title.set_text(angle + '\n' + 'preRayleigh = ' + str(np.around(self.Rayleigh[str('pre_' + angle)][counter][0],2)) + ', sig = ' + str(np.around(self.Rayleigh_sig[str('pre_' + angle)][counter][0],2))
                                    + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[angle][counter][0],2)) + ', sig = ' + str(np.around(self.Rayleigh_sig[angle][counter][0],2)))
            
            if angle in self.tuning_dict:
                counter = np.where(self.Rayleigh_cluster[angle] == cluster)[0]
                
                if len(counter) > 0:
                    ax.bar(self.tuning_dict['angles'], self.tuning_dict[angle][counter,:][0], width=(2*np.pi)/(len(self.tuning_dict['angles'])+1), bottom=0.0, color='green', alpha=0.5)
                    ax.vlines(self.Rayleigh_theta[angle][counter][0], 0, self.Rayleigh[angle][counter][0]*np.amax(self.tuning_dict[angle][counter,:][0]), colors='green')
                                        
                    if not(str('pre_' + angle) in self.tuning_dict):
                        ax.title.set_text(angle + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[angle][counter][0],2)) + ', sig = ' + str(np.around(self.Rayleigh_sig[angle][counter][0],2)))

        plt.tight_layout()

        cluster_path = os.path.join(self.processed_data.Visualize.session.processed_path, str(self.processed_data.select_clusters + "_cluster_plots"))
        if not(os.path.exists(cluster_path)): os.makedirs(cluster_path)
        if np.logical_or(self.processed_data.select_clusters == "all", self.processed_data.select_clusters == "synthetic"):
            this_cluster = self.processed_data.clu_label.filter(self.processed_data.clu_label["spike_clusters"] == [cluster])
            plt.savefig(str(cluster_path + "/" + this_cluster["cluster_group"].to_numpy()[0] + "_cluster" + str(cluster) + "_polar_plots.png"))
        else:
            plt.savefig(str(cluster_path + "/cluster" + str(cluster) + "_polar_plots.png"))
        if self.processed_data.Visualize.settings.show_plots: 
            plt.show()  
        plt.close()

    def spatial_position_firing(self):
        """ A function that makes maps of mousie's position in arena
        and show where each cluster fired"""

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

        # what is firing rate per frame?
        for counter,cluster in enumerate(self.processed_data.spikedataframe["spike_clusters"].unique()):
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            # filter spikes by cluster
            # spikes = self.processed_data.spikedataframe.filter(self.processed_data.spikedataframe['spike_clusters'] == cluster)
            # count number of spikes on each video frame, and then turn it into firing rate (Hz)
            # spikes = spikes.groupby("spike_aligned_to_frame").agg([pl.count("spike_aligned_to_frame").alias("spike_count")])
            spikes = self.processed_data.spikeCountByFrameAndCluster.filter(self.processed_data.spikeCountByFrameAndCluster['spike_clusters'] == cluster)
            spikes = spikes.with_columns(pl.col('spike_count')*self.processed_data.Visualize.session.video.fps)
            # align spike dataframe to video dataframe
            filtered_video_df = self.processed_data.Video_df.select([pl.col('frames').apply(float),pl.exclude('frames')])
            spike_to_video_df = filtered_video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="left")
            spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
            axs[counter-(nrows*ncols*fnum)].scatter(spike_to_video_df['mouse_x_position'].to_numpy(),
                                                    spike_to_video_df['mouse_y_position'].to_numpy(),
                                                    s=5,c=cc(spike_to_video_df['spike_count'].to_numpy()*2),linewidths=0,marker='.') # srate*2 increase contrast
            axs[counter-(nrows*ncols*fnum)].set_axis_off()
            axs[counter-(nrows*ncols*fnum)].invert_yaxis()
            axs[counter-(nrows*ncols*fnum)].set_aspect('equal')
            this_cluster = self.processed_data.clu_label.filter(self.processed_data.clu_label["spike_clusters"] == [cluster])
            axs[counter-(nrows*ncols*fnum)].title.set_text(str(this_cluster["cluster_group"].to_numpy()) + ' cluster ' + str(cluster))

            # save the figure
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1, counter == len(self.processed_data.spikedataframe["spike_clusters"].unique())-1):
                plt.tight_layout()
                plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + self.processed_data.select_clusters + "_clusters_spatial_position_firing_" + str(fnum) + ".png")
                if self.processed_data.Visualize.settings.show_plots: 
                    plt.show()
                #plt.close()
    
    def linear_discriminant_analysis(self, variable):
        """
        A function for doing LDA on data
        variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
        """
        epoch_num = 6 # chunks of time for training and testing data

        # edges for binning firing rate at different angles
        bin_angles, bin_angle_center = generate_bin_angles(number_of_bins = 19)

        # align ephys to behave
        filtered_video_df, angle_filt, title = filter_video_dataframe(self.processed_data.Video_df, variable, object_present = True)
        filtered_video_df = filtered_video_df.select([pl.col('frames').apply(float), pl.exclude('frames')])
        spike_to_video_df = filtered_video_df.join(self.processed_data.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")

        # bin angles
        spike_to_video_df = spike_to_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
        spike_to_video_df = spike_to_video_df.with_columns(spike_to_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in np.arange(len(bin_angle_center))])['category'].alias('binned_angles'))
        spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
        spike_to_video_df = spike_to_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 

        # chunk data into training and test data
        epoch_edge = np.round(np.linspace(spike_to_video_df["frames"].unique().min()-1,spike_to_video_df["frames"].unique().max(),epoch_num+1))
        epoch_df = spike_to_video_df.sort("frames")
        epoch_df = epoch_df.with_columns(epoch_df["frames"].cut(bins = epoch_edge, labels = [str(x) for x in np.arange(epoch_num+2)])['category'].alias('binned_frames'))
        epoch_df = epoch_df.fill_null(strategy="zero")
        epoch_df = epoch_df.select([pl.col('binned_frames').apply(float),pl.exclude('binned_frames')]) 

        # LDA
        train = epoch_df.filter((epoch_df['binned_frames'] == 1) | (epoch_df['binned_frames'] == 3) | (epoch_df['binned_frames'] == 5))
        
        # group the training data
        train_g2 = train.groupby(["frames"]).first()
        train_all = train.groupby(["frames"]).all()
        train_all.replace("binned_angles",train_g2['binned_angles'])
        train_all.replace("binned_frames",train_g2['binned_frames'])

        # make angle bins equally populated
        train_samples = train_all.groupby(['binned_angles']).count().min()
        samples = train_samples['count'].to_numpy()[0]

        for c, i in enumerate(train_all['binned_angles'].unique()):
            d_filt = train_all.filter(train_all['binned_angles'] == i)
            d_filt = d_filt.sample(samples)
            if c == 0: d_new = d_filt
            if c > 0: d_new = d_new.vstack(d_filt)
        
        d_new = d_new.sort('frames')
        # d_new = train_all
        X = np.zeros((len(d_new["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
        clu = epoch_df["spike_clusters"].unique().to_numpy()
        
        fillMatrix(d_new,X,clu)
        if clu[0] == 0: X = X[:,1:]
        X = X/np.amax(X,axis=0)

        # train model
        y = d_new["binned_angles"].to_numpy()
        
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)

        # plot confusion matrix of prediction on training data
        conf = confusion_matrix(y, clf.predict(X))
        conf = conf.astype('float64')
        conf = conf/np.sum(conf,axis=1)
        # for n in np.arange(len(conf)):
        #     conf[n,:] = conf[n,:]/np.sum(conf[n,:]) # the number of frames at each binned angle is the same as the sum of each row of conf matrix
        
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        ax = plt.subplot2grid(shape=(4, 2), loc=(2, 0))
        ax.imshow(conf, cmap = "Blues")
        ax.set_ylabel('real')
        ax.set_xlabel('predicted')
        ax.set_title('training data')

        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
        ax.hist(clf.predict(X))
        ax.hist(y)
        ax.set_title('training data')

        # look at data side-by-side
        ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
        ax.plot(clf.predict(X))
        ax.plot(y)
        ax.legend(["prediction","real"])
        ax.set_title("training data")
        ax.set_ylabel('binned angles')
        ax.set_xlabel('time')

        # predict test data
        test = epoch_df.filter((epoch_df['binned_frames'] == 2) | (epoch_df['binned_frames'] == 4) | (epoch_df['binned_frames'] == 6))
        
        # group the test data
        test_g2 = test.groupby(["frames"]).first()
        test_all = test.groupby(["frames"]).all()
        test_all.replace("binned_angles",test_g2['binned_angles'])
        test_all.replace("binned_frames",test_g2['binned_frames'])

        # make angle bins equally populated
        test_samples = test_all.groupby(['binned_angles']).count().min()
        samples = test_samples['count'].to_numpy()[0]

        for c, i in enumerate(test_all['binned_angles'].unique()):
            d_filt = test_all.filter(test_all['binned_angles'] == i)
            d_filt = d_filt.sample(samples)
            if c == 0: d_new_test = d_filt
            if c > 0: d_new_test = d_new_test.vstack(d_filt)
        
        d_new_test = d_new_test.sort('frames')
        # d_new_test = test_all
        X2 = np.zeros((len(d_new_test["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
        
        fillMatrix(d_new_test,X2,clu)
        if clu[0] == 0: X2 = X2[:,1:]
        X2 = X2/np.amax(X2,axis=0)

        # plot confusion matrix of prediction on test data
        y = d_new_test["binned_angles"].to_numpy()
        conf_test = confusion_matrix(y, clf.predict(X2))
        conf_test = conf_test.astype('float64')
        conf_test = conf_test/np.sum(conf_test,axis=1)
        # for n in np.arange(len(conf)):
        #     conf_test[n,:] = conf_test[n,:]/np.sum(conf_test[n,:])
        
        ax = plt.subplot2grid(shape=(4, 2), loc=(2, 1))
        ax.imshow(conf_test, cmap = "Blues")
        ax.set_ylabel('real')
        ax.set_xlabel('predicted')
        ax.set_title('test data')

        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
        ax.hist(clf.predict(X2))
        ax.hist(y)
        ax.set_title('test data')

        # look at data side-by-side
        ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
        ax.plot(clf.predict(X2))
        ax.plot(y)
        ax.legend(["prediction","real"])
        ax.set_title("test data")
        ax.set_ylabel('binned angles')
        ax.set_xlabel('time')

        plt.savefig(str(self.processed_data.Visualize.session.processed_path) + "/" + str(self.processed_data.select_clusters) + "_LDA_" + str(title) + ".png")
        if self.processed_data.Visualize.settings.show_plots: plt.show()
        plt.close()

# Utility functions ------------------------------------------------------------------------------------------------

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
    
def compute_rayleigh(angles,firing):
    x = np.sum(np.cos(angles)*(firing))/np.sum(firing)
    y = np.sum(np.sin(angles)*(firing))/np.sum(firing)
    theta = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    return r, theta

def filter_video_dataframe(dataframe, which_angle, object_present):
    """
    A function that filters the video dataframe (the behavioural data) by angle of interest and object presence (whether the barrier or shelter is present or not)
    """
    if which_angle == 'head_shelter_angle':
        filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == True) & 
                                            (dataframe["EscapePeriod"] == False) & 
                                            (dataframe["shelter_only"] == object_present))
        angle_filt = 'hsa'

    elif which_angle == 'head_south_barrier_angle':
        filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == True) & 
                                            (dataframe["EscapePeriod"] == False) & 
                                            (dataframe["barrier_present"] == object_present))
        angle_filt = 'h_bar_south_a'

    elif which_angle == 'head_north_barrier_angle':
        filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == True) &
                                            (dataframe["EscapePeriod"] == False) & 
                                            (dataframe["barrier_present"] == object_present))
        angle_filt = 'h_bar_north_a'

    elif which_angle == 'hdir':
        filtered_video_df = dataframe.filter((dataframe["OutofshelterIdx"] == True) &
                                            (dataframe["EscapePeriod"] == False))
        angle_filt = 'hdir'
        
    title = angle_filt
    if object_present == False: 
        title = str('pre_' + angle_filt)
        
    return filtered_video_df, angle_filt, title

def find_bin_labels(angles, bins, labels): 
    return np.array(labels)[np.digitize(angles, bins, right=False) - 1]

def generate_bin_angles(number_of_bins): 
    bin_angles = np.linspace(-np.pi, np.pi, number_of_bins)
    bin_angle_center = np.sort(np.append([-np.pi,np.pi], [bin_angles[:-1] + (np.mean(np.diff(bin_angles))/2)]))
    return bin_angles, bin_angle_center

def calculate_figure_plotting_axes(how_many_plots_you_need):
    max_plots_per_figure = 20
    num_cols = int(np.ceil(np.sqrt(max_plots_per_figure)))
    num_rows = int(np.ceil(max_plots_per_figure / num_cols))
    num_figures = int(np.ceil(how_many_plots_you_need / max_plots_per_figure))
    return num_cols, num_rows, num_figures

def fillMatrix(df,matrix,clu_id):
    for i, i2 in enumerate(df["frames"].unique()):
        d = df.filter(df["frames"] == i2).to_dict(as_series=False)
        matrix[i,np.where(np.in1d(clu_id, d.get('spike_clusters')))[0]] = d.get('spike_count') 
        