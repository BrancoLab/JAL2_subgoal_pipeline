# OS libaries
import numpy as np
from glob import glob
import polars as pl
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from loguru import logger

class Visualize_efizz():
    """
    A class for some sanity check efizz plots using kilosort clusters
    """
    def __init__(self, Visualize, run = "Production"):
        self.Visualize = Visualize
        self.run_type = run
        
        self.load_spike_data()
        self.process_spike_data()
    
# INIT FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def load_spike_data(self):
        """
        Loads the csv of aligned data
        """
        if self.run_type == "Production":
            self.csv_path = glob(os.path.join(self.Visualize.session.file_path, "Processed_efizz_data"))[0]
        
        elif self.run_type == "Test":
            logger.info("Synethic data is being used when visualizing efizz")
            self.csv_path = r"C:\Users\laurence\Documents\JAL-pipeline\behave_analysis\database\synthetic_data\synthetic_dataframe.csv"
    
        else: 
            raise ValueError("Run type not recognised")
        
    def process_spike_data(self):
        self.dataFrame = pl.read_csv(self.csv_path)
        self.dataFrame_filt_on_good_neurons = self.dataFrame.filter(self.dataFrame['cluster_group'] == 'good')
        self.array_of_good_neurons_IDs = self.dataFrame_filt_on_good_neurons["spike_clusters"].unique()
        
        # Old code leaving incase it breaks anything - Ideally we should be using the above code utilizing polars and not numpy for speed
        aligned_spike_data = pl.read_csv(self.csv_path, has_header=True)
    
        # Hard code for one neuron TODO remove
        # aligned_spike_data = aligned_spike_data.filter(aligned_spike_data['spike_clusters'] == 3)
        
        
        asd_np = aligned_spike_data.to_numpy() # What is asd? Is that aligned spike data?
        # self.aligned_spikes = aligned_spike_data.get_column("aligned_spike_times").to_numpy()
        
        # filter by 'good' clusters
        self.aligned_spikes = np.array([asd_np[asd_np[:,2] == 'good', 3]]).T # This says for every row select the 3rd column if it's good
        self.clu_spikes = asd_np[asd_np[:,2] == 'good',1]
        print("Loaded spike data")
    
# Extraction ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def extract_trial_spikes(self, 
                             stim_type, 
                             time_before_stim = 3, 
                             time_after_stim = 2, 
                             select_good_neurons = False,
                             onsets = "Production") -> dict:
        """
        A function that extracts spikes times between trials and aligns it to the stimulus onset.
        The function returns a dictionary with the cluster ID as the key and a list of polars dataframes as the value.
        Each polar dataframe contains the spikes for a single trial.
        
        Returns Dictionary = {cluster_ID1: [polar_dataframe_trial_1, polar_dataframe_trial_2, ...], 
                              cluster_ID2: [polar_dataframe_trial_1, polar_dataframe_trial_2, ...], 
                              ...}
                              
        """
        
        # If no onsets are provided, use the onsets from the session
        if onsets is "Production":
            self.onsets = self.Visualize.session.__dict__[stim_type].onset_frames
            
        elif onsets is "Synthetic_test_onsets":
            self.onsets = np.load(r"C:\Users\laurence\Documents\JAL-pipeline\behave_analysis\database\synthetic_data\synthetic_onsets.pkl", allow_pickle=True)

        dic = {}
        
        if select_good_neurons:
            iterator = self.array_of_good_neurons_IDs
        
        elif not select_good_neurons:
            iterator = self.dataFrame["spike_clusters"].unique()
            
        for index, cluster in enumerate(iterator):
            spikes = self.dataFrame.filter(self.dataFrame["spike_clusters"] == cluster)
            spikesByTrial = []
            for trial, onset in enumerate(self.onsets):                
                time1 = (onset / 30000) - time_before_stim
                time2 = (onset / 30000) + time_after_stim
                trialSpikesPolars = spikes.filter((spikes["aligned_spike_times"] > time1) & (spikes["aligned_spike_times"] < time2))
                adjustedSpikeTimes = trialSpikesPolars.with_column(trialSpikesPolars['aligned_spike_times'] - onset / 30000)
                spikesByTrial.append(adjustedSpikeTimes)
            dic[cluster] = spikesByTrial
        
        return dic
    
    def filter_spikes(self,angles,times):
        """
        Filter the data for times when the mouse was not in the shelter,
        and for barrier-only or shelter-only times
        """
        # find times when the mouse was not inside the shelter
        OutofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0],
            self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]),
            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1],
            self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1])))
       
        # only look at tuning when shelter (and no barrier) was in arena
        if times[1] == -60: # -1 times 60 hehe
            idx = self.aligned_spikes > times[0]
            end_time = len(angles)/self.Visualize.session.video.fps
            angles = angles[times[0]*self.Visualize.session.video.fps:]
            OutofShelterIdx = OutofShelterIdx[times[0]*self.Visualize.session.video.fps:]
        else:
            idx = np.logical_and(self.aligned_spikes > times[0],self.aligned_spikes < times[1])
            end_time = times[1]
            angles = angles[times[0]*self.Visualize.session.video.fps:times[1]*self.Visualize.session.video.fps]
            OutofShelterIdx = OutofShelterIdx[times[0]*self.Visualize.session.video.fps:times[1]*self.Visualize.session.video.fps]
        spikes = self.aligned_spikes[idx]
        clusters = self.clu_spikes[idx[:,0]]
        return spikes,clusters,angles,OutofShelterIdx,end_time
    
# RASTER PLOTS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def plot_single_cluster_raster(self, **kwargs):
        """
        A function that extracts spike times for each good cluster and aligns it to trials as a raster plot
        """
        # Load data - Choose whether to use good neurons or all neurons
        # cluster_trial_spikes_dic = self.extract_trial_spikes(kwargs["stim_type"], select_good_neurons = False)
        
        # How many plots do we need?
        number_of_plots = len(kwargs["spikes_by_trials_and_cluster"])
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
                        cluster = list(kwargs["spikes_by_trials_and_cluster"].keys())[plot_counter]
                        
                        for trial, onset_frames in enumerate(self.onsets):
                            x_values = kwargs["spikes_by_trials_and_cluster"][cluster][trial]["aligned_spike_times"]
                            y_values = [trial] * len(x_values)
                            axes[rows, columns].scatter(x_values, y_values, marker='|', s=100, c='k')
                        axes[rows, columns].set_title(f"Cluster: {cluster}")
                        axes[rows, columns].vlines(0, 0, len(self.onsets), colors='r', linestyles='solid')
                        axes[rows, columns].set_xbound(-4, 5)
                    
                    # Remove the extra axes if there are no more plots
                    else:
                        fig.delaxes(axes[rows, columns])
                    
                    plot_counter += 1
            
            # SAVE FIGURE
            fig.tight_layout()
            plt.savefig(str(self.Visualize.session.file_path) + "/" + str(kwargs["stim_type"]) + "_single_cluster_raster_" + str(figure_idx) + ".png")
        
        plt.show()                
        
        if self.Visualize.settings.show_plots: 
            plt.show()
                            
    def rasters(self, stim_type):
        """
        A function that extracts spike times and aligns it to trials as a raster plot
        """
        # make a raster plot for each trial
        ntrial = len(self.Visualize.session.__dict__[stim_type].onset_frames)
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)
        # set number of rows and calculate number of columns
        nrows = 3
        ncols = ntrial // nrows + (ntrial % nrows > 0)
        timeBeforeStim = 5 # in seconds

        for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            t1 = (onset_frames/self.Visualize.session.video.fps) - timeBeforeStim
            t2 = (onset_frames/self.Visualize.session.video.fps) + stimulus_durations
            idx = np.logical_and(self.aligned_spikes[:,0] > t1,self.aligned_spikes[:,0] < t2)
            ax.eventplot(self.aligned_spikes[idx]-(onset_frames/self.Visualize.session.video.fps), lineoffsets = self.clu_spikes[idx])
            ax.plot([0,0],[0, np.amax(self.clu_spikes)],'r-')
            ax.set_ylabel('clusters')
            ax.set_xlabel('time from stim (s)')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + "all_cluster_raster_trial.png")
        
        logger.success("Raster plots produced and saved to: " + str(self.Visualize.session.file_path))
        
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()

    def OLD_single_cluster_raster(self,stim_type):
        """
        A function that extracts spike times for each goo cluster and aligns it to trials as a raster plot
        """
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)
        # set number of rows and calculate number of columns
        ncols = 10
        nrows = 5 # nclu // ncols + (nclu % ncols > 0)
        figg, axs = plt.subplots(nrows,ncols)
        figg.set_figwidth(30)
        figg.set_figheight(15)
        fnum = 1
        axs = axs.ravel()
        timeBeforeStim = 10 # in seconds

        for i,c in enumerate(np.unique(self.clu_spikes)):
            spikes = self.aligned_spikes[self.clu_spikes == c]
            trial_idx = np.zeros_like(spikes)
            spikes_idx = np.zeros_like(spikes)
            trial_length = np.amax(self.Visualize.session.__dict__[stim_type].stimulus_durations)
            for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
                t1 = (onset_frames/self.Visualize.session.video.fps) - timeBeforeStim
                t2 = (onset_frames/self.Visualize.session.video.fps) + trial_length
                trial_idx[np.logical_and(spikes > t1,spikes < t2)] = trial_num + 1
                spikes_idx[np.logical_and(spikes > t1,spikes < t2)] = spikes[np.logical_and(spikes > t1,spikes < t2)] - (onset_frames/self.Visualize.session.video.fps)
            if i >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            axs[i-(nrows*ncols*fnum)].plot([0,0],[0, np.amax(trial_idx)],'r-')
            if np.sum(trial_idx.astype(int)) > 0:
                axs[i-(nrows*ncols*fnum)].eventplot(np.array([spikes_idx[trial_idx > 0]]).T,lineoffsets = np.array(trial_idx[trial_idx > 0]))
                axs[i-(nrows*ncols*fnum)].set_xlabel('time from stim (s)')
                axs[i-(nrows*ncols*fnum)].set_xbound(-timeBeforeStim,np.amax(self.Visualize.session.__dict__[stim_type].stimulus_durations))
                axs[i-(nrows*ncols*fnum)].set_ylabel('trials')
            axs[i-(nrows*ncols*fnum)].title.set_text('cluster ' + str(c))
            if np.logical_or(i-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1,
                            i == len(np.unique(self.clu_spikes))-1):
                plt.tight_layout()
                plt.savefig(str(self.Visualize.session.file_path) + "/" + str(stim_type) + "_single_cluster_raster_" + str(fnum) + ".png")
                if self.Visualize.settings.show_plots: plt.show()
                plt.close()

# Vectorial functions--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def HD_tuning(self) -> None:
        """
        Make heatmaps of each cell's firing at each HD, for first and second half of recording, sorted on first half
        """
        spikes,clusters,angles,OutofShelterIdx,end_time = self.filter_spikes(self.Visualize.tracking_data['hdir'], [0, -60])
        
        logger.info("Calculating Rayleigh vector for each cluster with respect to head direction")
        
        self.rayleigh_vector(spikes = spikes,
                             clusters = clusters,
                             angles = angles,
                             OutofShelterIdx = OutofShelterIdx,
                             times = [0, -60],
                             end_time = end_time,
                             title = 'head_dir')
        
        logger.info("Calculating tuning heatmap for each cluster with respect to head direction")
        
        self.tuning_heatmap(title = 'head_dir',
                            times = [0, -60],
                            spikes = spikes,
                            clusters = clusters,
                            angles = angles,
                            OutofShelterIdx = OutofShelterIdx,
                            end_time = end_time) # find times when the mouse was not inside the shelter

    def rayleigh_vector(self,spikes,clusters,angles,OutofShelterIdx,times,end_time, title):
        """A function that calculates the Rayleigh vector (amplitude and angle) for each cluster with respect to the angles given (e.g. HD or HSA)
        It subsamples angles within 20 degree bins to ensure that angles are more uniformly sampled
        It only considers times when the mouse was outside the shelter
        It also performs bootstrapping by computing the rayleigh vector at random time shifts of the spikes with respect to the angles
        The Rayleigh vector is significant if the amplitude is above the 95th percentile of boostrapped amplitudes"""    
        # timepoints in seconds for each frame (and angles, OutofShelterIdx)
        timepoints = np.arange(times[0], # start of timewindow
                               end_time, # end of timewindow
                               1/self.Visualize.session.video.fps) # each time bin is 1 frame
        
        # randomly subsample frames at 20degree angle intervals 
        # this ensure uniform sampling at all angles, even if mouse behavior was not uniform
        binned_angles = np.digitize(angles,np.linspace(-np.pi,np.pi,19)) # steps of 20deg
        all_permuted_angles = np.zeros_like(angles)
        for b in np.arange(1,np.amax(binned_angles)+1):
            angles_in_bin = np.where(binned_angles == b)[0]
            rng = np.random.default_rng()
            permuted_angles = rng.permuted(angles_in_bin)
            permuted_angles = permuted_angles[:int(len(binned_angles)*.03)] # an index of subsampled angles
            all_permuted_angles[permuted_angles] = 1

        # initialize variables
        self.Rayleigh_theta = np.empty([len(np.unique(clusters))]) # preferred angle
        self.Rayleigh = np.empty([len(np.unique(clusters))]) # amplitude of Rayleigh vector
        self.Rayleigh_sig = np.zeros([len(np.unique(clusters))]) # is the Ryleigh significant?
        self.Rayleigh_cluster = np.empty([len(np.unique(clusters))]) # which cluster ID is this Rayleigh value for?

        # assign spike times of each cluster to the corresponding video frame, then assign HD
        for counter,c in enumerate(np.unique(clusters)):
            # bin spikes per cluster to 1 ms (to get rid of double counting the wider spikes)
            spikes_by_cluster = spikes[clusters == c]
            time_bins_1ms = np.arange(spikes_by_cluster[0],spikes_by_cluster[-1],.001)
            spike_by_bin = np.histogram(spikes_by_cluster,time_bins_1ms)[0]>0
            spikes_by_cluster = (time_bins_1ms[np.where(spike_by_bin)[0]]+.0005)
            # at which video frame did the spikes happen? 
            frame_per_spike = np.digitize(spikes_by_cluster,timepoints)
            frame_per_spike = frame_per_spike[frame_per_spike <= len(angles)] # delete the spikes happening after the last of timepoints
            # select spikes when mouse is outside shelter and at one of subsampled angles
            angle_per_spike = np.zeros_like(frame_per_spike)
            # keep_spike = np.zeros_like(frame_per_spike)
            
            frames_by_spikes_polars = pl.DataFrame(
                {
                    "spike_times": spikes_by_cluster,
                    "frame_per_spike": frame_per_spike.astype(np.int32),
                }
            )
            
            angle_by_frames_polars = pl.DataFrame(
                {
                    "frames": np.arange(1,len(angles)+1),
                    "angles": angles,
                    "OutofShelter": OutofShelterIdx,
                    "all_permuted_angles": all_permuted_angles
                }
            )
            
            result_df = frames_by_spikes_polars.join(angle_by_frames_polars, left_on="frame_per_spike", right_on="frames", how="left")
            
            # for f in np.unique(frame_per_spike):
            #     angle_per_spike[frame_per_spike == f] = angles[f-1]
            #     if np.logical_and(OutofShelterIdx[f-1] == True,all_permuted_angles[f-1] == 1): 
            #         keep_spike[frame_per_spike == f] = 1
                
            # angle_per_spike = angle_per_spike[keep_spike == 1]
            filtered_df = result_df.filter((result_df["OutofShelter"] == True) & (result_df["all_permuted_angles"] == 1))
            angle_per_spike = filtered_df["angles"].to_numpy()
            # angle_per_spike = result_df["angles"][np.logical_and(OutofShelterIdx== True,all_permuted_angles == 1)]
            # compute rayleigh
            x = np.nanmean(np.cos(angle_per_spike))
            y = np.nanmean(np.sin(angle_per_spike))
            self.Rayleigh_theta[counter] = np.arctan(y/x)
            self.Rayleigh[counter] = np.sqrt(x**2 + y**2)
            self.Rayleigh_cluster[counter] = c
            # # bootstrap x times with variable shifts in time
            # x = 100
            # shift_dist = np.empty(x)
            # for bs in np.arange(len(shift_dist)): 
            #     shift = int(np.random.uniform(1,x))*self.Visualize.session.video.fps # temporal shift in video frames
            #     ang_roll = np.roll(angles,shift)
            #     angle_per_spike = np.zeros_like(frame_per_spike)
            #     keep_spike = np.zeros_like(frame_per_spike)
            #     for f in np.unique(frame_per_spike):
            #         angle_per_spike[frame_per_spike == f] = ang_roll[f-1]
            #         if np.logical_and(OutofShelterIdx[f-1] == True,all_permuted_angles[f-1] == 1): keep_spike[frame_per_spike == f] = 1
            #     angle_per_spike = angle_per_spike[keep_spike == 1]
            #     x = np.nanmean(np.cos(angle_per_spike))
            #     y = np.nanmean(np.sin(angle_per_spike))
            #     shift_dist[bs] = np.sqrt(x**2 + y**2)
            # if self.Rayleigh[counter] > np.percentile(shift_dist,95):
            #     self.Rayleigh_sig[counter] = 1

        # histogram of rayleighs
        plt.figure()
        plt.hist(self.Rayleigh,np.arange(0,1,.1))
        plt.hist(self.Rayleigh[self.Rayleigh_sig == 1],np.arange(0,1,.1))
        plt.xlabel('Rayleigh R')
        plt.ylabel('number of clusters')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_Rayleigh_vector_hist.png")
        if self.Visualize.settings.show_plots: plt.show()

    def tuning_heatmap(self,title,times,spikes,clusters,angles,OutofShelterIdx,end_time):
        """
        Mean firing of each cell at each HSA orientation as a heatmap in which they are sorted by HSA with greatest firing.
        It also computes rayleigh vectors (a circular vector sum) which gives us how oblong vs. round their tuning profile is. 
        Rayleigh's R close to zero = untuned, fires at all head directions
        Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation
        It makes a histogram of all Rayleigh vectors and remakes the heatmaps but splitting them up into high vs. low Rayleigh R
        """
        
        # bin the angles 
        ang_step = np.linspace(-np.pi,np.pi,24,endpoint = True)
        angles = np.digitize(angles,ang_step) # np.pi/10 determines the intervals

        # set up polar plots figure
        # set number of rows and calculate number of columns
        ncols = 10
        nrows = 5 # nclu // ncols + (nclu % ncols > 0)
        figg, axs = plt.subplots(nrows,ncols)
        figg.set_figwidth(30)
        figg.set_figheight(15)
        fnum = 1
        axs = axs.ravel()

        # firing per head/shelter angle for each cluster
        start = [0, int(np.round(len(angles[OutofShelterIdx])/2))] # for splitting up in first and second half
        end = [int(np.round(len(angles[OutofShelterIdx])/2)),int(len(angles[OutofShelterIdx]))]
        max_rate = np.zeros(shape = [len(np.unique(clusters))])
        anglesfiring_clu = np.empty(shape = [len(np.unique(clusters)),len(ang_step)-1,2])
        timepoints = np.arange(times[0]-1/(2*self.Visualize.session.video.fps), # start of timewindow
                               end_time+1/(2*self.Visualize.session.video.fps), # end of timewindow
                               1/self.Visualize.session.video.fps) # each time bin is 1 frame
        cc = ['green','red']
        for counter,c in enumerate(np.unique(clusters)):
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            ax = plt.subplot(nrows,ncols,1+counter-(nrows*ncols*(fnum-1)),projection = 'polar')
            # the firing rate is computed in bins that are centered on the occurrence of a camera frame
            srate,_ = np.histogram(spikes[clusters == c],timepoints)
            if len(srate)>len(OutofShelterIdx): srate = srate[:-1]
            srate = srate[OutofShelterIdx]
            srate = srate*self.Visualize.session.video.fps # make it Hz
            for i,s in enumerate(zip(start,end)):
                for ang in np.arange(1,len(np.linspace(-np.pi,np.pi,24,endpoint = True))):
                    anglesfiring_clu[counter,ang-1,i] = np.nanmean(srate[np.logical_and(angles[OutofShelterIdx] == ang,
                                                                                    np.logical_and(np.arange(len(srate))>=s[0],np.arange(len(srate))<=s[1]))])
                if len(np.where(np.isnan(anglesfiring_clu[counter,:,i]))[0]) < len(anglesfiring_clu[counter,:,i]): # if the whole thing is NaN
                    if s[0] == 0: max_rate[counter] = np.nanargmax(anglesfiring_clu[counter,:,i])
                    # make polar plots of first and second half
                    ax.bar(ang_step[:-1] + np.diff(ang_step[:2])/2, anglesfiring_clu[counter,:,i], width=(2*np.pi)/24, bottom=0.0, color=cc[i], alpha=0.5)
                anglesfiring_clu[counter,:,i] = anglesfiring_clu[counter,:,i]/np.nanmax(anglesfiring_clu[counter,:,i])
            if self.Rayleigh_sig[counter] == 1:
                ax.title.set_text('clu ' + str(c) + ' sig.' + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[counter],2)))
            else:
                ax.title.set_text('clu ' + str(c) + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[counter],2)))
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1,
                            counter == len(np.unique(clusters))-1):
                plt.tight_layout()
                plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_cluster_polar_plots_" + str(fnum) + ".png")
                if self.Visualize.settings.show_plots: plt.show()  
                plt.close()              
        
        _, axs = plt.subplots(1, 2)
        # heatmap of first half, sorted by angle with max firing
        axs[0].imshow(anglesfiring_clu[np.argsort(max_rate),:,0],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(np.unique(clusters))])
        axs[0].set_ylabel('cluster (sort on pref HSA)')
        axs[0].set_xlabel(title + ' (radians)')
        axs[0].title.set_text('first half')
        # heatmap of second half, sorted on first half
        axs[1].imshow(anglesfiring_clu[np.argsort(max_rate),:,1],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(np.unique(clusters))])
        axs[1].set_xlabel(title + ' (radians)')
        axs[1].title.set_text('second half')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_cluster_tuning.png")
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()

        # heatmap, but restricted to clusters with significant rayleigh vectors
        _, axs = plt.subplots(1, 2)
        # heatmap of first half, sorted by angle with max firing
        A = anglesfiring_clu[self.Rayleigh_sig == 1,:,:]
        M = max_rate[self.Rayleigh_sig == 1]
        axs[0].imshow(A[np.argsort(M),:,0],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(M)])
        axs[0].set_ylabel('cluster (sort on pref HSA)')
        axs[0].set_xlabel(title + ' (radians)')
        axs[0].title.set_text('first half')
        # heatmap of second half, sorted on first half
        axs[1].imshow(A[np.argsort(M),:,1],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(M)])
        axs[1].set_xlabel(title + ' (radians)')
        axs[1].title.set_text('second half')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_cluster_tuning_sig_Rayleigh.png")
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def PSTH_all_neurons(self, stim_type):
        """
        Plot the mean firing rate of all cells (ignore clusters) to each trial. For each trial, retrieve:
        - the onset frame of that stimulus
        - the duration of that stimulus
        """
        # Hyperparameters
        timeBeforeStim = 5 # seconds

        # plot a line of mean activity for each trial
        for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
            time1 = (onset_frames / self.Visualize.session.video.fps) - timeBeforeStim 
            time2 = (onset_frames / self.Visualize.session.video.fps) + stimulus_durations
            
            # Mask spikes that are within the time window
            idx = np.logical_and(self.aligned_spikes[:,0] > time1, self.aligned_spikes[:,0] < time2)
            assert len(idx) == self.aligned_spikes.shape[0], "idx and aligned_spikes are not the same length"
            
            # Bin the spikes
            mult = 10 # binsize for looking at data - 1/10 of a second so 100ms bins 
            binEdges = np.arange(time1, time2, 1 / mult)
            firingrate, _ = np.histogram(self.aligned_spikes[idx], binEdges)
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
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()

    def HSA_tuning(self):
        """
        Make heatmaps of each cell's firing at each HSA, for first and second half of recording, sorted on first half
        """
        spikes,clusters,angles,OutofShelterIdx,end_time = self.filter_spikes(self.Visualize.tracking_data['hdir_shelt'], self.Visualize.sheltertime)
        self.rayleigh_vector(spikes,clusters,angles,OutofShelterIdx,self.Visualize.sheltertime,end_time,'head_shelter_angle')
        
        self.tuning_heatmap('head_shelter_angle',
                            self.Visualize.sheltertime,
                            spikes,clusters,angles,OutofShelterIdx,end_time) # find times when the mouse was not inside the shelter

    def barrier_tuning(self):
        """
        Make heatmaps of each cell's firing at each HSA, for first and second half of recording, sorted on first half
        """
        for i in np.arange(2):
            spikes,clusters,angles,OutofShelterIdx,end_time = self.filter_spikes(self.Visualize.tracking_data['hdir_barrier'],
                                                                                 self.Visualize.barriertime)
            self.rayleigh_vector(spikes,clusters,angles,OutofShelterIdx,self.Visualize.barriertime,end_time,'head_barrier_angle'+str(i+1))
            self.tuning_heatmap('head_barrier_angle'+str(i+1),
                                self.Visualize.barriertime,
                                spikes,clusters,angles,OutofShelterIdx,end_time) # find times when the mouse was not inside the shelter

    def PSTH_single_neurons(self, stim_type):
        """
        Plot the mean firing rate of all cells (ignore clusters) to each trial. For each trial, retrieve:
        - the onset frame of that stimulus
        - the duration of that stimulus
        """
        # Hyperparameters
        timeBeforeStim = 5 # seconds
        
        # Select only the good neurons
        conditionGood = self.dataFrame['cluster_group'] == 'good'
        good_neurons_data_frame = self.dataFrame.filter(conditionGood)
        assert len(good_neurons_data_frame['spike_clusters'].unique()) == self.Visualize.session.efizzDataLoaded.num_of_good_units, "The number of good neurons is not the same as the number of good neurons in the data frame"
        
        # Set the number of rows and columns for the subplot grid
        n_rows = 10
        n_columns = 5
        counter = 0
        row = 0
        column = 0

        # Create a figure and an array of subplots with the specified grid size
        _, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(20, 10))
        
        # For each of the good neurons and then each of the trials 
        for neuron in good_neurons_data_frame['spike_clusters'].unique():
            
            filtered_data_frame = pl.DataFrame()
            
            for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
                
                # Find trial window time
                # Should this now be / by 30000
                time1 = (onset_frames / self.Visualize.session.video.fps) - timeBeforeStim 
                time2 = (onset_frames / self.Visualize.session.video.fps) + stimulus_durations
                
                # Find the spikes that are within the time window and assigned to that neuron
                # Conditions
                GreaterThan = good_neurons_data_frame['aligned_spike_times'] > time1
                LessThan = good_neurons_data_frame['aligned_spike_times'] < time2
                neuronOfInterest = good_neurons_data_frame['spike_clusters'] == neuron
                
                # Filter and stack
                filtered_data_frame = filtered_data_frame.vstack(good_neurons_data_frame.filter(GreaterThan & LessThan & neuronOfInterest))
            
            # Plot the PSTH
            # Bin the spikes
            mult = 10 # binsize for looking at data - 1/10 of a second so 100ms bins 
            binEdges = np.arange(time1, time2, 1 / mult)
            firingrate, _ = np.histogram(filtered_data_frame["aligned_spike_times"], binEdges)
            assert len(firingrate) == len(binEdges) - 1, "firingrate and binedges are not the same length"
            
            # Generate x values for plotting
            xValues = binEdges - time1 - timeBeforeStim
            assert xValues[0] == -timeBeforeStim, f"xValues[0] is not -{timeBeforeStim}"
            
            # Plot the PSTH
            # plt.plot(xValues[:-1], gaussian_filter1d(firingrate * mult, sigma = 1), label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second
            axes[row, column].plot(xValues[:-1], firingrate * mult, label = f"Neuron #: {neuron}") # because our bin size is 1/mult of a second
            axes[row, column].axvline(x = 0, color = 'k', linestyle = '-')
            axes[row, column].set_title(f'Neuron {neuron} PSTH')
            
             # Axes logic
            counter += 1 # increment counter don't plot more than 50 neurons
            if counter > n_columns * n_rows: # if counter is greater than 50, break
                break
            
            # Once you've plotted on all the rows, move to the next column
            row += 1
            if row > 9:
                column += 1
                row = 0
            
            # If you've reached the end of the columns, break
            if column > 4:
                break
                
        plt.tight_layout()
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()

    def spatial_position_firing(self):
        """ A function that makes maps of mousie's position in arena
        and show where each cluster fired"""
        
        cc = matplotlib.cm.Reds # could use Reds or copper
        # set number of rows and calculate number of columns
        ncols = 10
        nrows = 5 # nclu // ncols + (nclu % ncols > 0)
        figg, axs = plt.subplots(nrows,ncols)
        figg.set_figwidth(30)
        figg.set_figheight(15)
        fnum = 1
        axs = axs.ravel()

        mass = self.Visualize.tracking_data['avg_loc']
        # what is firing rate per frame?
        timepoints = np.arange(0-1/(2*self.Visualize.session.video.fps), # start of timewindow
                               (len(self.Visualize.tracking_data['avg_loc'])/self.Visualize.session.video.fps)+1/(2*self.Visualize.session.video.fps), # end of timewindow
                               1/self.Visualize.session.video.fps) # each time bin is 1 frame
        for counter,c in enumerate(np.unique(self.clu_spikes)):
            # the firing rate is computed in bins that are centered on the occurrence of a camera frame
            srate,_ = np.histogram(self.aligned_spikes[self.clu_spikes == c],timepoints)
            srate = srate*self.Visualize.session.video.fps # make it Hz
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            axs[counter-(nrows*ncols*fnum)].scatter(mass[:,0],mass[:,1],s=5,c=cc(srate*2),linewidths=0,marker='.') # srate*2 increase contrast
            axs[counter-(nrows*ncols*fnum)].set_axis_off()
            axs[counter-(nrows*ncols*fnum)].invert_yaxis()
            axs[counter-(nrows*ncols*fnum)].set_aspect('equal')
            axs[counter-(nrows*ncols*fnum)].title.set_text('cluster ' + str(c))
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1,
                             counter == len(np.unique(self.clu_spikes))-1):
                plt.tight_layout()
                plt.savefig(str(self.Visualize.session.file_path) + "/" + "spatial_position_firing_" + str(fnum) + ".png")
                if self.Visualize.settings.show_plots: plt.show()
                plt.close()

# -----------------------------------LASEER SYNC TEST-----------------------------------

    def extract_trial_spikes_for_laser_sync_test(self, time_before_stim = 0.5, time_after_stim = 1, select_good_neurons = False) -> dict:
        """
        A function that extracts spikes times between trials and aligns it to the stimulus onset.
        The function returns a dictionary with the cluster ID as the key and a list of polars dataframes as the value.
        Each polar dataframe contains the spikes for a single trial.
        
        Returns Dictionary = {cluster_ID1: [polar_dataframe_trial_1, polar_dataframe_trial_2, ...], 
                              cluster_ID2: [polar_dataframe_trial_1, polar_dataframe_trial_2, ...], 
                              ...}
                              
        """
        
        data = self.dataFrame
        
        dic = {}
        
        if select_good_neurons:
            iterator = self.array_of_good_neurons_IDs
        
        elif not select_good_neurons:
            iterator = data["spike_clusters"].unique()
            
        for index, cluster in enumerate(iterator):
                        
            spikes = data.filter(data["spike_clusters"] == cluster)
            spikesByTrial = []
            
            counter = 0
            for trial, onset in enumerate(self.Visualize.session.laser_sync.laser_onsets):
                time1 = onset / 30000 - time_before_stim
                time2 = onset / 30000 + time_after_stim
                trialSpikesPolars = spikes.filter((spikes["aligned_spike_times"] > time1) & (spikes["aligned_spike_times"] < time2))
                adjustedSpikeTimes = trialSpikesPolars.with_column(trialSpikesPolars['aligned_spike_times'] - onset / 30000)
                spikesByTrial.append(adjustedSpikeTimes)
            
            dic[cluster] = spikesByTrial
               
        return dic
    
    def single_cluster_raster_Laser_test(self):
        """
        A function that extracts spike times for each good cluster and aligns it to trials as a raster plot
        """
        # Load data - Choose whether to use good neurons or all neurons
        cluster_trial_spikes_dic = self.extract_trial_spikes_for_laser_sync_test(select_good_neurons = False)
        
        # How many plots do we need?
        number_of_plots = len(cluster_trial_spikes_dic)
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
                        cluster = list(cluster_trial_spikes_dic.keys())[plot_counter]
                        
                        for trial, onset_frames in enumerate(self.Visualize.session.laser_sync.laser_onsets):
                            x_values = cluster_trial_spikes_dic[cluster][trial]["aligned_spike_times"]
                            y_values = [trial] * len(x_values)
                            axes[rows, columns].scatter(x_values, y_values, marker='|', s=10, c='b')
                        axes[rows, columns].set_title(f"Cluster: {cluster}")
                        axes[rows, columns].vlines(0, 0, len(self.Visualize.session.laser_sync.laser_onsets), colors='r', linestyles='solid')
                    
                    # Remove the extra axes if there are no more plots
                    else:
                        fig.delaxes(axes[rows, columns])
                    
                    plot_counter += 1
            
            # SAVE FIGURE
            fig.tight_layout()
            plt.savefig(str(self.Visualize.session.file_path) + "/" + "_single_cluster_raster_" + str(figure_idx) + ".png")                
        
        if self.Visualize.settings.show_plots: 
            plt.show()
        
        pass
        plt.close()
        
# ------------------------ Break out graph functions
