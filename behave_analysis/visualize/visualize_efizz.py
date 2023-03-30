# OS libaries
import numpy as np
from glob import glob
import polars as pl
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

class Visualize_efizz():
    """
    A class for some sanity check efizz plots using kilosort clusters
    """
    def __init__(self, Visualize):
        self.Visualize = Visualize
        # load data
        self.load_spike_data()

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
        if self.Visualize.settings.show_plots: plt.show()

    def single_cluster_raster(self,stim_type):
        """
        A function that extracts spike times for each goo cluster and aligns it to trials as a raster plot
        """
        nclu = len(np.unique(self.clu_spikes))
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.2)
        # set number of rows and calculate number of columns
        ncols = 6
        nrows = nclu // ncols + (nclu % ncols > 0)
        timeBeforeStim = 5 # in seconds

        for i,c in enumerate(np.unique(self.clu_spikes)):
            ax = plt.subplot(nrows, ncols, i + 1)
            spikes = self.aligned_spikes[self.clu_spikes == c]
            trial_idx = np.zeros_like(spikes)
            spikes_idx = np.zeros_like(spikes)
            for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
                t1 = (onset_frames/self.Visualize.session.video.fps) - timeBeforeStim
                t2 = (onset_frames/self.Visualize.session.video.fps) + stimulus_durations
                trial_idx[np.logical_and(spikes > t1,spikes < t2)] = trial_num + 1
                spikes_idx[np.logical_and(spikes > t1,spikes < t2)] = spikes[np.logical_and(spikes > t1,spikes < t2)] - (onset_frames/self.Visualize.session.video.fps)
            ax.plot([0,0],[0, np.amax(trial_idx)],'r-')
            if np.sum(trial_idx.astype(int)) > 0:
                ax.eventplot(np.array([spikes_idx[trial_idx > 0]]).T,lineoffsets = np.array(trial_idx[trial_idx > 0]))
                ax.set_xlabel('time from stim (s)')
                ax.set_ylabel('trials')
            ax.title.set_text('cluster ' + str(c))
        plt.tight_layout()
        plt.savefig(str(self.Visualize.session.file_path) + "/" + str(stim_type) + "_single_cluster_raster.png")
        if self.Visualize.settings.show_plots: plt.show()

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

    def HSA_tuning(self):
        """
        Make heatmaps of each cell's firing at each HSA, for first and second half of recording, sorted on first half
        """
        self.tuning_heatmap(angles = self.Visualize.tracking_data['hdir_shelt'],
                            title = 'head_shelter_angle',
                            times = self.Visualize.sheltertime) # find times when the mouse was not inside the shelter

    def barrier_tuning(self):
        """
        Make heatmaps of each cell's firing at each HSA, for first and second half of recording, sorted on first half
        """
        for i in np.arange(2):
            self.tuning_heatmap(self.Visualize.tracking_data['hdir_barrier'][:,i],
                                'head_barrier_angle'+str(i+1),
                                self.Visualize.barriertime) # find times when the mouse was not inside the shelter

    def tuning_heatmap(self,angles,title,times):
        """
        Mean firing of each cell at each HSA orientation as a heatmap in which they are sorted by HSA with greatest firing.
        It also computes rayleigh vectors (a circular vector sum) which gives us how oblong vs. round their tuning profile is. 
        Rayleigh's R close to zero = untuned, fires at all head directions
        Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation
        It makes a histogram of all Rayleigh vectors and remakes the heatmaps but splitting them up into high vs. low Rayleigh R
        """
        
        # bin the angles 
        angles = np.digitize(angles,np.arange(-np.pi,np.pi,np.pi/10)) # np.pi/10 determines the intervals
        # find times when the mouse was not inside the shelter
        OutofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0],
            self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]),
            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1],
            self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1])))
       
        # only look at tuning when shelter (and no barrier) was in arena
        if times[1] == -60: # -1 times 60 hehe
            idx = self.aligned_spikes > times[0]
            end_time = len(angles)/self.Visualize.session.video.fps
            angles = angles[times[0]*self.Visualize.session.video.fps:-1]
            OutofShelterIdx = OutofShelterIdx[times[0]*self.Visualize.session.video.fps:-1]
        else:
            idx = np.logical_and(self.aligned_spikes > times[0],self.aligned_spikes < times[1])
            end_time = times[1]
            angles = angles[times[0]*self.Visualize.session.video.fps:times[1]*self.Visualize.session.video.fps]
            OutofShelterIdx = OutofShelterIdx[times[0]*self.Visualize.session.video.fps:times[1]*self.Visualize.session.video.fps]
        spikes = self.aligned_spikes[idx]
        clusters = self.clu_spikes[idx[:,0]]

        # firing per head/shelter angle for each cluster
        start = [0, int(np.round(len(angles[OutofShelterIdx])/2))]
        end = [int(np.round(len(angles[OutofShelterIdx])/2)),int(len(angles[OutofShelterIdx]))]
        max_rate = np.zeros(shape = [len(np.unique(clusters))])
        anglesfiring_clu = np.empty(shape = [len(np.unique(clusters)),len(np.arange(-np.pi,np.pi,np.pi/10)),2])
        timepoints = np.arange(times[0]-1/(2*self.Visualize.session.video.fps), # start of timewindow
                               end_time+1/(2*self.Visualize.session.video.fps), # end of timewindow
                               1/self.Visualize.session.video.fps) # each time bin is 1 frame
        for counter,c in enumerate(np.unique(clusters)):
            # the firing rate is computed in bins that are centered on the occurrence of a camera frame
            srate,_ = np.histogram(spikes[clusters == c],timepoints)
            if len(srate)>len(OutofShelterIdx): srate = srate[:-1]
            srate = srate[OutofShelterIdx]
            for i,s in enumerate(zip(start,end)):
                for ang in np.arange(1,len(np.arange(-np.pi,np.pi,np.pi/10))-1):
                    anglesfiring_clu[counter,ang,i] = np.nanmean(srate[np.logical_and(angles[OutofShelterIdx] == ang,
                                                                                    np.logical_and(np.arange(len(srate))>=s[0],np.arange(len(srate))<=s[1]))])
                anglesfiring_clu[counter,:,i] = anglesfiring_clu[counter,:,i]/np.nanmax(anglesfiring_clu[counter,:,i])
                if np.logical_and(s[0] == 0,len(np.where(np.isnan(anglesfiring_clu[counter,:,i]))[0])<len(anglesfiring_clu[counter,:,i])): max_rate[counter] = np.nanargmax(anglesfiring_clu[counter,:,i])
        
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
        
    def load_spike_data(self):
        """
        Loads the csv of aligned data
        """
        csv_path = glob(os.path.join(self.Visualize.session.file_path, "Processed_efizz_data"))[0]
        self.dataFrame = pl.read_csv(csv_path)
        
        aligned_spike_data = pl.read_csv(csv_path,has_header=True)
        asd_np = aligned_spike_data.to_numpy()
        # filter by 'good' clusters
        self.aligned_spikes = np.array([asd_np[asd_np[:,2] == 'good',3]]).T
        self.clu_spikes = asd_np[asd_np[:,2] == 'good',1]

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

    def spatial_position_firing(self):
        """ A function that makes maps of mousie's position in arena
        and show where each cluster fired"""
        cc = matplotlib.cm.Reds # could use Reds or copper
        nclu = len(np.unique(self.clu_spikes))
        plt.figure(figsize=(30, 12))
        plt.subplots_adjust(hspace=0.2)
        # set number of rows and calculate number of columns
        ncols = 10
        nrows = nclu // ncols + (nclu % ncols > 0)

        mass = self.Visualize.tracking_data['avg_loc']
        # what is firing rate per frame?
        timepoints = np.arange(0-1/(2*self.Visualize.session.video.fps), # start of timewindow
                               (len(self.Visualize.tracking_data['avg_loc'])/self.Visualize.session.video.fps)+1/(2*self.Visualize.session.video.fps), # end of timewindow
                               1/self.Visualize.session.video.fps) # each time bin is 1 frame
        for counter,c in enumerate(np.unique(self.clu_spikes)):
            # the firing rate is computed in bins that are centered on the occurrence of a camera frame
            srate,_ = np.histogram(self.aligned_spikes[self.clu_spikes == c],timepoints)
            srate = srate*self.Visualize.session.video.fps # make it Hz
            ax = plt.subplot(nrows, ncols, counter + 1)
            ax.scatter(mass[:,0],mass[:,1],s=5,c=cc(srate*2),linewidths=0,marker='.') # srate*2 increase contrast
            ax.set_axis_off()
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.title.set_text('cluster ' + str(c))
        plt.tight_layout()
        plt.savefig(str(self.Visualize.session.file_path) + "/" + "spatial_position_firing.png")
        if self.Visualize.settings.show_plots: plt.show()


    # bin spike times of all neurons by each frame (1/40 fps), then assign HD
        # spike_by_frame = np.digitize(spikes, np.arange(0,len(hsa)/fps,1/fps))
        # HD_at_spike = np.zeros_like(spike_by_frame)
        # for t in np.arange(len(hsa)):
        #     HD_at_spike[spike_by_frame == t] = hsa[t]

        # # by cluster calculate rayleigh
        # r = np.zeros(len(clu))
        # for counter,c in enumerate(clu):
        #     alpha = HD_at_spike[c_spikes==c]
        #     alpha = np.asarray(alpha)
        #     w = np.ones_like(alpha)
        #     cmean = np.sum((w * np.exp(1j * alpha)) / np.sum(w))
        #     r[counter] = np.abs(cmean)
        # _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # # histogram of rayleighs
        # ax1.hist(r,np.arange(0,1,.1))
        # ax1.set(xlabel='Rayleigh R', ylabel='number of clusters')
        # # heatmap of high rayleighs sorted by pref HSA
        # thresh = .5
        # idx = r > thresh
        # hfc = hsafiring_clu[idx,:]
        # mr = max_rate[idx]
        # ax2.imshow(hfc[np.argsort(mr),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(hfc)])
        # ax2.set_title('HSA tuned cells')
        # ax2.set(xlabel='head-shelter ang (radians)', ylabel='cluster (sort on pref HSA)')
        # # heatmap of low rayleighs sorted by pref HSA
        # idx = r < thresh
        # hfc = hsafiring_clu[idx,:]
        # mr = max_rate[idx]
        # ax3.imshow(hfc[np.argsort(mr),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(hfc)])
        # ax3.set_title('Untuned cells')
        # ax3.set(xlabel='head-shelter ang (radians)', ylabel='cluster (sort on pref HSA)')
        # plt.show()