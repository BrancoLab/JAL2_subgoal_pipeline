# Custom classes
from behave_analysis.utils.open_tracking_data import open_tracking_data

# OS libaries
from scipy.ndimage import gaussian_filter1d
import numpy as np
from glob import glob
import polars as pl
import os
import dill as pickle
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

        for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            fps = 40
            t1 = (onset_frames/fps)-5 # from 5s before onset
            t2 = (onset_frames/fps)+stimulus_durations
            idx = np.logical_and(self.aligned_spikes[:,0] > t1,self.aligned_spikes[:,0] < t2)
            ax.eventplot(self.aligned_spikes[idx], lineoffsets = self.clu_spikes[idx])
            ax.plot([onset_frames/fps,onset_frames/fps],[0, np.amax(self.clu_spikes)],'r-')
        plt.show()

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
        plt.show()

    def HSA_tuning(self):
        """
        Mean firing of each cell at each HSA orientation as a heatmap in which they are sorted by HSA with greatest firing.
        It also computes rayleigh vectors (a circular vector sum) which gives us how oblong vs. round their tuning profile is. 
        Rayleigh's R close to zero = untuned, fires at all head directions
        Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation
        It makes a histogram of all Rayleigh vectors and remakes the heatmaps but splitting them up into high vs. low Rayleigh R
        """
        hsa = self.Visualize.tracking_data['hdir_shelt']
        fps = 40 # camera frame rate
        # only look at tuning after shelter was placed in arena
        print("When (in minutes) was the start and end of the period of shelter-only? (use -1 for end of session)")
        x, y = map(int, input().split())
        # sheltertime = input("When (in minutes) was the shelter placed in the arena?")
        sheltertime = int(x)*60 # in seconds
        if y == -1:
            shelterend = y
            idx = self.aligned_spikes > sheltertime
        else:
            shelterend = int(y)*60
            idx = np.logical_and(self.aligned_spikes > sheltertime,self.aligned_spikes < shelterend)
        hsa = np.digitize(hsa,np.arange(-np.pi,np.pi,np.pi/10)) # np.pi/10 determines the intervals
        hsa = hsa[(sheltertime*fps):(shelterend*fps)]
        clu = np.unique(self.clu_spikes)
        hsafiring_clu = np.empty(shape = [len(clu),len(np.arange(-np.pi,np.pi,np.pi/10))])
        max_rate = np.empty(shape = [len(clu)])
        spikes = self.aligned_spikes[idx]
        clu_spikes = self.clu_spikes[idx[:,0]]
        counter = 0
        for c in clu:
            s = spikes[clu_spikes==c]
            # the firing rate is computed in bins that are centered on the occurrence of a camera frame
            srate,_ = np.histogram(s,np.arange(0-1/(2*fps),(len(hsa)/fps)+1/(2*fps),1/fps))
            for ang in np.arange(1,len(np.arange(-np.pi,np.pi,np.pi/10))-1):
                # print(c)
                # print(ang)
                hsafiring_clu[counter,ang] = np.mean(srate[hsa == ang])
            max_rate[counter] = np.nanargmax(hsafiring_clu[counter,:])
            hsafiring_clu[counter,:] = hsafiring_clu[counter,:]/np.nanmax(hsafiring_clu[counter,:])
            counter = counter+1
        plt.imshow(hsafiring_clu[np.argsort(max_rate),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(clu)])
        plt.xlabel('head-shelter ang (radians)')
        plt.ylabel('cluster (sort on pref HSA)')
        plt.show()

        # bin spike times of all neurons by each frame (1/40 fps), then assign HD
        spike_by_frame = np.digitize(spikes, np.arange(0,len(hsa)/fps,1/fps))
        HD_at_spike = np.zeros_like(spike_by_frame)
        for t in np.arange(len(hsa)):
            HD_at_spike[spike_by_frame == t] = hsa[t]
        # by cluster calculate rayleigh
        r = np.zeros(len(clu))
        counter = 0
        for c in clu:
            alpha = HD_at_spike[clu_spikes==c]
            alpha = np.asarray(alpha)
            w = np.ones_like(alpha)
            cmean = np.sum((w * np.exp(1j * alpha)) / np.sum(w))
            r[counter] = np.abs(cmean)
            counter = counter+1
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # histogram of rayleighs
        ax1.hist(r,np.arange(0,1,.1))
        ax1.set(xlabel='Rayleigh R', ylabel='number of clusters')
        # heatmap of high rayleighs sorted by pref HSA
        thresh = .5
        idx = r > thresh
        hfc = hsafiring_clu[idx,:]
        mr = max_rate[idx]
        ax2.imshow(hfc[np.argsort(mr),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(hfc)])
        ax2.set_title('HSA tuned cells')
        ax2.set(xlabel='head-shelter ang (radians)', ylabel='cluster (sort on pref HSA)')
        # heatmap of low rayleighs sorted by pref HSA
        idx = r < thresh
        hfc = hsafiring_clu[idx,:]
        mr = max_rate[idx]
        ax3.imshow(hfc[np.argsort(mr),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(hfc)])
        ax3.set_title('Untuned cells')
        ax3.set(xlabel='head-shelter ang (radians)', ylabel='cluster (sort on pref HSA)')
        plt.show()
        
    def load_spike_data(self):
        """
        Loads the csv of aligned data
        """
        csv_path = glob(os.path.join(self.Visualize.session.file_path, "Processed_efizz_data"))[0]
        self.dataFrame = pl.read_csv(csv_path)
        
        aligned_spike_data = pl.read_csv(csv_path,has_header=True)
        asd_np = aligned_spike_data.to_numpy()
        self.aligned_spikes = np.array([asd_np[:,2]]).T
        self.clu_spikes = asd_np[:,1]

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
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(20, 10))
        
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
        plt.show()