# Custom classes
from behave_analysis.utils.open_tracking_data import open_tracking_data

# OS libaries
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
            idx = np.logical_and(self.aligned_spikes[:,0]>t1,self.aligned_spikes[:,0]<t2)
            ax.eventplot(self.aligned_spikes[idx], lineoffsets=self.clu_spikes[idx])
            ax.plot([onset_frames/fps,onset_frames/fps],[0, np.amax(self.clu_spikes)],'r-')
        plt.show()

    def PSTH(self,stim_type):
        """
        Plot the mean firing rate of all cells to each trial"""

        # plot a line of mean activity for each trial
        for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.Visualize.session.__dict__[stim_type].onset_frames, self.Visualize.session.__dict__[stim_type].stimulus_durations)):
            fps = 40 # camera frame rate
            t1 = (onset_frames/fps)-5 # from 5s before onset
            t2 = (onset_frames/fps)+stimulus_durations
            idx = np.logical_and(self.aligned_spikes[:,0]>t1,self.aligned_spikes[:,0]<t2)
            mult = 10 # binsize for looking at data
            binedges = np.arange(t1,t2,1/mult)
            firingrate,_ = np.histogram(self.aligned_spikes[idx],binedges)
            xval = np.arange(-5,stimulus_durations,1/mult)
            xval = xval[:-1]+1/(2*mult)
            plt.plot(xval,firingrate*mult) # because our bin size is 1/mult of a second
            # line is at 5*mult because 5 seconds before trial onset, mult timepoints per second
            plt.plot([0,0],[0, np.amax(firingrate*mult)],'r-')
            plt.title('stimulus duration = ' + str(stimulus_durations))
            plt.ylabel('cumulative firing rate (Hz)')
            plt.xlabel('time (s)')
        plt.show()

    def HSA_tuning(self):
        """
        Mean firing of each cell at each HSA orientation as a heatmap"""
        hsa = self.Visualize.tracking_data['hdir_shelt']
        hsa = np.digitize(hsa,np.arange(-np.pi,np.pi,np.pi/10)) # np.pi/10 determines the intervals
        #hsa2 = hsa[]
        fps = 40 # camera frame rate
        clu = np.unique(self.clu_spikes)
        hsafiring_clu = np.empty(shape = [len(clu),len(np.arange(-np.pi,np.pi,np.pi/10))])
        max_rate = np.empty(shape = [len(clu)])
        counter = 0
        for c in clu:
            s = self.aligned_spikes[self.clu_spikes==c]
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

        # bin spike times of all neurons by 1/40, then assign HD
        spike_by_frame = np.digitize(self.aligned_spikes, np.arange(0,len(hsa)/fps,1/fps))
        print(np.max(spike_by_frame))
        HD_at_spike = np.zeros_like(spike_by_frame)
        for t in np.arange(len(hsa)):
            HD_at_spike[spike_by_frame == t] = hsa[t]
        # by cluster calculate rayleigh
        r = np.zeros(len(clu))
        counter = 0
        for c in clu:
            alpha = HD_at_spike[self.clu_spikes==c]
            alpha = np.asarray(alpha)
            w = np.ones_like(alpha)
            cmean = np.sum((w * np.exp(1j * alpha)) / np.sum(w))
            r[counter] = np.abs(cmean)
            counter = counter+1
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # histogram of rayleighs
        ax1.hist(r,np.arange(0,1,.1))
        # heatmap of high rayleighs sorted by pref HSA
        idx = r > .3
        hfc = hsafiring_clu[idx,:]
        mr = max_rate[idx]
        plt.imshow(hfc[np.argsort(mr),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(hfc)])
        plt.xlabel('head-shelter ang (radians)')
        plt.ylabel('cluster (sort on pref HSA)')
        plt.title('HSA tuned cells')
        # heatmap of low rayleighs sorted by pref HSA
        idx = r < .3
        hfc = hsafiring_clu[idx,:]
        mr = max_rate[idx]
        plt.imshow(hfc[np.argsort(mr),:],cmap = 'hot',aspect = .05,extent = [-np.pi,np.pi,0,len(hfc)])
        plt.xlabel('head-shelter ang (radians)')
        plt.ylabel('cluster (sort on pref HSA)')
        plt.title('HSA tuned cells')
        plt.show()
        
    def load_spike_data(self):
        """
        Loads the csv of aligned data
        """
        csv_path = glob(os.path.join(self.Visualize.session.file_path, "Processed_efizz_data"))
        aligned_spike_data = pl.read_csv(csv_path[0],has_header=True)
        asd_np = aligned_spike_data.to_numpy()
        self.aligned_spikes = np.array([asd_np[:,2]]).T
        self.clu_spikes = asd_np[:,1]
