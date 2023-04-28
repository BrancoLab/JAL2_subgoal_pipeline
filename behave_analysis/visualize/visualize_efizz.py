# OS libaries
from loguru import logger
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
    def __init__(self, Visualize, run = "Production", select_good_neurons = True):
        self.Visualize = Visualize
        self.run_type = run
        self.select_good_neurons = select_good_neurons
        # load data
        self.load_spike_data()
        self.process_spike_data()
        self.track_to_polars()

# INIT FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def load_spike_data(self):
        """
        Loads the csv of aligned data
        """
        if self.run_type == "Production":
            self.csv_path = glob(os.path.join(self.Visualize.session.file_path, "Processed_efizz_data"))[0]
        
        elif self.run_type == "Test":
            logger.warning("Synethic spike data is being used when visualizing efizz - Real positional data is used from databank")
            self.csv_path = r"C:\Users\laurence\Documents\JAL-pipeline\behave_analysis\database\synthetic_data\synthetic_dataframe.csv"
    
        else: 
            raise ValueError("Run type not recognised")

    def process_spike_data(self):
        self.dataFrame = pl.read_csv(self.csv_path)

        # New ingestion for speed
        dataFrame = pl.read_csv(self.csv_path)

        if self.select_good_neurons:
            self.spikedataframe = dataFrame.filter(dataFrame['cluster_group'] == 'good')
        else:
            self.spikedataframe = dataFrame

        # self.dataFrame_filt_on_good_neurons = self.dataFrame.filter(self.dataFrame['cluster_group'] == 'good')
        # self.array_of_good_neurons_IDs = self.dataFrame_filt_on_good_neurons["spike_clusters"].unique()
        
        # # Old code leaving incase it breaks anything - Ideally we should be using the above code utilizing polars and not numpy for speed
        # aligned_spike_data = pl.read_csv(self.csv_path, has_header=True)
    
        # # Hard code for one neuron TODO remove
        # # aligned_spike_data = aligned_spike_data.filter(aligned_spike_data['spike_clusters'] == 3)

        # asd_np = aligned_spike_data.to_numpy() # What is asd? Is that aligned spike data?
        # # self.aligned_spikes = aligned_spike_data.get_column("aligned_spike_times").to_numpy()
        
        # # filter by 'good' clusters
        # self.aligned_spikes = np.array([asd_np[asd_np[:,2] == 'good', 3]]).T # This says for every row select the 3rd column if it's good
        # self.clu_spikes = asd_np[asd_np[:,2] == 'good',1]
        print("Loaded spike data")

    def track_to_polars(self):
        """
        Adds all the behavioral variables from track to the polars psike dataframe
        """
        OutofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0],
            self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]),
            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1],
            self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1])))
        
        if len(self.Visualize.session.shelter_time) > 0: # is there a time with shelter only?
            if self.Visualize.session.shelter_time[1] == -1: # shelter only until the end of the session
                shelteronly = np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps)
            else:
                shelteronly = np.logical_and(np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps),
                                             np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) < (self.Visualize.sheltertime[1]*self.Visualize.session.video.fps))
        else:
            shelteronly = np.zeros(len(OutofShelterIdx)) == 1

        if len(self.Visualize.session.barrier_time) > 0: # is there a time with shelter only?
            if self.Visualize.session.barrier_time[1] == -1: # shelter only until the end of the session
                barrier_present = np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps)
            else:
                barrier_present = np.logical_and(np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps),
                                             np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) < (self.Visualize.sheltertime[1]*self.Visualize.session.video.fps))
        else:
            barrier_present = np.zeros(len(OutofShelterIdx)) == 1

        # make a video dataframe where for each video frame:
        self.Video_df = pl.DataFrame(
                {"frames": np.arange(1,len(self.Visualize.tracking_data['hdir'])+1).astype(np.int64),
                "hdir": self.Visualize.tracking_data['hdir'],
                "hsa": self.Visualize.tracking_data['hdir_shelt'],
                "h_bar_north_a": self.Visualize.tracking_data['hdir_barrier'][:,0],
                "h_bar_south_a": self.Visualize.tracking_data['hdir_barrier'][:,1],
                "OutofshelterIdx": OutofShelterIdx, # was the mouse in the shelter?
                "shelter_only": shelteronly, # was this in a shelter only period? or was there a barrier?
                "barrier_present": barrier_present,}) # was this in a barrier period? or was there a barrier?

    def HSA_tuning(self):
        """
        Make heatmaps of each cell's firing at each HSA, for first and second half of recording, sorted on first half
        """
        self.rayleigh_vector(which_angle = 'head_shelter_angle')
        logger.info(f"Finished calculating Rayleigh vectors, moving on to polar plots")
        self.polar_plots(which_angle = 'head_shelter_angle') 

    def HD_tuning(self):
        """
        Make heatmaps of each cell's firing at each HSA, for first and second half of recording, sorted on first half
        """
        self.rayleigh_vector(which_angle = 'hdir')
        logger.info(f"Finished calculating Rayleigh vectors, moving on to polar plots")
        self.polar_plots(which_angle = 'hdir') 

    def rayleigh_vector(self,which_angle):
        """A function that calculates the Rayleigh vector (amplitude and angle) for each cluster with respect to the angles given (e.g. HD or HSA)
        It subsamples angles within 20 degree bins to ensure that angles are more uniformly sampled
        It only considers times when the mouse was outside the shelter
        It also performs bootstrapping by computing the rayleigh vector at random time shifts of the spikes with respect to the angles
        The Rayleigh vector is significant if the amplitude is above the 95th percentile of boostrapped amplitudes"""
        logger.info("Calculating Rayleigh vectors")
        
        # subselect frames of interest:
        # 1. mouse has to be outside shelter
        # 2. for hdir take all time, for hsa take times when only a shelter was present in the arena, for hba take times when barrier was present
        # TODO 3. exclude threat stimuli times and the escape
        if which_angle == 'head_shelter_angle':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) & (self.Video_df["shelter_only"] == True))
            angle_filt = 'hsa'
        elif which_angle == 'head_south_barrier_angle':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) & (self.Video_df["barrier_present"] == True))
            angle_filt = 'h_bar_south_a'
        elif which_angle == 'head_north_barrier_angle':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) & (self.Video_df["barrier_present"] == True))
            angle_filt = 'h_bar_north_a'
        elif which_angle == 'hdir':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True))
            angle_filt = 'hdir'

        # edges for binning firing rate at different angles
        bin_angles = np.linspace(-np.pi,np.pi,19)
        bin_angle_center = np.sort(np.append([-np.pi,np.pi], [bin_angles[:-1] + (np.mean(np.diff(bin_angles))/2)]))

        # initialize variables
        number_of_clusters = self.spikedataframe["spike_clusters"].unique()
        self.Rayleigh_theta = np.empty([len(number_of_clusters)]) # preferred angle
        self.Rayleigh = np.empty([len(number_of_clusters)]) # amplitude of Rayleigh vector
        self.Rayleigh_sig = np.zeros([len(number_of_clusters)]) # is the Ryleigh significant?
        self.Rayleigh_cluster = np.empty([len(number_of_clusters)]) # which cluster ID is this Rayleigh value for?

        # assign spike times of each cluster to the corresponding video frame, then assign HD
        for counter,c in enumerate(number_of_clusters):
            # filter by cluster
            spikes = self.spikedataframe.filter(self.spikedataframe['spike_clusters'] == c)
            # count number of spikes on each video frame, and then turn it into firing rate (Hz)
            spikes = spikes.groupby("spike_aligned_to_frame").agg([pl.count("spike_aligned_to_frame").alias("spike_count")])
            spikes = spikes.with_columns(pl.col('spike_count')*self.Visualize.session.video.fps)
            # align spike dataframe to video dataframe
            filtered_video_df = filtered_video_df.select([pl.col('frames').apply(float),pl.exclude('frames')])
            spike_to_video_df = filtered_video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="left")
            if spike_to_video_df.select(pl.col('spike_count').is_null().sum()).item() == len(spike_to_video_df):
                logger.info(f"Cluster {c} had no spikes")
                break
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
            x = np.sum(np.cos(all_angles_firing['all_angles'].to_numpy())*(all_angles_firing['mean_firing_rate'].to_numpy()))/np.sum(all_angles_firing['mean_firing_rate'].to_numpy())
            y = np.sum(np.sin(all_angles_firing['all_angles'].to_numpy())*(all_angles_firing['mean_firing_rate'].to_numpy()))/np.sum(all_angles_firing['mean_firing_rate'].to_numpy())
            self.Rayleigh_theta[counter] = np.arctan(y/x)
            self.Rayleigh[counter] = np.sqrt(x**2 + y**2)
            self.Rayleigh_cluster[counter] = c
            # bootstrap x times with variable shifts in time
            # x = 100
            # shift_dist = np.empty(x)
            # for it in np.arange(len(shift_dist)): 
            #     # shuffled shifts performed at a random offset between 0 and 100 seconds
            #     shift = int(np.random.uniform(1,100))*self.Visualize.session.video.fps # temporal shift in video frames
            #     angles = filtered_video_df[angle_filt].to_numpy()
            #     ang_roll = np.roll(angles,shift)
            #     rolled_filtered_video_df = filtered_video_df.select(pl.col('*'),pl.Series(name="rolled_angles", values = ang_roll))
            #     # align spike dataframe to video dataframe
            #     spike_to_video_df = rolled_filtered_video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="left")
            #     # calculate firing rates in angle bins
            #     spike_to_video_df = spike_to_video_df.sort('rolled_angles') # polars can be annoying, when using cut it doesn't preserve order :/
            #     spike_to_video_df = spike_to_video_df.with_columns(spike_to_video_df['rolled_angles'].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
            #     spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
            #     spike_to_video_df = spike_to_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) # TODO add this line to rayleigh v function
            #     angles_firing = (spike_to_video_df.groupby(by ='binned_angles').agg(pl.col('spike_count').mean().alias('mean_firing_rate')))            
            #     angles_firing = angles_firing.sort('binned_angles')
            #     # make sure that if any angles returned empty sets of spikes, they are registered as zeros and are not missing
            #     all_angles_firing = pl.DataFrame({'all_angles': bin_angle_center[1:-1]})
            #     all_angles_firing = all_angles_firing.join(angles_firing, left_on="all_angles", right_on="binned_angles", how="left")
            #     all_angles_firing = all_angles_firing.fill_null(strategy="zero")
            #     # compute rayleigh
            #     x = np.sum(np.cos(all_angles_firing['all_angles'].apply(float).to_numpy())*(all_angles_firing['mean_firing_rate'].to_numpy()))/np.sum(all_angles_firing['mean_firing_rate'].to_numpy())
            #     y = np.sum(np.sin(all_angles_firing['all_angles'].apply(float).to_numpy())*(all_angles_firing['mean_firing_rate'].to_numpy()))/np.sum(all_angles_firing['mean_firing_rate'].to_numpy())
            #     # add to distribution of rayleigh vectors with shift
            #     shift_dist[it] = np.sqrt(x**2 + y**2)
            # # significance logical
            # if self.Rayleigh[counter] > np.percentile(shift_dist,95):
            #     self.Rayleigh_sig[counter] = 1
            #     print('yay!')

        # histogram of rayleighs
        plt.figure()
        plt.hist(self.Rayleigh,np.arange(0,1,.1))
        plt.hist(self.Rayleigh[self.Rayleigh_sig == 1],np.arange(0,1,.1))
        plt.xlabel('Rayleigh R')
        plt.ylabel('number of clusters')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + str(which_angle) + "_Rayleigh_vector_hist.png")
        if self.Visualize.settings.show_plots: plt.show()

    def polar_plots(self,which_angle):
        """
        Mean firing of each cell at each HSA orientation as a heatmap in which they are sorted by HSA with greatest firing.
        It also computes rayleigh vectors (a circular vector sum) which gives us how oblong vs. round their tuning profile is. 
        Rayleigh's R close to zero = untuned, fires at all head directions
        Rayleigh's R close to 1 = very tuned, fires only when head is in one orientation
        It makes a histogram of all Rayleigh vectors and remakes the heatmaps but splitting them up into high vs. low Rayleigh R
        """
        # ---------------------------------------------------
        
        # bin the angles 
        number_of_bins = 19
        bin_angles = np.linspace(-np.pi,np.pi,number_of_bins)
        bin_angle_center = np.sort(np.append([-np.pi,np.pi], [bin_angles[:-1] + (np.mean(np.diff(bin_angles))/2)]))

        # set up polar plots figure
        # set number of rows and calculate number of columns
        ncols = 10
        nrows = 5 # nclu // ncols + (nclu % ncols > 0)
        figg, axs = plt.subplots(nrows,ncols)
        figg.set_figwidth(30)
        figg.set_figheight(15)
        fnum = 1
        axs = axs.ravel()

        # subselect frames of interest:
        # 1. mouse has to be outside shelter
        # 2. for hdir take all time, for hsa take times when only a shelter was present in the arena, for hba take times when barrier was present
        # TODO 3. exclude threat stimuli times and the escape
        if which_angle == 'head_shelter_angle':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) & (self.Video_df["shelter_only"] == True))
            angle_filt = 'hsa'
        elif which_angle == 'head_south_barrier_angle':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) & (self.Video_df["barrier_present"] == True))
            angle_filt = 'h_bar_south_a'
        elif which_angle == 'head_north_barrier_angle':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) & (self.Video_df["barrier_present"] == True))
            angle_filt = 'h_bar_north_a'
        elif which_angle == 'hdir':
            filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True))
            angle_filt = 'hdir'
                
        # Preprocess ----------------------------------------
        number_of_bins = 19
        bin_angles, bin_angle_center = generate_bin_angles(number_of_bins)
        number_of_clusters = self.spikedataframe["spike_clusters"].unique()
        num_cols, num_rows, num_figures = calculate_figure_plotting_axes(how_many_plots_you_need = len(number_of_clusters))
        cluster_counter = 0
       
        # ---------------------------------------------------
        for figure in range(num_figures):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 8), subplot_kw={'projection': 'polar'})

        # assign spike times of each cluster to the corresponding video frame, then assign HD
        number_of_clusters = self.spikedataframe["spike_clusters"].unique()
        for counter,c in enumerate(number_of_clusters):
            # if you have filled a figure with polar plots, move onto next figure
            if counter >= (ncols*nrows)*fnum:
                figg, axs = plt.subplots(nrows,ncols)
                figg.set_figwidth(30)
                figg.set_figheight(15)
                fnum = fnum + 1
                axs = axs.ravel()
            ax = plt.subplot(nrows,ncols,1+counter-(nrows*ncols*(fnum-1)),projection = 'polar')
            # filter spikes by cluster
            spikes = self.spikedataframe.filter(self.spikedataframe['spike_clusters'] == c)
            # count number of spikes on each video frame, and then turn it into firing rate (Hz)
            spikes = spikes.groupby("spike_aligned_to_frame").agg([pl.count("spike_aligned_to_frame").alias("spike_count")])
            spikes = spikes.with_columns(pl.col('spike_count')*self.Visualize.session.video.fps)
            # align spike dataframe to video dataframe
            filtered_video_df = filtered_video_df.select([pl.col('frames').apply(float),pl.exclude('frames')]) 
            spike_to_video_df = filtered_video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="left")
            if spike_to_video_df.select(pl.col('spike_count').is_null().sum()).item() == len(spike_to_video_df):
                logger.info(f"Cluster {c} had no spikes")
                break
            # calculate firing rates in angle bins
            # TODO add this line to rayleigh v function
            spike_to_video_df = spike_to_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
            spike_to_video_df = spike_to_video_df.with_columns(spike_to_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
            spike_to_video_df = spike_to_video_df.fill_null(strategy="zero")
            spike_to_video_df = spike_to_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) # TODO add this line to rayleigh v function
            angles_firing = (spike_to_video_df.groupby(by='binned_angles').agg(pl.col('spike_count').mean().alias('mean_firing_rate')))            
            angles_firing = angles_firing.sort('binned_angles') # TODO add this line to rayleigh v function
            ax.bar(angles_firing['binned_angles'].to_numpy(), angles_firing['mean_firing_rate'].to_numpy(), width=(2*np.pi)/19, bottom=0.0, color='green', alpha=0.5)
            ax.vlines(self.Rayleigh_theta[counter], 0, self.Rayleigh[counter]*np.amax(angles_firing['mean_firing_rate'].to_numpy()), colors='black')
            # add title to the subplot
            if self.Rayleigh_sig[counter] == 1:
                ax.title.set_text('clu ' + str(c) + ' sig.' + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[counter],2)))
            else:
                ax.title.set_text('clu ' + str(c) + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[counter],2)))
            # save the whole figure
            if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1,
                            counter == len(number_of_clusters)-1):
                plt.tight_layout()
                plt.savefig(str(self.Visualize.session.file_path) + "/" + str(which_angle) + "_cluster_polar_plots_" + str(fnum) + ".png")
                if self.Visualize.settings.show_plots: plt.show()  
                plt.close() 

        # # firing per head/shelter angle for each cluster
        # start = [0, int(np.round(len(angles[OutofShelterIdx])/2))] # for splitting up in first and second half
        # end = [int(np.round(len(angles[OutofShelterIdx])/2)),int(len(angles[OutofShelterIdx]))]
        # max_rate = np.zeros(shape = [len(np.unique(clusters))])
        # anglesfiring_clu = np.empty(shape = [len(np.unique(clusters)),len(ang_step)-1,2])
        # timepoints = np.arange(times[0]-1/(2*self.Visualize.session.video.fps), # start of timewindow
        #                        end_time+1/(2*self.Visualize.session.video.fps), # end of timewindow
        #                        1/self.Visualize.session.video.fps) # each time bin is 1 frame
        # cc = ['green','red']
        # for counter,c in enumerate(np.unique(clusters)):
        #     if counter >= (ncols*nrows)*fnum:
        #         figg, axs = plt.subplots(nrows,ncols)
        #         figg.set_figwidth(30)
        #         figg.set_figheight(15)
        #         fnum = fnum + 1
        #         axs = axs.ravel()
        #     ax = plt.subplot(nrows,ncols,1+counter-(nrows*ncols*(fnum-1)),projection = 'polar')
        #     # the firing rate is computed in bins that are centered on the occurrence of a camera frame
        #     srate,_ = np.histogram(spikes[clusters == c],timepoints)
        #     if len(srate)>len(OutofShelterIdx): srate = srate[:-1]
        #     srate = srate[OutofShelterIdx]
        #     srate = srate*self.Visualize.session.video.fps # make it Hz
        #     for i,s in enumerate(zip(start,end)):
        #         for ang in np.arange(1,len(np.linspace(-np.pi,np.pi,24,endpoint = True))):
        #             anglesfiring_clu[counter,ang-1,i] = np.nanmean(srate[np.logical_and(angles[OutofShelterIdx] == ang,
        #                                                                             np.logical_and(np.arange(len(srate))>=s[0],np.arange(len(srate))<=s[1]))])
        #         if len(np.where(np.isnan(anglesfiring_clu[counter,:,i]))[0]) < len(anglesfiring_clu[counter,:,i]): # if the whole thing is NaN
        #             if s[0] == 0: max_rate[counter] = np.nanargmax(anglesfiring_clu[counter,:,i])
        #             # make polar plots of first and second half
        #             ax.bar(ang_step[:-1] + np.diff(ang_step[:2])/2, anglesfiring_clu[counter,:,i], width=(2*np.pi)/24, bottom=0.0, color=cc[i], alpha=0.5)
        #         anglesfiring_clu[counter,:,i] = anglesfiring_clu[counter,:,i]/np.nanmax(anglesfiring_clu[counter,:,i])
        #     if self.Rayleigh_sig[counter] == 1:
        #         ax.title.set_text('clu ' + str(c) + ' sig.' + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[counter],2)))
        #     else:
        #         ax.title.set_text('clu ' + str(c) + '\n' + 'Rayleigh = ' + str(np.around(self.Rayleigh[counter],2)))
        #     if np.logical_or(counter-(nrows*ncols*(fnum-1)) == (ncols*nrows)-1,
        #                     counter == len(np.unique(clusters))-1):
        #         plt.tight_layout()
        #         plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_cluster_polar_plots_" + str(fnum) + ".png")
        #         if self.Visualize.settings.show_plots: plt.show()  
        #         plt.close()              
        
        # _, axs = plt.subplots(1, 2)
        # # heatmap of first half, sorted by angle with max firing
        # axs[0].imshow(anglesfiring_clu[np.argsort(max_rate),:,0],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(np.unique(clusters))])
        # axs[0].set_ylabel('cluster (sort on pref HSA)')
        # axs[0].set_xlabel(title + ' (radians)')
        # axs[0].title.set_text('first half')
        # # heatmap of second half, sorted on first half
        # axs[1].imshow(anglesfiring_clu[np.argsort(max_rate),:,1],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(np.unique(clusters))])
        # axs[1].set_xlabel(title + ' (radians)')
        # axs[1].title.set_text('second half')
        # plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_cluster_tuning.png")
        # if self.Visualize.settings.show_plots: plt.show()
        # plt.close()

        # # heatmap, but restricted to clusters with significant rayleigh vectors
        # _, axs = plt.subplots(1, 2)
        # # heatmap of first half, sorted by angle with max firing
        # A = anglesfiring_clu[self.Rayleigh_sig == 1,:,:]
        # M = max_rate[self.Rayleigh_sig == 1]
        # axs[0].imshow(A[np.argsort(M),:,0],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(M)])
        # axs[0].set_ylabel('cluster (sort on pref HSA)')
        # axs[0].set_xlabel(title + ' (radians)')
        # axs[0].title.set_text('first half')
        # # heatmap of second half, sorted on first half
        # axs[1].imshow(A[np.argsort(M),:,1],cmap = 'hot',aspect = .1,extent = [-np.pi,np.pi,0,len(M)])
        # axs[1].set_xlabel(title + ' (radians)')
        # axs[1].title.set_text('second half')
        # plt.savefig(str(self.Visualize.session.file_path) + "/" + str(title) + "_cluster_tuning_sig_Rayleigh.png")
        # if self.Visualize.settings.show_plots: plt.show()
        # plt.close()


# Utility functions ------------------------------------------------------------------------------------------------
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
