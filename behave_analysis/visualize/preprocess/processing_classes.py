# OS libaries
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
from glob import glob
import polars as pl
import time
import os

# Custom libaries sdf dsfgsdf
from behave_analysis.database.synthetic_data.synthetic_main import generate_synthetic_dataframe

class BaseDataPreprocessor(ABC):
    """
    A parent class to support the real and synthetic data preprocessor children. 
    """
    def __init__(self, visualize_object, cluster_labels_to_filter):
        logger.info("Preprocessing started")
        self.Visualize = visualize_object
        self.select_cluster_labels = cluster_labels_to_filter

    # --------------- Abstract methods to be implemented by all children ---------------------------------------------

    @abstractmethod
    def merge_and_save_spike_count_df_with_frame_data(self):
        pass
    
    @abstractmethod
    def load_spike_data(self):
        pass
    
    # --------------- Concrete methods implemented ----------------------------------------------------------------------
    
    def extract_cluster_labels(self):
        """
        Extracts the cluster labels from the spike data dataframe and returns them
        """
        
        clu_label = self.spike_data.groupby(["spike_clusters"]).first()
        clu_label = clu_label.drop(["spike_aligned_to_frame", "spike_times", "aligned_spike_times", "aligned_spike_times_in_samples"])
        return clu_label

    def load_spike_data(self) -> pl.DataFrame:
        spike_data = pl.read_csv(self.csv_path)
        logger.success("Data found ready for preprocessing")
        return spike_data
        
    def behaviourally_pure_tracking_data(self, video_df):
        """
        Filter out all the data where the mouse is in the shelter for example
        """
        filtered_video_df = video_df.filter((video_df["OutofshelterIdx"] == True) & (video_df["EscapePeriod"] == False))        
        assert len(filtered_video_df) > 0, "No data left after filtering"
        return filtered_video_df
    
    def track_to_polars(self) -> pl.DataFrame:
        """
        Adds all the behavioral variables from track to a polars dataframe, video_df - and saves it

        Returns: Video_df
        """
        # if mushroom, estend size to outer circle
        if np.logical_and(self.Visualize.tracking_data['shelter_loc'][1][0] - self.Visualize.tracking_data['shelter_loc'][0][0]<50,
                            self.Visualize.tracking_data['shelter_loc'][1][1] - self.Visualize.tracking_data['shelter_loc'][0][1]<50):
            self.Visualize.tracking_data['shelter_loc'][0] = [x - 35 for x in self.Visualize.tracking_data['shelter_loc'][0]]
            self.Visualize.tracking_data['shelter_loc'][1] = [x + 35 for x in self.Visualize.tracking_data['shelter_loc'][1]]
        
        # if side shelter make sure it goes all the way to the edge of image, mouse can't be 'behind' shelter
        if self.Visualize.tracking_data['shelter_loc'][1][1] > 900: self.Visualize.tracking_data['shelter_loc'][1][1] = 1024
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
        video_df = pl.DataFrame(
                {"frames": np.arange(1,len(self.Visualize.tracking_data['hdir'])+1).astype(np.int64),
                "hdir": self.Visualize.tracking_data['hdir'],
                "hsa": self.Visualize.tracking_data['hdir_shelt'],
                "mouse_x_position": self.Visualize.tracking_data['avg_loc'][:,0],
                "mouse_y_position": self.Visualize.tracking_data['avg_loc'][:,1],
                "OutofshelterIdx": OutofShelterIdx, # was the mouse in the shelter?
                "EscapePeriod": EscapePeriod == 1, # frames from 1 second before to 10 seconds after escape
                "shelter_only": shelteronly, # was this in a shelter only period? or was there a barrier?
                "barrier_present": barrier_present,}) # was this in a barrier period? or was there a barrier?

        # if barrier in session, add the angles to video_df
        if 'hdir_barrier' in self.Visualize.tracking_data:
            video_df = video_df.hstack([pl.Series("h_bar_north_a",self.Visualize.tracking_data['hdir_barrier'][:,0])])
            video_df = video_df.hstack([pl.Series("h_bar_south_a",self.Visualize.tracking_data['hdir_barrier'][:,1])])
            video_df = video_df.hstack([pl.Series("h_bar_centre_a",self.Visualize.tracking_data['hdir_barrier'][:,2])])

        # if random points were included, add the angles to video_df
        if 'hdir_randP' in self.Visualize.tracking_data:
            for i in np.arange(np.shape(self.Visualize.tracking_data['hdir_randP'])[1]):
                video_df = video_df.hstack([pl.Series(str('head_randP_' + str(i)),self.Visualize.tracking_data['hdir_randP'][:,i])])

        video_df.write_csv(self.Visualize.session.processed_path + "/" + "full_video_dataframe.csv")
        return video_df
        
    def count_spikes_and_units_to_frames(self) -> pl.DataFrame:
        """
        Uses polars query logic to map each cluster to a frame and count how many times each cluster fired in that frame.
        The lazy() function means that computations are not immediately executed. This allows the computer to plan the operations before
        proceeding. NOTE - if there are changes to the code, ensure to delete any existing CSVs so you can see the changes of the code.
        """
        try:
            logger.info("Attempting to load a previously computed spike frame count")
            with open(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_cluster_labels +"cluster.csv", "rb") as file:
                spikecountbyframe_neuron = pl.read_csv(file.read())
            logger.success("Found spike count by frame and cluster dataframe, loading it now")
            return spikecountbyframe_neuron
                    
        except FileNotFoundError:
            logger.info("Could not find spike count by frame and cluster dataframe, creating it now")
            logger.info("Commencing long computation to count spikes for each cluster for each frame")
            query = (self.spike_data.lazy().groupby(["spike_aligned_to_frame", "spike_clusters"]).agg([pl.count("spike_aligned_to_frame").alias("spike_count")])) # Lazy query to plan computation
            start_time = time.time() # Collect lazy query and time it for user as this is the longest computation in the pipeline
            spikecountbyframe_neuron = query.collect()
            print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
            spikecountbyframe_neuron.write_csv(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_cluster_labels +"cluster.csv")
            return spikecountbyframe_neuron

    def merge_and_save_spike_count_df_with_frame_data(self):
        logger.info("merging video df and spike df into a super df")
        video_df = self.video_df.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
        large_dataFrame = video_df.join(self.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        large_dataFrame = large_dataFrame.fill_null(strategy="zero")
        large_dataFrame.write_csv(self.Visualize.session.processed_path + "/" + str(self.select_clusters) + "_large_dataframe.csv")

    def export_large_df_to_frame_by_cluster_matrix(self):
        logger.info("building a frame by cluster matrix of firing rates")
        clu = self.spikeCountByFrameAndCluster["spike_clusters"].unique().to_numpy()
        # group the  data
        df = self.spikeCountByFrameAndCluster.groupby(["spike_aligned_to_frame"]).all()
        df = df.sort('spike_aligned_to_frame')

        # frames by firing per cluster matrix
        frames = self.video_df['frames'].unique().to_numpy().astype(int)

        X = np.zeros((np.amax(frames),len(clu)))

        for i2 in frames:
            d = df.filter(df["spike_aligned_to_frame"] == i2).to_dict(as_series=False)
            if len(d['spike_count']) > 0:
                spikes = np.array(d.get('spike_count')[0])
                clusters = np.array(d.get('spike_clusters')[0])
                spikes = spikes[np.argsort(clusters)]
                clusters = np.sort(clusters)
                X[int(i2)-1,np.where(np.in1d(clu,clusters))[0]] = spikes
            
        if clu[0] == 0: X = X[:,1:]

        # transform to firing rate estimate in Hz
        sampling_rate = self.Visualize.session.video.fps # in fps
        window_size = 100 # in ms
        nbins = 1000/window_size
        for i in np.arange(np.shape(X)[1]):
            X[:,i] = np.convolve(X[:,i],np.ones(int(sampling_rate/nbins),dtype = int),'same')*nbins

        np.save(str(self.Visualize.session.processed_path + "/" + "frame_by_" + self.select_cluster_labels +"_cluster_matrix"),X)
      
##### -------- synthetic data processor
class SyntheticDataPreprocessor(BaseDataPreprocessor):
    """
    A child class to support the synthetic data preprocessing pipeline. 
    """
    def __init__(self, visualize_object, cluster_labels_to_filter, expand_behavioural_data = False):
        super().__init__(visualize_object, cluster_labels_to_filter)
        self.csv_path = os.path.join(self.Visualize.session.processed_path, str(str(cluster_labels_to_filter) + "_efizz_data.csv"))
        self.select_clusters = cluster_labels_to_filter
        self.video_df = self.track_to_polars()
        self.expand_behavioural_data = expand_behavioural_data
        if expand_behavioural_data: 
            self.video_df = self.expand_tracking_data(video_df = self.video_df, 
                                                      new_entries_to_insert = self.Visualize.settings.num_samples_of_expansion)
        self.check_synthetic_data_exists_if_not_generate_it() # creates a csv in working dir
        self.spike_data = self.load_spike_data()
        self.clu_label = self.extract_cluster_labels()
        self.spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames()
        self.merge_and_save_spike_count_df_with_frame_data()
        self.export_large_df_to_frame_by_cluster_matrix()
    
    def check_synthetic_data_exists_if_not_generate_it(self) -> None:
        if not os.path.exists(self.csv_path):
            self.activate_synthetic_data_generation()
        else:
            logger.info("Synthetic spike data found")
    
    def activate_synthetic_data_generation(self) -> None:
        logger.info("Synthetic spike data doesn't exist and will now be generated")
        tuning = ['hdir']
        if np.logical_or(np.logical_and(len(self.Visualize.session.shelter_time) > 0,self.select_clusters == 'synthetic'),'hsa' in self.select_clusters): 
            tuning.append('hsa')
        if np.logical_and(len(self.Visualize.session.barrier_time) > 0,self.select_clusters == 'synthetic'): 
            tuning.append('h_bar_north_a')
            tuning.append('h_bar_south_a')
        synth_df = generate_synthetic_dataframe(tuning, pass_video_df = self.video_df)
        synth_df.write_csv(self.csv_path)

    def expand_tracking_data(self, video_df: pl.DataFrame, new_entries_to_insert: int) -> pl.DataFrame:
        """
        Uniformly expands the tracking data by a specified number of entries to simulate a longer, perfectly sampled experiment.
        NOTE this function adds angles to hsa, barrier north and barrier south even if they don't exist in the data
        """
        
        # Generate polar series ranging from [last_frame_index + 1, last_frame_index + 1 + new_entries_to_insert]
        last_frame_index = video_df['frames'].max()
        new_frames = pl.Series(
            'frames', 
            np.arange(last_frame_index + 1, last_frame_index + 1 + new_entries_to_insert).astype(np.int64)
                              ) # Generate \ polar series ranging from [last_frame_index+1, last_frame_index+1+new_entries_to_insert]
        
        # Generate new angles sampled from a uniform distribution between -pi and pi for number of new entries
        angle_columns = ['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a']
        new_angle_cols = [pl.Series(col, np.random.uniform(-np.pi, np.pi, new_entries_to_insert)) for col in angle_columns] # Create a list of polar series for each angle column
        
        # Generate new boolean columns with a specified value for number of new entries
        new_out_of_shelter_idx = pl.Series('OutofshelterIdx', np.full(new_entries_to_insert, fill_value=True))
        new_escape_period_idx = pl.Series('EscapePeriod', np.full(new_entries_to_insert, fill_value=False))
        new_shelter_only_idx = pl.Series('shelter_only', np.random.choice([True, False], size=new_entries_to_insert))
        new_barrier_present_idx = pl.Series('barrier_present', np.random.choice([True, False], size=new_entries_to_insert))
        
        # Generate new mouse position columns sampled from a uniform distribution between -1 and 1 for number of new entries
        min_x, max_x = min(video_df['mouse_x_position']), max(video_df['mouse_x_position'])
        min_y, max_y = min(video_df['mouse_y_position']), max(video_df['mouse_y_position'])
        new_mouse_x_position = pl.Series('mouse_x_position', np.random.uniform(min_x, max_x, new_entries_to_insert))
        new_mouse_y_position = pl.Series('mouse_y_position', np.random.uniform(min_y, max_y, new_entries_to_insert))
        
        # Generate new dataframe
        df_new = pl.DataFrame([new_frames] + 
                              new_angle_cols + 
                              [new_mouse_x_position, new_mouse_y_position] + 
                              [new_out_of_shelter_idx, new_escape_period_idx, new_shelter_only_idx, new_barrier_present_idx])
        
        # Concatenate new dataframe with original dataframe
        expanded_synthetic_tracking_data_by_frame = pl.concat([video_df, df_new])
        logger.success("Tracking data synthetically expanded by " + str(new_entries_to_insert) + " entries.")
        
        return expanded_synthetic_tracking_data_by_frame

#### ------------ data processor
class DataPreprocessor(BaseDataPreprocessor):
    """
    A child class to support the production data preprocessing pipeline. 
    """
    def __init__(self, visualize_object, cluster_labels_to_filter):
        super().__init__(visualize_object, cluster_labels_to_filter)
        
        assert cluster_labels_to_filter != "synthetic", "Synthetic data is not supported by this class."
        
        self.csv_path = glob(os.path.join(self.Visualize.session.processed_path, "Processed_efizz_data"))[0]
        self.select_clusters = cluster_labels_to_filter
        unfiltered_spike_data = self.load_spike_data()
        self.spike_data = self.filter_spike_data(unfiltered_spike_data)
        self.clu_label = self.extract_cluster_labels()
        self.video_df = self.track_to_polars()
        self.spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames()
        self.merge_and_save_spike_count_df_with_frame_data() # Saves to a csv
        self.export_large_df_to_frame_by_cluster_matrix()
        
    def filter_spike_data(self,df):
        """
        Filter the spike data to only include good neurons or good + MUA
        """        
        if self.select_clusters == 'all':
            filtered_spike_data = df.filter((df['cluster_group'] == "good") | (df['cluster_group'] == "mua"))
            logger.info("Loaded good and multi unit clusters")
        else:
            filtered_spike_data = df.filter(df['cluster_group'] == self.select_clusters)
            numNeurons = len(filtered_spike_data['spike_clusters'].unique())
            logger.info(f"Loaded {numNeurons} {self.select_clusters} clusters")
        
        return filtered_spike_data
