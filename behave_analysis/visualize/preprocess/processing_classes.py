# OS libaries
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
from glob import glob
import polars as pl
import time
import os

# Custom libaries
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
    def filter_spike_data(self):
        pass
    
    @abstractmethod
    def merge_and_save_spike_count_df_with_frame_data(self):
        pass
    
    @abstractmethod
    def load_spike_data(self):
        pass
    
    # --------------- Concrete methods implemented ------------------------------------------------
        
    def behaviourally_pure_tracking_data(self, video_df):
        """
        Filter out all the data where the mouse is in the shelter for example
        """
        filtered_video_df = video_df.filter((video_df["OutofshelterIdx"] == True) & (video_df["EscapePeriod"] == False))        
        assert len(filtered_video_df) > 0, "No data left after filtering"
        return filtered_video_df
    
    def track_to_polars(self) -> pl.DataFrame:
        """
        Adds all the behavioral variables from track to the polars sike dataframe

        # NOTE: Wait does it do that? this function looks like it takes the kalman filter data and creates the video dataframe?

        # It seems the main point of this function is to generate video_df which will later be combined with the spike data

        Returns: Video_df
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
        video_df = pl.DataFrame(
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
    
        return video_df
        
    def count_spikes_and_units_to_frames(self, spike_data_frame) -> pl.DataFrame:
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
            query = (spike_data_frame.lazy().groupby(["spike_aligned_to_frame", "spike_clusters"]).agg([pl.count("spike_aligned_to_frame").alias("spike_count")])) # Lazy query to plan computation
            start_time = time.time() # Collect lazy query and time it for user as this is the longest computation in the pipeline
            spikecountbyframe_neuron = query.collect()
            print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
            spikecountbyframe_neuron.write_csv(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_cluster_labels +"cluster.csv")
            return spikecountbyframe_neuron

class SyntheticDataPreprocessor(BaseDataPreprocessor):
    """
    A child class to support the synthetic data preprocessing pipeline. 
    """
    def __init__(self, visualize_object, cluster_labels_to_filter, expand_behavioural_data = False):
        super().__init__(visualize_object, cluster_labels_to_filter)
        self.csv_path = os.path.join(self.Visualize.session.processed_path, "synthetic_efizz_data.csv")
        self.select_clusters = "synthetic"
        self.video_df = self.track_to_polars()
        if expand_behavioural_data: 
            self.video_df = self.expand_tracking_data(video_df = self.video_df, new_entries_to_insert = 1000000)
        self.check_synthetic_data_exists_if_not_generate_it() # creates a csv in working dir
        self.spike_data = self.load_spike_data()
        self.spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames(self.spike_data)
        self.merge_and_save_spike_count_df_with_frame_data()
    
    def check_synthetic_data_exists_if_not_generate_it(self) -> None:
        if not os.path.exists(self.csv_path):
            self.activate_synthetic_data_generation()
        else:
            logger.info("Synethic spike data found.")
    
    def load_spike_data(self) -> pl.DataFrame:
        spike_data = pl.read_csv(self.csv_path)
        logger.success("Data found ready for preprocessing")
        return spike_data
    
    def activate_synthetic_data_generation(self) -> None:
        logger.info("Synthetic spike data doesn't exist and will now be generated")
        tuning = ['hdir']
        if len(self.Visualize.session.shelter_time) > 0: 
            tuning.append('hsa')
        if len(self.Visualize.session.barrier_time) > 0: 
            tuning.append('h_bar_north_a')
            tuning.append('h_bar_south_a') # Adding as seperate line as bug when adding two params at once
        synth_df = generate_synthetic_dataframe(tuning, pass_video_df = self.video_df)
        synth_df.write_csv(self.csv_path)
    
    def merge_and_save_spike_count_df_with_frame_data(self) -> None:
        video_df = self.video_df.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
        large_dataFrame = video_df.join(self.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        large_dataFrame = large_dataFrame.fill_null(strategy="zero")
        large_dataFrame.write_csv(self.Visualize.session.processed_path + "/" + str(self.select_clusters) + "_large_dataframe.csv")

    def expand_tracking_data(self, video_df: pl.DataFrame, new_entries_to_insert: int) -> pl.DataFrame:
        """
        Uniformly expands the tracking data by a specified number of entries.
        """
        last_frame_index = video_df['frames'].max() # Get the last frame index to generate frames from there (add to end of dataframe)
        new_frames = pl.Series('frames', np.arange(last_frame_index+1, last_frame_index+1+new_entries_to_insert).astype(np.int64)) # Generate new frames column
        angle_columns = ['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a']  # Generate random angles in radians for specified columns
        new_angle_cols = [pl.Series(col, np.random.uniform(-np.pi, np.pi, new_entries_to_insert)) for col in angle_columns]
        bool_columns = ['OutofshelterIdx', 'EscapePeriod', 'shelter_only', 'barrier_present']
        bool_values = [True, False, False, True]  # Set your desired True/False values for each column
        
        new_bool_cols = [pl.Series(col, np.full(new_entries_to_insert, fill_value=val)) for col, val in zip(bool_columns, bool_values)]
        
        new_mouse_x_position = pl.Series('mouse_x_position', np.random.uniform(-1, 1, new_entries_to_insert))
        new_mouse_y_position = pl.Series('mouse_y_position', np.random.uniform(-1, 1, new_entries_to_insert))
        df_new = pl.DataFrame([new_frames] + new_angle_cols + [new_mouse_x_position, new_mouse_y_position] + new_bool_cols)
        expanded_synthetic_tracking_data_by_frame = pl.concat([video_df, df_new])
        return expanded_synthetic_tracking_data_by_frame

    # ----------------------Currently not used ------------------------------
    def filter_spike_data(self) -> NotImplementedError:
        """
        I actually don't think this function is needed for synethic, there was a think called self.clu_label but it wasn't used across the code base so assuming not needed.
        """
        raise NotImplementedError
    
class DataPreprocessor(BaseDataPreprocessor):
    """
    A child class to support the production data preprocessing pipeline. 
    """
    def __init__(self, visualize_object, cluster_labels_to_filter):
        super().__init__(visualize_object, cluster_labels_to_filter)
        
        assert cluster_labels_to_filter != "synthetic", "Synthetic data is not supported by this class."
        
        self.csv_path = glob(os.path.join(self.Visualize.session.processed_path, "Processed_efizz_data"))[0]
        self.select_clusters = cluster_labels_to_filter
        self.spike_data = self.load_spike_data()
        self.filtered_spike_data, self.clu_label = self.filter_spike_data()
        self.video_df = self.track_to_polars()
        self.spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames(self.filtered_spike_data)
        self.merge_and_save_spike_count_df_with_frame_data() # Saves to a csv
        
    def load_spike_data(self) -> pl.DataFrame:
        """
        This function is actually identical but I can't figure out how to get it into the base clase as a concrete object because of it's reliance on self.csv_path
        """
        spike_data = pl.read_csv(self.csv_path)
        logger.success("Data found ready for preprocessing")
        return spike_data

    def filter_spike_data(self):
        """
        Filter the spike data to only include good neurons or good + MUA
        """        
        if self.select_clusters == 'all':
            filtered_spike_data = self.spike_data.filter((self.spike_data['cluster_group'] == "good") | (self.spike_data['cluster_group'] == "mua"))
            logger.info("Loaded good and multi unit clusters")
        else:
            filtered_spike_data = self.spike_data.filter(self.spike_data['cluster_group'] == self.select_clusters)
            numNeurons = len(filtered_spike_data['spike_clusters'].unique())
            logger.info(f"Loaded {numNeurons} {self.select_clusters} clusters")
        
        # NOTE - If these two lines are to extract clu_label then can be removed to another function - Check with Jazz
        clu_label = filtered_spike_data.groupby(["spike_clusters"]).first()
        clu_label = clu_label.drop(["spike_aligned_to_frame", "spike_times", "aligned_spike_times", "aligned_spike_times_in_samples"])
        
        return filtered_spike_data, clu_label
    
    def merge_and_save_spike_count_df_with_frame_data(self) -> pl.DataFrame:
        """
        Merge and save the spike count dataframe with the video dataframe and save it as a new dataframe for later use in the pipeline in the processed file
        """
        video_df = self.video_df.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
        large_dataFrame = video_df.join(self.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        large_dataFrame = large_dataFrame.fill_null(strategy="zero")
        large_dataFrame.write_csv(self.Visualize.session.processed_path + "/" + str(self.select_clusters) + "_large_dataframe.csv")
        return large_dataFrame




# ------------------------------------------------------------------------------------------------------------------------------------------------------

# OLD preprocessing class for reference
# class PreProcess:
#     """
#     A class that loads the csv of aligned data and processes it into a dataframe that can be used for visualisation
#     """
#     def __init__(self,  
#                  visualize_object, 
#                  run = "Production", 
#                  select_clusters = "good", 
#                  user_wants_to_regenerate_spike_by_frame_count = False):
        
#         logger.info("Preprocessing started")
#         self.Visualize = visualize_object
#         self.run_type = run
#         if run == "Test": 
#             self.select_clusters = "synthetic"
#         else:
#             self.select_clusters = select_clusters
#         self.load_spike_data()
#         self.filter_spike_data()
#         self.track_to_polars()
        
#         # TODO: Functions that assign variable to self should return the variable to it's clear what is being assigned e.g the below functions
#         # Refactor this class for the above functions to do that
#         self.spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames(user_wants_to_regenerate_spike_by_frame_count)
#         self.clean_behavioural_data = self.behaviourally_pure_tracking_data()
#         self.large_dataFrame = self.merge_and_save_spike_count_df_with_frame_data(expand_behavioural_data = False)
    
#     def activate_synthetic_data_generation(self):
#         logger.info("Synethic spike data doesn't exist and will now be generated")
#         tuning = ['hdir']
        
#         if len(self.Visualize.session.shelter_time) > 0: 
#             tuning.append('hsa')
            
#         if len(self.Visualize.session.barrier_time) > 0: 
#             tuning.append('h_bar_north_a')
#             tuning.append('h_bar_south_a') # Adding as seperate line as bug when adding two params at once
            
#         synth_df = generate_synthetic_dataframe(tuning)
#         synth_df.write_csv(self.csv_path)
    
#     def load_spike_data(self):
#         """
#         Depending on how the pipeline is flagged. This flag loads either real efizz data or fake data.
#         """
#         if self.run_type == "Production":
#             self.csv_path = glob(os.path.join(self.Visualize.session.processed_path, "Processed_efizz_data"))[0]
            
#         elif self.run_type == "Test":
#             self.csv_path = os.path.join(self.Visualize.session.processed_path, "synthetic_efizz_data.csv")
#             if not os.path.exists(self.csv_path):
#                 self.activate_synthetic_data_generation()
#             else:
#                 logger.info("Synethic spike data is being used when visualizing efizz - Real positional data is used from databank")
                
#         else: 
#             raise ValueError("Run type not recognised")

#     def filter_spike_data(self):
#         """
#         Filter the spike data to only include good neurons or good + MUA
#         """
#         # NOTE - This will break if user says yes to both mua and good - too tired to fix 
        
#         dataFrame = pl.read_csv(self.csv_path)
        
#         if self.run_type == "Production":
#             if self.select_clusters == 'all':
#                 self.spikedataframe = dataFrame.filter((dataFrame['cluster_group'] == "good")
#                                                     | (dataFrame['cluster_group'] == "mua"))
#                 logger.info("Loaded good and multi unit clusters")
#             else:
#                 self.spikedataframe = dataFrame.filter(dataFrame['cluster_group'] == self.select_clusters)
#                 numneurons = len(self.spikedataframe['spike_clusters'].unique())
#                 logger.info(f"Loaded {numneurons} {self.select_clusters} clusters")
                
#         elif self.run_type == "Test":
#             self.spikedataframe = dataFrame
#             logger.info("Loaded all clusters")
        
#         self.clu_label = self.spikedataframe.groupby(["spike_clusters"]).first()
#         self.clu_label = self.clu_label.drop(["spike_aligned_to_frame", "spike_times", "aligned_spike_times", "aligned_spike_times_in_samples"])

#     def track_to_polars(self) -> pl.DataFrame:
#         """
#         Adds all the behavioral variables from track to the polars sike dataframe
        
#         # NOTE: Wait does it do that? this function looks like it takes the kalman filter data and creates the video dataframe?
        
#         # It seems the main point of this function is to generate video_df which will later be combined with the spike data
        
#         Returns: Video_df
#         """
#         OutofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0],
#             self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]),
#             np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1],
#             self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1])))
        
#          # is there a time with shelter only?
#         if len(self.Visualize.session.shelter_time) > 0:
#             if not(np.logical_and(self.Visualize.session.shelter_time[0] == 0, self.Visualize.session.shelter_time[1] == -1)):
#                 if self.Visualize.session.shelter_time[1] == -1: # shelter only until the end of the session
#                     shelteronly = np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps)
#                 else:
#                     shelteronly = np.logical_and(np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.sheltertime[0]*self.Visualize.session.video.fps),
#                                                 np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) < (self.Visualize.sheltertime[1]*self.Visualize.session.video.fps))
#             else:
#                 shelteronly = np.zeros(len(OutofShelterIdx)) == 0
#                 print('shelter always present')
#         else:
#             shelteronly = np.zeros(len(OutofShelterIdx)) == 0
#             print('no shelter in this session')
#          # what period in the recording was there a barrier?
#         if len(self.Visualize.session.barrier_time) > 0:
#             if self.Visualize.session.barrier_time[1] == -1: # shelter only until the end of the session
#                 barrier_present = np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.barriertime[0]*self.Visualize.session.video.fps)
#             else:
#                 barrier_present = np.logical_and(np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) > (self.Visualize.barriertime[0]*self.Visualize.session.video.fps),
#                                              np.arange(1,len(self.Visualize.tracking_data['hdir'])+1) < (self.Visualize.barriertime[1]*self.Visualize.session.video.fps))
#         else:
#             barrier_present = np.zeros(len(OutofShelterIdx)) == 1
#             print('no barrier in this session')
#         # find the escape periods
#         EscapePeriod = np.zeros_like(OutofShelterIdx)
#         for onsets in self.Visualize.session.audio.onset_frames:
#             EscapePeriod[(onsets[0]-self.Visualize.session.video.fps):(onsets[0]+(10*self.Visualize.session.video.fps))] = 1
#         # make a video dataframe where for each video frame:
#         self.Video_df = pl.DataFrame(
#                 {"frames": np.arange(1,len(self.Visualize.tracking_data['hdir'])+1).astype(np.int64),
#                 "hdir": self.Visualize.tracking_data['hdir'],
#                 "hsa": self.Visualize.tracking_data['hdir_shelt'],
#                 "h_bar_north_a": self.Visualize.tracking_data['hdir_barrier'][:,0],
#                 "h_bar_south_a": self.Visualize.tracking_data['hdir_barrier'][:,1],
#                 "mouse_x_position": self.Visualize.tracking_data['avg_loc'][:,0],
#                 "mouse_y_position": self.Visualize.tracking_data['avg_loc'][:,1],
#                 "OutofshelterIdx": OutofShelterIdx, # was the mouse in the shelter?
#                 "EscapePeriod": EscapePeriod == 1, # frames from 1 second before to 10 seconds after escape
#                 "shelter_only": shelteronly, # was this in a shelter only period? or was there a barrier?
#                 "barrier_present": barrier_present,}) # was this in a barrier period? or was there a barrier?

#     def behaviourally_pure_tracking_data(self):
#         """
#         Filter out all the data where the mouse is in the shelter for example
#         """
#         filtered_video_df = self.Video_df.filter((self.Video_df["OutofshelterIdx"] == True) 
#                                                  & (self.Video_df["EscapePeriod"] == False))        
 
#         assert len(filtered_video_df) > 0, "No data left after filtering"
        
#         return filtered_video_df

#     def count_spikes_and_units_to_frames(self, user_wants_to_regenerate_spike_by_frame_count = False):
#         """
#         Testing the query format of polars. In theory by using the query formation we can speed up the computation of the spike count outside of a loop
#         by using the lazy() function, which means that computations are not immediately executed. This allows the computer to plan the operations before
#         proceeding. Additionally the computational power is not linear, and thread operations are at play. This means outside of a loop should be faster. 
#         """
        
#         # NOTE - THis will create an arror if the filteer on the cells changes e.g good vs mua as the dataframe will not update
#         # TODO - Fix this
        
#         if user_wants_to_regenerate_spike_by_frame_count == False:
#             try:
#                 with open(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_clusters +"cluster.csv", "rb") as file:
#                     spikecountbyframe_neuron = pl.read_csv(file.read())
#                 logger.success("Found spike count by frame and cluster dataframe, loading it now")
#                 return spikecountbyframe_neuron
                    
#             except FileNotFoundError:
#                 logger.info("Could not find spike count by frame and cluster dataframe, creating it now")
#                 logger.info("Commencing long computation to count spikes for each cluster for each frame")
#                 query = (self.spikedataframe.lazy().groupby(["spike_aligned_to_frame", "spike_clusters"]).agg([pl.count("spike_aligned_to_frame").alias("spike_count")])) # Lazy query to plan computation
#                 start_time = time.time() # Collect lazy query and time it for user as this is the longest computation in the pipeline
#                 spikecountbyframe_neuron = query.collect()
#                 print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
#                 spikecountbyframe_neuron.write_csv(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_clusters +"cluster.csv")
#                 return spikecountbyframe_neuron
        
#         elif user_wants_to_regenerate_spike_by_frame_count == True:
#             logger.info("recreating the spike count by frame and unit dataframe as requested by user, likely because of changing the filter on cluster type, creating it now")
#             logger.info("Commencing long computation to count spikes for each cluster for each frame")
#             query = (self.spikedataframe.lazy().groupby(["spike_aligned_to_frame", "spike_clusters"]).agg([pl.count("spike_aligned_to_frame").alias("spike_count")])) # Lazy query to plan computation
#             start_time = time.time() # Collect lazy query and time it for user as this is the longest computation in the pipeline
#             spikecountbyframe_neuron = query.collect()
#             print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
#             spikecountbyframe_neuron.write_csv(self.Visualize.session.processed_path + "/" + "spike_count_by_frame_and_" + self.select_clusters +"cluster.csv")
#             return spikecountbyframe_neuron
    
#     def merge_and_save_spike_count_df_with_frame_data(self, expand_behavioural_data = True) -> pl.DataFrame:
#         """
#         Merge and save the spike count dataframe with the video dataframe and save it as a new dataframe for later use in the pipeline in the processed file
#         """
#         if self.run_type == "Test":
            
#             if not expand_behavioural_data:
#                 video_df = self.Video_df.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
#                 large_dataFrame = video_df.join(self.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
#                 large_dataFrame.write_csv(self.Visualize.session.processed_path + "/" + self.run_type + "_large_dataframe.csv") # TODO - Change this name to change depending on synthetic or not
            
#             if expand_behavioural_data:
#                 self.expanded_tracking_data_for_synethic_tests = self.expand_tracking_data(self.Video_df, new_entries_to_insert = 100000)
#                 video_df = self.expanded_tracking_data_for_synethic_tests.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
#                 large_dataFrame = video_df.join(self.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
#                 large_dataFrame.write_csv(self.Visualize.session.processed_path + "/" + self.run_type + "_large_dataframe.csv") # TODO - Change this name to change depending on synthetic or not
        
#         elif self.run_type == "Production":
#             video_df = self.Video_df.select([pl.col('frames').apply(float), pl.exclude('frames')]) # Cast frames to float to permit join and remove old frames column with wrong type 
#             large_dataFrame = video_df.join(self.spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
#             large_dataFrame.write_csv(self.Visualize.session.processed_path + "/" + self.run_type + "_large_dataframe.csv") # TODO - Change this name to change depending on synthetic or not
            
#         return large_dataFrame
    
#     def expand_tracking_data(self, 
#                              video_df: pl.DataFrame, 
#                              new_entries_to_insert: int) -> pl.DataFrame:
#         """Potentially breaking change. The point of this function is to artifically enhance the video_df that joins with the spike data.
#         Because the video_df holds the tracking data and frame numbers which the synthetic spikes are generated from. If we expand this
#         dataframe we can test the limits of how our analysis models work with different sample sizes. 

#         Returns:
#             _type_: _description_
#         """
        
#         last_frame_index = video_df['frames'].max()

#         # Generate new frames column
#         new_frames = pl.Series('frames', np.arange(last_frame_index+1, last_frame_index+1+new_entries_to_insert).astype(np.int64))

#         # Generate random angles in radians for specified columns
#         angle_columns = ['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a']
#         new_angle_cols = [pl.Series(col, np.random.uniform(-np.pi, np.pi, new_entries_to_insert)) for col in angle_columns]

#         # random choice of the boolean values for specified columns would be better to maintain the value
#         bool_columns = ['OutofshelterIdx', 'EscapePeriod', 'shelter_only', 'barrier_present']
#         new_bool_cols = [pl.Series(col, np.random.choice([True, False], new_entries_to_insert)) for col in bool_columns]

#         # Generate new mouse position (x and y)
#         new_mouse_x_position = pl.Series('mouse_x_position', np.random.uniform(-1, 1, new_entries_to_insert))
#         new_mouse_y_position = pl.Series('mouse_y_position', np.random.uniform(-1, 1, new_entries_to_insert))

#         # Concatenate original dataframe with new entries
#         df_new = pl.DataFrame([new_frames] + new_angle_cols + [new_mouse_x_position, new_mouse_y_position] + new_bool_cols)
#         expanded_synthetic_tracking_data_by_frame = pl.concat([video_df, df_new])

#         return expanded_synthetic_tracking_data_by_frame