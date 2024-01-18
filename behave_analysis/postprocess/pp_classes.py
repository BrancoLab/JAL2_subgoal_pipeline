import os
from abc import ABC, abstractmethod
from glob import glob
import time

from loguru import logger
import numpy as np
import polars as pl

from behave_analysis.database.synthetic_data.synthetic_main import generate_synthetic_dataframe
from behave_analysis.postprocess.out_of_shelter import out_of_shelter_filter


class BaseDataPostprocessor(ABC):
    """
    A parent class to support the real and synthetic data postprocessing children.
    """

    def __init__(self, cluster_labels_to_filter, tracking_data, session, settings):
        logger.info("Postprocessing started")
        self.select_cluster_labels = cluster_labels_to_filter
        self.tracking_data = tracking_data
        self.session = session
        (
            self.sheltertime,
            self.barriertime,
            self.barrierfliptime,
        ) = self.convert_experimental_settings_to_conditon_timings(session)

    # --------------- Abstract methods to be implemented by all children ---------------------------------------------

    @abstractmethod
    def merge_and_save_spike_count_df_with_frame_data(self):
        pass

    @abstractmethod
    def load_spike_data(self):
        pass

    # --------------- Concrete methods implemented ----------------------------------------------------------------------

    def convert_experimental_settings_to_conditon_timings(self, session):
        """
        Take the times inserted into the experimental class and convert them to seconds

        NOTE: The naming of sheltertime is not great as there is another variable called shelter_time in the session class. This is true for the other variables as well.
        consider renaming them to something more descriptive.
        """

        # Init variables incase they are not defined
        sheltertime = None
        barriertime = None
        barrierfliptime = None

        if session.shelter_time:
            sheltertime = np.array(self.session.shelter_time) * 60

        if session.barrier_time:
            barriertime = np.array(self.session.barrier_time) * 60

        if session.barrier_flip_time:
            barrierfliptime = np.array(self.session.barrier_flip_time) * 60

        return sheltertime, barriertime, barrierfliptime

    def extract_cluster_labels(self):
        """
        Extracts the cluster labels from the spike data dataframe and returns them
        """

        clu_label = self.spike_data.groupby(["spike_clusters"]).first()
        clu_label = clu_label.drop(["spike_aligned_to_frame", "spike_times", "aligned_spike_times", "aligned_spike_times_in_samples"])
        return clu_label

    def load_spike_data(self) -> pl.DataFrame:
        spike_data = pl.read_csv(self.csv_path)
        if len(spike_data.filter(spike_data["spike_clusters"] == 0)) > 0:
            spike_data = spike_data.with_column(spike_data["spike_clusters"] + 1)
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
        
        This function also saves so you can run just this function to regenerate it

        Returns: Video_df
        """
        # if mushroom, estend size to outer circle
        if np.logical_and(
            self.tracking_data["shelter_loc"][1][0] - self.tracking_data["shelter_loc"][0][0] < 50,
            self.tracking_data["shelter_loc"][1][1] - self.tracking_data["shelter_loc"][0][1] < 50,
        ):
            self.tracking_data["shelter_loc"][0] = [x - 35 for x in self.tracking_data["shelter_loc"][0]]
            self.tracking_data["shelter_loc"][1] = [x + 35 for x in self.tracking_data["shelter_loc"][1]]

        # if side shelter make sure it goes all the way to the edge of image, mouse can't be 'behind' shelter
        if self.tracking_data["shelter_loc"][1][1] > 900:
            self.tracking_data["shelter_loc"][1][1] = 1024

        OutofShelterIdx = out_of_shelter_filter(tracking_data=self.tracking_data)

        # when was the shelter in the arena?
        if len(self.session.shelter_time) > 0:
            if not (np.logical_and(self.session.shelter_time[0] == 0, self.session.shelter_time[1] == -1)):
                if self.session.shelter_time[1] == -1:  # shelter until the end of the session
                    shelter = np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.sheltertime[0] * self.session.video.fps)
                else:
                    shelter = np.logical_and(
                        np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.sheltertime[0] * self.session.video.fps),
                        np.arange(1, len(self.tracking_data["hdir"]) + 1) < (self.sheltertime[1] * self.session.video.fps),
                    )
            else:
                shelter = np.zeros(len(OutofShelterIdx)) == 0
                print("shelter always present")
        else:
            shelter = np.zeros(len(OutofShelterIdx)) == 1
            print("no shelter in this session")

        # what period in the recording was there a barrier?
        if len(self.session.barrier_time) > 0:
            if self.session.barrier_time[1] == -1:  # shelter only until the end of the session
                barrier_present = np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.barriertime[0] * self.session.video.fps)
            else:
                barrier_present = np.logical_and(
                    np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.barriertime[0] * self.session.video.fps),
                    np.arange(1, len(self.tracking_data["hdir"]) + 1) < (self.barriertime[1] * self.session.video.fps),
                )
        else:
            barrier_present = np.zeros(len(OutofShelterIdx)) == 1
            print("no barrier in this session")
        
        # Was the barrier removed during the session?
        if self.session.barrier_removal_time:
            # The defautl is False, so upuntil the barrier removal time frames are False, and then they are True
            barrier_removed = np.arange(1, len(self.tracking_data["hdir"]) + 1) > ((self.session.barrier_removal_time * 60) * self.session.video.fps)

        # when was the barrier flipped?
        if self.session.barrier_flip_time:
            barrier_flipped = np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.barrierfliptime * self.session.video.fps)
        else:
            barrier_flipped = np.zeros(len(OutofShelterIdx)) == 1
            print("barrier was not flipped in this session")

        # find the escape periods
        EscapePeriod = np.zeros_like(OutofShelterIdx)
        for onsets in self.session.audio.onset_frames:
            EscapePeriod[(onsets[0] - self.session.video.fps) : (onsets[0] + (10 * self.session.video.fps))] = 1

        # make a video dataframe where for each video frame:
        video_df = pl.DataFrame(
            {
                "frames": np.arange(1, len(self.tracking_data["hdir"]) + 1).astype(np.int64),
                "hdir": self.tracking_data["hdir"],
                "hsa": self.tracking_data["hdir_shelt"],
                "mouse_x_position": self.tracking_data["avg_loc"][:, 0],
                "mouse_y_position": self.tracking_data["avg_loc"][:, 1],
                "OutofshelterIdx": OutofShelterIdx,  # was the mouse in the shelter?
                "EscapePeriod": EscapePeriod == 1,  # frames from 1 second before to 10 seconds after escape
                "shelter": shelter,  # true when the shelter is in the arena
                "barrier_present": barrier_present,  # true when the barrier is in the arena
                "barrier_flipped": barrier_flipped,
                "barrier_removed": barrier_removed,
            }
        )  # true after the shelter was flipped

        # if barrier in session, add the angles to video_df
        if "hdir_barrier" in self.tracking_data:
            video_df = video_df.hstack([pl.Series("h_bar_north_a", self.tracking_data["hdir_barrier"][:, 0])])
            video_df = video_df.hstack([pl.Series("h_bar_south_a", self.tracking_data["hdir_barrier"][:, 1])])
            video_df = video_df.hstack([pl.Series("h_bar_centre_a", self.tracking_data["hdir_barrier"][:, 2])])

        # if random points were included, add the angles to video_df
        if "hdir_randP" in self.tracking_data:
            for i in np.arange(np.shape(self.tracking_data["hdir_randP"])[1]):
                video_df = video_df.hstack([pl.Series(str("head_randP_" + str(i)), self.tracking_data["hdir_randP"][:, i])])

        video_df.write_csv(os.path.join(self.session.base_path, self.session.processed_path) + "/" + "full_video_dataframe.csv")
        return video_df

    def count_spikes_and_units_to_frames(self) -> pl.DataFrame:
        """
        Uses polars query logic to map each cluster to a frame and count how many times each cluster fired in that frame.
        The lazy() function means that computations are not immediately executed. This allows the computer to plan the operations before
        proceeding. NOTE - if there are changes to the code, ensure to delete any existing CSVs so you can see the changes of the code.

        #NOTE - This logic seems suspciious, doesn't delete exsisting files when running the code again
        """

        try:
            logger.info("Attempting to load a previously computed spike frame count")
            with open(
                os.path.join(self.session.base_path, self.session.processed_path)
                + "/"
                + "spike_count_by_frame_and_"
                + self.select_cluster_labels
                + "cluster.csv",
                "rb",
            ) as file:
                spikecountbyframe_neuron = pl.read_csv(file.read())
            logger.success("Found spike count by frame and cluster dataframe, loading it now")
            return spikecountbyframe_neuron

        except FileNotFoundError:
            logger.info("Could not find spike count by frame and cluster dataframe, creating it now")
            logger.info("Commencing long computation to count spikes for each cluster for each frame")
            query = (
                self.spike_data.lazy()
                .groupby(["spike_aligned_to_frame", "spike_clusters"])
                .agg([pl.count("spike_aligned_to_frame").alias("spike_count")])
            )  # Lazy query to plan computation
            start_time = time.time()  # Collect lazy query and time it for user as this is the longest computation in the pipeline
            spikecountbyframe_neuron = query.collect()
            print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
            # spikecountbyframe_neuron.write_csv(os.path.join(self.session.base_path,self.session.processed_path) + "/" + "spike_count_by_frame_and_" + self.select_cluster_labels +"cluster.csv")
            return spikecountbyframe_neuron

    def merge_and_save_spike_count_df_with_frame_data(self, spikeCountByFrameAndCluster, video_df):
        logger.info("merging video df and spike df into a super df")
        video_df = video_df.select(
            [pl.col("frames").apply(float), pl.exclude("frames")]
        )  # Cast frames to float to permit join and remove old frames column with wrong type
        large_dataFrame = video_df.join(spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        large_dataFrame = large_dataFrame.fill_null(strategy="zero")  # this assigns some cluster IDs zero which is invalid!
        # large_dataFrame.write_csv(os.path.join(self.session.base_path,self.session.processed_path) + "/" + str(self.select_clusters) + "_large_dataframe.csv")
        return large_dataFrame

    def export_large_df_to_frame_by_cluster_matrix(self, spikeCountByFrameAndCluster, video_df) -> None:
        """
        This function takes the spike count by frame and cluster dataframe and extracts the spike count of each cluster on each frame,
        populating a large matrix.
        Additionally it uses a sliding window to estimate firing rate.
        Output: a matrix of frames x clusters of firing rates in Hz"""
        logger.info("building a frame by cluster matrix of firing rates")
        clu = spikeCountByFrameAndCluster["spike_clusters"].unique().to_numpy()
        # group the  data
        df = spikeCountByFrameAndCluster.groupby(["spike_aligned_to_frame"]).all()
        df = df.sort("spike_aligned_to_frame")

        # frames by firing per cluster matrix
        frames = video_df["frames"].unique().to_numpy().astype(int)

        X = np.zeros((np.amax(frames), len(clu)))

        for i2 in frames:
            d = df.filter(df["spike_aligned_to_frame"] == i2).to_dict(as_series=False)
            if len(d["spike_count"]) > 0:
                spikes = np.array(d.get("spike_count")[0])
                clusters = np.array(d.get("spike_clusters")[0])
                spikes = spikes[np.argsort(clusters)]
                clusters = np.sort(clusters)
                X[int(i2) - 1, np.where(np.in1d(clu, clusters))[0]] = spikes

        if clu[0] == 0:
            X = X[:, 1:]

        # transform to firing rate estimate in Hz
        sampling_rate = self.session.video.fps  # in fps
        window_size = 100  # in ms
        nbins = 1000 / window_size
        for i in np.arange(np.shape(X)[1]):
            X[:, i] = np.convolve(X[:, i], np.ones(int(sampling_rate / nbins), dtype=int), "same") * nbins

        # np.save(str(os.path.join(self.session.base_path,self.session.processed_path) + "/" + "frame_by_" + self.select_cluster_labels + "_cluster_matrix"), X)
        return X


class SyntheticDataPostprocessor(BaseDataPostprocessor):
    """
    A child class to support the synthetic data postprocessing pipeline.
    """

    def __init__(self, cluster_labels_to_filter, tracking_data, session, settings):
        super().__init__(cluster_labels_to_filter, tracking_data, session, settings)
        self.csv_path = os.path.join(session.base_path, session.processed_path, str(str(cluster_labels_to_filter) + "_efizz_data.csv"))
        self.select_clusters = cluster_labels_to_filter
        video_df = self.track_to_polars()
        if settings.efizz:
            self.check_synthetic_data_exists_if_not_generate_it(video_df)  # creates a csv in working dir
            self.spike_data = self.load_spike_data()
            self.clu_label = self.extract_cluster_labels()
            spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames()
            self.video_spike_count_df = self.merge_and_save_spike_count_df_with_frame_data(spikeCountByFrameAndCluster, video_df)
            self.frame_by_cluster_matrix = self.export_large_df_to_frame_by_cluster_matrix(spikeCountByFrameAndCluster, video_df)

    def check_synthetic_data_exists_if_not_generate_it(self, video_df) -> None:
        if not os.path.exists(self.csv_path):
            self.activate_synthetic_data_generation(video_df)
        else:
            logger.info("Synthetic spike data found")

    def activate_synthetic_data_generation(self, video_df) -> None:
        logger.info("Synthetic spike data doesn't exist and will now be generated")
        tuning = ["hdir"]
        if np.logical_or(
            np.logical_and(len(self.session.shelter_time) > 0, self.select_clusters == "synthetic"),
            "hsa" in self.select_clusters,
        ):
            tuning.append("hsa")
        if np.logical_and(len(self.session.barrier_time) > 0, self.select_clusters == "synthetic"):
            tuning.append("h_bar_north_a")
            tuning.append("h_bar_south_a")
        synth_df = generate_synthetic_dataframe(tuning, pass_video_df=video_df)
        synth_df.write_csv(self.csv_path)

    def expand_tracking_data(self, video_df: pl.DataFrame, new_entries_to_insert: int) -> pl.DataFrame:
        """
        Uniformly expands the tracking data by a specified number of entries to simulate a longer, perfectly sampled experiment.
        NOTE this function adds angles to hsa, barrier north and barrier south even if they don't exist in the data
        """

        # Generate polar series ranging from [last_frame_index + 1, last_frame_index + 1 + new_entries_to_insert]
        last_frame_index = video_df["frames"].max()
        new_frames = pl.Series(
            "frames", np.arange(last_frame_index + 1, last_frame_index + 1 + new_entries_to_insert).astype(np.int64)
        )  # Generate \ polar series ranging from [last_frame_index+1, last_frame_index+1+new_entries_to_insert]

        # Generate new angles sampled from a uniform distribution between -pi and pi for number of new entries
        angle_columns = ["hdir", "hsa", "h_bar_north_a", "h_bar_south_a"]
        new_angle_cols = [
            pl.Series(col, np.random.uniform(-np.pi, np.pi, new_entries_to_insert)) for col in angle_columns
        ]  # Create a list of polar series for each angle column

        # Generate new boolean columns with a specified value for number of new entries
        new_out_of_shelter_idx = pl.Series("OutofshelterIdx", np.full(new_entries_to_insert, fill_value=True))
        new_escape_period_idx = pl.Series("EscapePeriod", np.full(new_entries_to_insert, fill_value=False))
        new_shelter_only_idx = pl.Series("shelter_only", np.random.choice([True, False], size=new_entries_to_insert))
        new_barrier_present_idx = pl.Series("barrier_present", np.random.choice([True, False], size=new_entries_to_insert))

        # Generate new mouse position columns sampled from a uniform distribution between -1 and 1 for number of new entries
        min_x, max_x = min(video_df["mouse_x_position"]), max(video_df["mouse_x_position"])
        min_y, max_y = min(video_df["mouse_y_position"]), max(video_df["mouse_y_position"])
        new_mouse_x_position = pl.Series("mouse_x_position", np.random.uniform(min_x, max_x, new_entries_to_insert))
        new_mouse_y_position = pl.Series("mouse_y_position", np.random.uniform(min_y, max_y, new_entries_to_insert))

        # Generate new dataframe
        df_new = pl.DataFrame(
            [new_frames]
            + new_angle_cols
            + [new_mouse_x_position, new_mouse_y_position]
            + [new_out_of_shelter_idx, new_escape_period_idx, new_shelter_only_idx, new_barrier_present_idx]
        )

        # Concatenate new dataframe with original dataframe
        expanded_synthetic_tracking_data_by_frame = pl.concat([video_df, df_new])
        logger.success("Tracking data synthetically expanded by " + str(new_entries_to_insert) + " entries.")

        return expanded_synthetic_tracking_data_by_frame


class DataPostprocessor(BaseDataPostprocessor):
    """
    A child class to support the production data postrocessing pipeline.
    """

    def __init__(self, cluster_labels_to_filter, tracking_data, session, settings):
        super().__init__(cluster_labels_to_filter, tracking_data, session, settings)
        assert cluster_labels_to_filter != "synthetic", "Synthetic data is not supported by this class."
        self.csv_path = glob(os.path.join(session.base_path, session.processed_path, "Processed_efizz_data"))[0]
        self.select_clusters = cluster_labels_to_filter
        video_df = self.track_to_polars()
        if settings.efizz:
            unfiltered_spike_data = self.load_spike_data()
            self.spike_data = self.filter_spike_data(unfiltered_spike_data)
            self.clu_label = self.extract_cluster_labels()
            spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames()
            self.video_spike_count_df = self.merge_and_save_spike_count_df_with_frame_data(spikeCountByFrameAndCluster, video_df)

        # This is slow can we speed it up?
        # self.frame_by_cluster_matrix = self.export_large_df_to_frame_by_cluster_matrix(spikeCountByFrameAndCluster, video_df)

    def filter_spike_data(self, df):
        """
        Filter the spike data to only include good neurons or good + MUA
        """

        if self.select_clusters == "all":
            filtered_spike_data = df.filter((df["cluster_group"] == "good") | (df["cluster_group"] == "mua"))
            logger.info("Loaded good and multi unit clusters")
        else:
            filtered_spike_data = df.filter(df["cluster_group"] == self.select_clusters)
            numNeurons = len(filtered_spike_data["spike_clusters"].unique())
            logger.info(f"Loaded {numNeurons} {self.select_clusters} clusters")

        return filtered_spike_data
