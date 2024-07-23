import os
from abc import ABC, abstractmethod
from glob import glob
import time

from loguru import logger
import numpy as np
import polars as pl

from behave_analysis.database.synthetic_data.synthetic_main import generate_synthetic_dataframe
from behave_analysis.postprocess.out_of_shelter import out_of_shelter_filter
from behave_analysis.postprocess.trials.escapes import get_Escapes
from behave_analysis.utils.data_loading import load_or_extract_homings
from behave_analysis.homings.homings import Homings


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
        np.save(
            str(os.path.join(self.session.base_path, self.session.processed_path) + "/" + self.select_clusters + "_cluster_Ids"),
            clu_label["spike_clusters"].unique().to_numpy(),
        )
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

    def track_to_polars(self, homing_obj) -> pl.DataFrame:
        """Adds all a set of behavioral variables to a polars dataframe we call video_df.
        
        Args:
            homing_obj (obj class): An object representing the homing data for the session used to fill when homing frames are occuring

        The following columns are added to the video dataframe:
            -- frames
            -- hdir
            -- mouse_x_position
            -- mouse_y_position
            -- OutofshelterIdx (bool)
            -- EscapePeriod (bool)
            -- shelter (bool)
            -- barrier_present (bool)
            -- barrier_flipped (bool)

        Returns: 
            Video_df (pl.DataFrame)"""
        
        number_of_frames = len(self.tracking_data["hdir"])
         
        # if homing data is present, create a boolean array to indicate when homing is occuring in the session
        if homing_obj:
            homing_bool = np.zeros(number_of_frames, dtype=bool)
            onset_frames = homing_obj.onset_frames
            offset_frames = homing_obj.offset_frames
            for onset, offset in zip(onset_frames, offset_frames):
                homing_bool[onset - 1 : offset -1] = True
        else:
            homing_bool = [None] * number_of_frames
            
        if len(self.session.shelter_time) > 0:
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
        else:
            OutofShelterIdx = np.logical_not(np.zeros(len(self.tracking_data["hdir"])))

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
        else:
            shelter = np.zeros(len(OutofShelterIdx)) == 1

        # what period in the recording was there a barrier?
        if len(self.session.barrier_time) > 0:
            if self.session.barrier_time[1] == -1:  # barrier present until the end of the session
                barrier_present = np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.barriertime[0] * self.session.video.fps)
            else:
                barrier_present = np.logical_and(
                    np.arange(1, len(self.tracking_data["hdir"]) + 1) > (self.barriertime[0] * self.session.video.fps),
                    np.arange(1, len(self.tracking_data["hdir"]) + 1) < (self.barriertime[1] * self.session.video.fps),
                )
        else:
            barrier_present = np.zeros(len(OutofShelterIdx)) == 1
            print("no barrier in this session")

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
                "mouse_x_position": self.tracking_data["avg_loc"][:, 0],
                "mouse_y_position": self.tracking_data["avg_loc"][:, 1],
                "OutofshelterIdx": OutofShelterIdx,  # was the mouse in the shelter?
                "EscapePeriod": EscapePeriod == 1,  # frames from 1 second before to 10 seconds after escape
                "shelter": shelter,  # true when the shelter is in the arena
                "barrier_present": barrier_present,  # true when the barrier is in the arena
                "barrier_flipped": barrier_flipped,
                "homingPeriod": homing_bool,
            }
        )  # true after the shelter was flipped

        # if barrier in session, add the angles to video_df
        if "hdir_shelt" in self.tracking_data:
            video_df = video_df.hstack([pl.Series("hsa", self.tracking_data["hdir_shelt"])])

        # if barrier in session, add the angles to video_df
        # NOTE - North angle is always pre flip and south angle is always post flip for refactor to be pre and post flip edge
        if "hdir_barrier" in self.tracking_data:
            video_df = video_df.hstack([pl.Series("h_preflipbar_a", self.tracking_data["hdir_barrier"][:, 0])])
            video_df = video_df.hstack([pl.Series("h_postflipbar_a", self.tracking_data["hdir_barrier"][:, 1])])
            video_df = video_df.hstack([pl.Series("h_bar_centre_a", self.tracking_data["hdir_barrier"][:, 2])])

        # if random points were included, add the angles to video_df
        if "hdir_randP" in self.tracking_data:
            for i in np.arange(np.shape(self.tracking_data["hdir_randP"])[1]):
                video_df = video_df.hstack([pl.Series(str("head_randP_" + str(i)), self.tracking_data["hdir_randP"][:, i])])
                
        # save the video dataframe
        video_df.write_csv(os.path.join(self.session.base_path, self.session.processed_path) + "/" + "full_video_dataframe.csv")

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
            logger.info("This can take up to 1.5 hours depending on the size of the data")
            query = (
                self.spike_data.lazy()
                .groupby(["spike_aligned_to_frame", "spike_clusters"])
                .agg([pl.count("spike_aligned_to_frame").alias("spike_count")])
            )  # Lazy query to plan computation
            start_time = time.time()  # Collect lazy query and time it for user as this is the longest computation in the pipeline
            spikecountbyframe_neuron = query.collect()
            print("Time to query data and create spike count by frame and unit dataframe: ", time.time() - start_time)
            spikecountbyframe_neuron.write_csv(
                os.path.join(self.session.base_path, self.session.processed_path)
                + "/"
                + "spike_count_by_frame_and_"
                + self.select_cluster_labels
                + "cluster.csv"
            )
            return spikecountbyframe_neuron

    def merge_and_save_spike_count_df_with_frame_data(self, spikeCountByFrameAndCluster, video_df):
        """Merges the video dataframe with the spike count by frame and cluster dataframe and saves the result as a parquet file.
        
        Dataframe output:
        -- rows: frames
        -- columns: all angles, postition, spike counts, cluster ids"""
        logger.info("merging video df and spike df into a super df")
        video_df = video_df.select(
            [pl.col("frames").apply(float), pl.exclude("frames")]
        )  # Cast frames to float to permit join and remove old frames column with wrong type
        large_dataFrame = video_df.join(spikeCountByFrameAndCluster, left_on="frames", right_on="spike_aligned_to_frame", how="left")
        large_dataFrame = large_dataFrame.fill_null(strategy="zero")  # this assigns some cluster IDs zero which is invalid!
        large_dataFrame.write_parquet(os.path.join(self.session.base_path, self.session.processed_path + "/" + str(self.select_clusters) + "_video_spike_count_df.parquet"))
        return large_dataFrame

    def export_large_df_to_frame_by_cluster_matrix(self, spikeCountByFrameAndCluster, video_df) -> None:
        """
        This function takes the spike count by frame and cluster dataframe and extracts the spike count of each cluster on each frame,
        populating a large matrix.
        Additionally it uses a sliding window to estimate firing rate.
        Output: a matrix of frames x clusters of firing rates in Hz"""
        logger.info("Building a frame by cluster matrix of firing rates -- very slow")
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

        # save the matrix
        base_path = os.path.join(self.session.base_path, self.session.processed_path)
        np.save(
            str(base_path + "/" + "frame_by_" + self.select_cluster_labels + "_cluster_matrix"),
            X,
        )

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
        tuning = []
        if "hdir" in self.select_clusters:
            tuning.append("hdir")
        if np.logical_or(
            np.logical_and(len(self.session.shelter_time) > 0, self.select_clusters == "synthetic"),
            "hsa" in self.select_clusters,
        ):
            tuning.append("hsa")
        if np.logical_and(len(self.session.barrier_time) > 0, self.select_clusters == "synthetic"):
            tuning.append("h_preflipbar_a")
            tuning.append("h_postflipbar_a")
        synth_df = generate_synthetic_dataframe(tuning, pass_video_df=video_df)
        synth_df.write_csv(self.csv_path)

    def expand_tracking_data(self, video_df: pl.DataFrame, new_entries_to_insert: int) -> pl.DataFrame:
        """
        Uniformly expands the tracking data by a specified number of entries to simulate a longer, perfectly sampled experiment.
        NOTE this function adds angles to hsa, preflip barrier and postflip barrier even if they don't exist in the data
        """

        # Generate polar series ranging from [last_frame_index + 1, last_frame_index + 1 + new_entries_to_insert]
        last_frame_index = video_df["frames"].max()
        new_frames = pl.Series(
            "frames", np.arange(last_frame_index + 1, last_frame_index + 1 + new_entries_to_insert).astype(np.int64)
        )  # Generate \ polar series ranging from [last_frame_index+1, last_frame_index+1+new_entries_to_insert]

        # Generate new angles sampled from a uniform distribution between -pi and pi for number of new entries
        angle_columns = ["hdir", "hsa", "h_preflipbar_a", "h_postflipbar_a"]
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

    TODO: Check if sythetic class has diverged from this class and if so, update it.
    """

    def __init__(self, cluster_labels_to_filter, tracking_data, session, settings):
        super().__init__(cluster_labels_to_filter, tracking_data, session, settings)
        assert cluster_labels_to_filter != "synthetic", "Synthetic data is not supported by this class."
        self.csv_path = glob(os.path.join(session.base_path, session.processed_path, "Processed_efizz_data"))[0]
        self.select_clusters = cluster_labels_to_filter

        if np.logical_and(len(self.session.shelter_time) > 0, len(self.session.barrier_time) > 0):
            homings = load_or_extract_homings(session)
        
        # Handle if homings is not run run
        if not isinstance(homings, Homings): 
            homings = None
            
        video_df = self.track_to_polars(homings)
        QcPreProcessedData._check_for_vals_outside_arena(video_df, self.session) # For now just log the warning and don't touch the data
            
        if np.logical_and(len(self.session.shelter_time) > 0, len(self.session.barrier_time) > 0):
            _ = get_Escapes(settings, session, tracking_data, video_df, homings)

        if settings.efizz:
            unfiltered_spike_data = self.load_spike_data()
            self.spike_data = self.filter_spike_data(unfiltered_spike_data)
            self.clu_label = self.extract_cluster_labels()
            spikeCountByFrameAndCluster = self.count_spikes_and_units_to_frames()
            self.video_spike_count_df = self.merge_and_save_spike_count_df_with_frame_data(spikeCountByFrameAndCluster, video_df)
            self.frame_by_cluster_matrix = self.export_large_df_to_frame_by_cluster_matrix(spikeCountByFrameAndCluster, video_df) # This is slow can we speed it up?

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
        
        # save the filtered spike data
        filtered_spike_data.write_csv(os.path.join(self.session.base_path, self.session.processed_path) + "/" + self.select_cluster_labels + "_spike_data.csv")

        return filtered_spike_data


class QcPreProcessedData:
    """Quality control class for the preprocessed data.

    QC stands for quality control. The purpose of this class is to provide a set of methods that
    can be used to check the quality of the data produced by the preprocess class. In theory no
    data augmentation should occur after the preprocess class and thus this class should be the final
    quality check before any analysis is performed on the data.
    """

    @staticmethod
    def _check_for_vals_outside_arena(video_df, session) -> tuple[bool, np.ndarray]:
        """Log a warning if the tracking data is outside the bounds of the arena.

        Returns:
        -- tuple[bool, np.ndarray]: A tuple containing a boolean and a numpy array.
            The boolean is True if the tracking data is outside the bounds of the arena and False if the tracking data is within the bounds of the arena.
            The numpy array contains the distance of the mouse from the center of the arena."""
        
        # we're assuming  a square image!
        CENTER = session.video.height/2 # This is the pixel value of the center of the arena
        SIZE = session.video.height

        all_posX = video_df["mouse_x_position"].to_numpy()
        all_posY = video_df["mouse_y_position"].to_numpy()
        dist = np.sqrt(
            ((all_posX - SIZE / 2) ** 2) + ((all_posY - SIZE / 2) ** 2)
        )  # calculate the distance of mouse from the center
        assert len(dist) > 0, "The tracking data is empty"
        assert len(dist) == len(all_posX) == len(all_posY), "The tracking data is not the same length"
        if np.any(dist > CENTER):
            logger.warning(
                "The tracking data is outside the bounds of the arena. This is could be due to lighting issues, poor training of DLC or DLC tracking the cable."
            )
            return True, dist  # Return True if the tracking data is outside the bounds of the arena
        else:
            logger.success("The tracking data is within the bounds of the arena")
            return False, dist  # Return False if the tracking data is within the bounds of the arena

    # NOTE - Below function is not used in the pipeline yet, but it is a critical function in the pipeline to implement soon
    # Commented it out as need to agree on how we handle the tracking data outside the arena and PR has been stale for a while
    @staticmethod
    def handle_tracking_outside_arena(video_df, session, back_fill=False) -> pl.DataFrame:
        """If tracking outside arena replace with nulls or back fill them.

        As a default the function will replace the rows that are outside the bounds
        of the arena with the next valid value. This prevents the tracking data from
        being outside the arena. But this function alone does not prevent tracking errors.
        For example if the cable is tracked outside the arena for a while whilst the mouse is
        some where else entirely, it will just track the cable points inside the arena. Thus
        making sure the following two things are done is important:
        1) The DLC model is trained well, the tracking data is good and the cable is not tracked
        2) The body points of the mouse can not jump from one side of the arena to the other side of the arena

        Args:
        -- replace: (str) which replacement method to use. Options are "null" or "bfill".
            where "null" replaces the rows that are outside the bounds of the arena with Nans and
            "bfill" replaces the rows that are outside the bounds of the arena with the next valid value.

        Returns:
        -- pl.DataFrame: The video dataframe with the rows that are outside the bounds of
                the arena replaced with Nans. If no rows are outside the bounds of the arena
                then the original video dataframe is returned.

        TODO:
        -- Write a pytest for this function as it is a critical function in the pipeline.
        -- Handle the NaNs in the video dataframe that are now present after the replacement.
        """

        # we're assuming  a square image!
        CENTER = session.video.height/2 # This is the pixel value of the center of the arena
        SIZE = session.video.height

        is_outside_arena, dist = QcPreProcessedData._check_for_vals_outside_arena(video_df, session)

        # Check to see if the mouse is outside the bounds of the arena
        if is_outside_arena:
            logger.warning("Handling tracking data outside the bounds of the arena")

            # Find the indexs of the frames that are outside the bounds of the arena
            idx = np.where(dist > CENTER)[0]
            logger.info(f"{len(idx)} frames outside the bounds of arena out of {len(dist)} frames. -> {len(idx)/len(dist)*100:0.1f}% of the data.")

            # Replace the rows that are outside the bounds of the arena with Nans
            mask = pl.Series(np.arange(len(video_df))).is_in(pl.Series(idx))  # boolean mask
            assert sum(mask) == len(idx), "The mask Trues is not the same length as the index"

            # Insert nulls into the dataframe
            # When mask is true, then assign nans to the whole row, otherwise assign the original value
            for column in video_df.columns:
                video_df = video_df.with_column(pl.when(mask).then(pl.lit(None)).otherwise(pl.col(column)).alias(column))

            # If the user wants to back fill the nans then do so
            if back_fill:
                video_df = video_df.fill_null(strategy="backward")

            # Check again to see if the tracking data is within the bounds of the arena
            is_outside_arena, _ = QcPreProcessedData._check_for_vals_outside_arena(video_df)
            assert is_outside_arena == False, "The tracking data is still outside the bounds of the arena"

            # Because we back filled the frame numbers are now duplicated because rows are duplicated, so reset the index
            frames = pl.Series(np.arange(1, len(video_df) + 1).astype(np.int64))
            video_df = video_df.with_columns(pl.Series("frames", frames))

            logger.warning("some datapoints were outside the arena - so we're saving a new version of video_df")
            video_df.write_csv(os.path.join(session.base_path, session.processed_path) + "/" + "full_video_dataframe.csv")
            return video_df

        else:
            logger.success("The tracking data is within the bounds of the arena")
            return video_df

    # def _qc_video_data_is_populated(self) -> None:
    #     """
    #     A function that checks the video dataframe for any invalid values or states that could cause problems later in the pipeline.
    #     """

    #     assert False == any(self.preprocessed_data.video_df.null_count().to_numpy()[0] > 0), "The video dataframe contains null values."
    #     assert False == self.preprocessed_data.video_df.is_empty(), "The video dataframe is empty."
    #     testOfzeros = self.preprocessed_data.video_df.select(["frames", "hdir", "hsa", "mouse_x_position", "mouse_y_position"])
    #     assert False == any(testOfzeros.to_numpy()[0] == 0), "The video dataframe contains values that are equal to zero."

    # def qc_video_data_frame_schema_is_correct(dataframe: pl.DataFrame, session: object) -> None:
    #     if "mush" in session.name:
    #         assert dataframe.schema == {
    #             "frames": pl.Int64,
    #             "hdir": pl.Float64,
    #             "hsa": pl.Float64,
    #             "mouse_x_position": pl.Float64,
    #             "mouse_y_position": pl.Float64,
    #             "OutofshelterIdx": pl.Boolean,
    #             "EscapePeriod": pl.Boolean,
    #             "shelter_only": pl.Boolean,
    #             "barrier_present": pl.Boolean,
    #         }, "The video dataframe schema does match the expected schema, this could have unexpected consequences later in the pipeline."
    #     elif "seq" in session.name:
    #         assert dataframe.schema == {
    #             "frames": pl.Int64,
    #             "hdir": pl.Float64,
    #             "hsa": pl.Float64,
    #             "mouse_x_position": pl.Float64,
    #             "mouse_y_position": pl.Float64,
    #             "OutofshelterIdx": pl.Boolean,
    #             "EscapePeriod": pl.Boolean,
    #             "shelter_only": pl.Boolean,
    #             "barrier_present": pl.Boolean,
    #             "h_preflipbar_a": pl.Float64,
    #             "h_postflipbar_a": pl.Float64,
    #         }, "The video dataframe schema does match the expected schema, this could have unexpected consequences later in the pipeline."

    # def qc_angular_velocity_is_logically_possible(dataframe: pl.DataFrame, session: object) -> None:
    #     """

    #     Rough logic - The time it takes me to start a milisecond stop watch and stop it is 0.17 seconds. A frame is 0.025 seconds. It should
    #     be impossible for any angular change of a mouse to be 3 radians (171 degrees) in 0.025 seconds, this is a full spin.

    #     Emperical logic from .describe() - The mean across hdir, hsa, h_preflipbar_a and h_postflipbar_a is ┆ 0.012057 ┆ 0.014083 ┆ 0.012175 ┆ 0.012256
    #     highlighting that a delta of 3 radians would be a 300x increase in the mean and thus is unlikely to be a valid value.

    #     Angular velocity logic - The formula for angular velocity which is measured in radians per second is: delta radians / delta time
    #         + Δ3 radians / Δ0.025 seconds = 120 radians per second
    #         + Δ2 radians / Δ0.025 seconds = 80 radians per second
    #         + Δ1 radians / Δ0.025 seconds = 40 radians per second
    #         + Δ0.5 radians / Δ0.025 seconds = 20 radians per second
    #     Given 20 radians per second, that would mean the mouse would spin 10x in a second, which is not possible. So the maximum radial change
    #     proposed should be 0.5 radians per frame which is 50x the mean and thus is likely to be a unconvservative upper bound in error checking.

    #     """

    #     if "mush" in session.name:
    #         angular_columns = dataframe.select("hdir", "hsa")

    #     elif "seq" in session.name:
    #         angular_columns = dataframe.select("hdir", "hsa", "h_preflipbar_a", "h_postflipbar_a")

    #     # A function to calculate the circular distance between two angles - https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    #     f = lambda circular_angle_delta: np.pi - abs(abs(circular_angle_delta) - np.pi)

    #     # For all columns calculate the delta between each frame
    #     angular_delta = angular_columns.with_columns(pl.all().diff())
    #     circular_dist_delta = angular_delta.with_columns(pl.all().apply(f))
    #     logger.info(circular_dist_delta.describe())

    #     # Create expectation masks for each column
    #     failed_qc_hdir = np.where(circular_dist_delta["hdir"] > 0.5)[0]
    #     failed_qc_hsa = np.where(circular_dist_delta["hsa"] > 0.5)[0]
    #     failed_qc_preflip_barrier_edge = np.where(circular_dist_delta["h_preflipbar_a"] > 0.5)[0]
    #     failed_qc_postflip_barrier_edge = np.where(circular_dist_delta["h_postflipbar_a"] > 0.5)[0]

    #     if np.any(failed_qc_hdir) or np.any(failed_qc_hsa) or np.any(failed_qc_preflip_barrier_edge) or np.any(failed_qc_postflip_barrier_edge):
    #         logger.error(
    #             f"There are {len(failed_qc_hdir) + len(failed_qc_hsa) + len(failed_qc_preflip_barrier_edge) + len(failed_qc_postflip_barrier_edge)} frames that have a radial delta greater than 0.5 radians per 0.025 milliseconds"
    #         )

    #     # assert len(sum(np.where(delta_mask == True))) == 0, "The angular delta  of the mouse is greater than 3 radians in 0.025 seconds."
    #     # plt.hist(hdir_dif.to_numpy(), bins=100, log = True)
    #     # plt.title("Log hist showing the delta in hdir in radians from frame to frame")

    #     raise NotImplementedError
