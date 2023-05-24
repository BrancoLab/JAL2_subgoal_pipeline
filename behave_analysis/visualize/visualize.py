# Custom classes
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.track.register import load_fisheye_correction_map, correct_and_register_frame
from behave_analysis.utils.color_funcs import get_color_based_on_speed, get_colormap
from behave_analysis.utils.generate_stim_status_array import generate_stim_status_array
from behave_analysis.utils.directory import Directory
from behave_analysis.visualize.visualize_efizz import Visualize_efizz, PreProcess
from behave_analysis.visualize.visualize_behave import Visualize_behave

# OS libaries
from loguru import logger
import cv2
import numpy as np
import os
import dill as pickle

class Visualize:
    """
    A class that visualizes the tracking data of a session. Can be used to ensure that the tracking is working.
    The tracking data is loaded from prior pipeline step into a self.tracking_data
    """

    def __init__(self, session: object, settings: object):
        self.session = session        
        self.settings = settings
        self.fisheye_correction_map = load_fisheye_correction_map(session.video)
        self.delay_between_frames = int(1000 / self.session.video.fps * (not self.settings.rapid) + self.settings.rapid)

        self.print_session_details() # let us know which session we're doing

        # Load kalman tracking data
        file = os.path.join(self.session.processed_path, "kalman_tracking_data.pickle")
        with open(file, "rb") as dill_file:
            self.kalman = pickle.load(dill_file)

        open_tracking_data(self)

        # get time in minutes of shelter only and when the barrier was introduced
        if len(self.session.shelter_time) > 0: self.sheltertime = np.array(self.session.shelter_time)*60 # in seconds
        if len(self.session.barrier_time) > 0: self.barriertime = np.array(self.session.barrier_time)*60 # in seconds

        if self.settings.efizz:  # this will only make efizz plots if you want them
            logger.info(f"Starting to make some efizz overview plots...")
            
            """ Load synthetic data into visual object"""
            # Sythetic test run 
            # visualObject = Visualize_efizz(self, run= "Test")
            # spike_dictionary = visualObject.extract_trial_spikes(stim_type = 'Synth', onsets = "Synthetic_test_onsets", select_good_neurons = True)
            # visualObject.plot_single_cluster_raster(spikes_by_trials_and_cluster = spike_dictionary, stim_type = 'Synthetic_test')

            """ Load mouse brain data into visual object"""
            # Production - Run
            preprocessObject = PreProcess(self, run = "Production", select_clusters = "good", user_wants_to_regenerate_spike_by_frame_count = False)
            visualObject = Visualize_efizz(preprocessObject)
            
            """Make tuning plots"""
            # Production of vectorized plots  
            # compute_bootstrap: decide if you want to boostrap the rayleigh vector calculation
            # object_present: restrict analysis to times when the relevant object (i.e. shelter, barrier) is or is not in the arena

            visualObject.compute_a_single_tuning_for_all_cells('hdir', spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False)
            # visualObject.compute_a_single_tuning_for_all_cells('head_shelter_angle', compute_bootstrap = False, object_present = False) # NOTE - Don't use this one if the shelter is always present
            # visualObject.compute_a_single_tuning_for_all_cells('head_shelter_angle', spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False, object_present = True)
            # visualObject.compute_a_single_tuning_for_all_cells('head_south_barrier_angle', spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False, object_present = True)
            # visualObject.compute_a_single_tuning_for_all_cells('head_north_barrier_angle', spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False, object_present = True)
            # visualObject.compute_a_single_tuning_for_all_cells('head_south_barrier_angle', spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False, object_present = False)
            # visualObject.compute_a_single_tuning_for_all_cells('head_north_barrier_angle', spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False, object_present = False)

            # make a figure of all tuning polar plots for each cluster
            visualObject.compute_all_tunings_for_each_cell(spike_count_by_frame_and_neuron = preprocessObject.spikeCountByFrameAndCluster, compute_bootstrap = False) 

            # visualObject.spatial_position_firing() # TODO
            # TODO: build edge-tuning maps
            # TODO: tuning heatmap

            """Make plots of stimulus response"""
            # logger.info(f"Starting to make some plots of stimulus responses.")
            # if self.settings.escape_trials: visualObject.rasters(stim_type = 'audio')
            # if self.settings.escape_trials: visualObject.PSTH_all_neurons(stim_type = 'audio')
            # if self.settings.escape_trials: visualObject.PSTH_single_neurons(stim_type = 'audio')
            # if self.settings.escape_trials: visualObject.single_cluster_raster(stim_type = 'audio')
            
            """Laser sync test"""
            # Laser sync test TODO: check if this still works with new polars data organization
            # if self.settings.escape_trials: visualObject.single_cluster_raster_Laser_test()

        # logger.info(f"Starting to make some behaviour ONLY overview plots.")
        # BehaveObject = Visualize_behave(self)
        # BehaveObject.position_by_bsa()
        # BehaveObject.location_occupancy()
        # BehaveObject.angle_histograms()

    def trials(self, stim_type) -> None:
        """
        A function that loops through all of the trials of a given type, and then loops through frame by frame.
        """

        print("\nPress 'q' to quit and 'n' to move to the next video")

        for trial_num, (onset_frames, stimulus_durations) in enumerate(
            zip(self.session.__dict__[stim_type].onset_frames, self.session.__dict__[stim_type].stimulus_durations)
        ):
            self.set_up_videos(stim_type, trial_num, onset_frames, stimulus_durations)

            # Loop through the frames in this trial
            for i in self.frames_in_this_trial:
                self.read_frame(onset_frames)
                self.correct_and_register_frame()
                self.get_current_position_and_speed()
                self.display_stimulus(i)
                self.display_trail(i)
                self.display_tracking(i)
                self.display_and_save_frames()
                key = cv2.waitKey(self.delay_between_frames)

                if key == ord("q") or key == ord("n"):
                    break

            if key == ord("q"):
                break

        self.release_video_objects()

    # -----FIRST-LEVEL FUNCTIONS---------------------------------------------------------------------------------------

    def read_frame(self, onset_frames):
        self.frame_num = int(self.source_video.get(cv2.CAP_PROP_POS_FRAMES))
        self.num_frames_past_stim = self.frame_num - onset_frames[0]
        self.successful_read, self.actual_frame = self.source_video.read()

    def correct_and_register_frame(self):
        self.actual_frame = correct_and_register_frame(
            self.actual_frame[:, :, 0], self.session.video, self.fisheye_correction_map
        )
        if self.settings.display_tracking or self.settings.display_trail:
            self.actual_frame = cv2.cvtColor(self.actual_frame, cv2.COLOR_GRAY2RGB)

    def get_current_position_and_speed(self) -> None:
        """
        A function that gets the body direction, speed, and average location of the animal. Under the condition that
        the tracking data is being displayed, or the trail is displayed, or the stimulus is displayed.
        """

        if self.settings.display_tracking or self.settings.display_trail or self.settings.display_stimulus:
            self.body_dir = self.tracking_data["body_dir"][self.frame_num]
            self.bod_shelt_dir = self.tracking_data["bod_shelt_dir"][self.frame_num]
            if np.any(self.tracking_data["bod_barrier_dir"]):
                self.bod_barr_dir = self.tracking_data["bod_barrier_dir"][self.frame_num, :]
            else:
                self.bod_barr_dir = []
            self.speed = self.tracking_data["avg_Velocity"][self.frame_num]
            self.avg_loc = (
                int(self.tracking_data["avg_loc"][self.frame_num][0]),
                int(self.tracking_data["avg_loc"][self.frame_num][1]),
            )
            self.hdir = self.tracking_data["hdir"][self.frame_num]

    def display_stimulus(self, i: int) -> None:
        if (
            self.settings.display_stimulus
            and self.stim_status[i] == 0
            and (self.stim_type == "audio" or (self.stim_type == "laser" and self.settings.display_tracking))
        ):
            if self.stim_type == "laser":
                exclamation_color = (255, 200, 0)
            else:
                exclamation_color = (100, 200, 255)
            cv2.putText(
                self.actual_frame,
                "!",
                (self.avg_loc[0] - 100, self.avg_loc[1] - 40),
                4,
                1.5,
                exclamation_color,
                thickness=6,
            )
            cv2.putText(
                self.actual_frame, "!", (self.avg_loc[0] - 100, self.avg_loc[1] - 40), 4, 1.5, (0, 0, 0), thickness=4
            )

    def display_trail(self, i):
        if self.settings.display_trail:
            self.get_new_trail_segment(i)
            self.display_all_trail_segments()

    def display_tracking(self, i):
        if self.settings.display_tracking:
            self.display_avg_location_on_frame()
            self.display_speed_on_frame()
            self.display_heading_dir_on_frame()
            self.display_goal_dir_on_frame()
            self.display_colored_dot_for_each_bodypart_on_frame()
            # self.display_colored_dot_for_regions_on_frame() # If you want to plot the regions of the body instead of the individual body parts

    def display_and_save_frames(self):
        cv2.imshow("{} stimulus effect".format(self.stim_type), self.actual_frame)
        self.trial_video.write(self.actual_frame)

    # -----TRACKING DISPLAY FUNCTIONS-----------------------------------------------------------------------------------

    def display_avg_location_on_frame(self):
        cv2.circle(self.actual_frame, self.avg_loc, 3, (220, 220, 220), -1)

    def display_speed_on_frame(self):
        """
        Print the speed of the animal on the displayed video.
        """
        # Set the color of the text based on the speed of the animal
        speed_text_color = get_color_based_on_speed(
            speed=self.speed, object_to_color="text", stim_status=None, stim_type=self.stim_type
        )

        # Print the speed on the frame
        cv2.putText(
            self.actual_frame,
            "{} cm/s".format(np.round(self.speed)),
            (self.actual_frame.shape[1] - 200, 45),
            0,
            1,
            speed_text_color,
            thickness=2,
        )

    def display_heading_dir_on_frame(self):
        """
        This code computes the x vector component and y vector component of an angle derived from two points on the animal.
        Currently the two points are looking at the upper and lower body and we will want to update this to have one for body
        direction and one for head direction. Or maybe just head direction.
        """
        magnitudeOfVector = 30  # This is the length of the arrow that will be plotted on the frame
        self.body_dir = -self.body_dir  # Without this it doesn't work
        heading_dir_x = int(
            magnitudeOfVector * np.cos(self.body_dir)
        )  # Convert the angle from radians to an x component
        heading_dir_y = -int(
            magnitudeOfVector * np.sin(self.body_dir)
        )  # Convert the angle from radians to an y component

        # Plot the heading direction on the frame centered at the animal's average location
        cv2.arrowedLine(
            self.actual_frame,
            self.avg_loc,
            (self.avg_loc[0] + heading_dir_x, self.avg_loc[1] + heading_dir_y),
            (220, 220, 220),
            1,
            16,
        )

        # Plot the body direction interger on the frame (for debugging)
        # cv2.putText(self.actual_frame, f"{int(np.rad2deg(self.body_dir))}deg", (self.actual_frame.shape[1]-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def display_goal_dir_on_frame(self):
        """
        This code shows you the angle between the mouse and its goals (shelter and barrier).
        """
        magnitudeOfVector = 30  # This is the length of the arrow that will be plotted on the frame

        # plot a blue arrow in direction of shelter
        heading_dir_x = int(
            magnitudeOfVector * np.cos(self.bod_shelt_dir)
        )  # Convert the angle from radians to an x component
        heading_dir_y = -int(
            magnitudeOfVector * np.sin(self.bod_shelt_dir)
        )  # Convert the angle from radians to an y component
        # Plot the heading direction on the frame centered at the animal's average location
        cv2.arrowedLine(
            self.actual_frame,
            self.avg_loc,
            (self.avg_loc[0] + heading_dir_x, self.avg_loc[1] + heading_dir_y),
            (220, 0, 0),
            1,
            16,
        )
        # Plot the body direction interger on the frame (for debugging)
        cv2.putText(
            self.actual_frame,
            f"{int(np.rad2deg(self.bod_shelt_dir))}deg",
            (self.actual_frame.shape[1] - 200, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # plot a green and red arrow in direction to two barrier edges (no arrows if no barrier)
        if np.any(self.bod_barr_dir):
            cmap = [[0, 255, 0], [0, 0, 255]]
            for i in np.arange(2):  # assuming two edges in barrier
                heading_dir_x = int(
                    magnitudeOfVector * np.cos(self.bod_barr_dir[i])
                )  # Convert the angle from radians to an x component
                heading_dir_y = -int(
                    magnitudeOfVector * np.sin(self.bod_barr_dir[i])
                )  # Convert the angle from radians to an y component
                # Plot the heading direction on the frame centered at the animal's average location
                cv2.arrowedLine(
                    self.actual_frame,
                    self.avg_loc,
                    (self.avg_loc[0] + heading_dir_x, self.avg_loc[1] + heading_dir_y),
                    cmap[i],
                    1,
                    16,
                )

    def display_colored_dot_for_each_bodypart_on_frame(self) -> None:
        """
        A function that uses the individual bodypart tracking data from the kalman filter and plots it on the frame.
        """
        for j, (bodypart, color) in enumerate(zip(self.kalman, get_colormap())):
            bodypart_loc = (
                int(self.kalman[bodypart]["x"][self.frame_num]),
                int(self.kalman[bodypart]["y"][self.frame_num]),
            )
            cv2.circle(self.actual_frame, bodypart_loc, 1, color, -1)
            cv2.putText(
                self.actual_frame,
                bodypart,
                (self.actual_frame.shape[0] - 85, self.actual_frame.shape[1] - 280 + j * 20),
                0,
                0.4,
                color,
                thickness=1,
            )

    def display_colored_dot_for_regions_on_frame(self):
        """
        A function that loads the aggregated bodyparts from the tracking data from the kalman filter and plots it on the frame. Currently
        not used but leaving in for debugging purposes. NOTE you will need to change the test variable to include all the aggregated regions
        if interested in using this function. Remember that the kalman filter data and the tracking data have different formats.
        """

        test = ["head_loc", "upper_body_loc"]

        for j, (bodypart, color) in enumerate(zip(test, get_colormap())):
            bodypart_loc = (
                int(self.tracking_data[bodypart][self.frame_num, 0]),
                int(self.tracking_data[bodypart][self.frame_num, 1]),
            )
            cv2.circle(self.actual_frame, bodypart_loc, 1, color, -1)
            cv2.putText(
                self.actual_frame,
                bodypart,
                (self.actual_frame.shape[0] - 85, self.actual_frame.shape[1] - 280 + j * 20),
                0,
                0.4,
                color,
                thickness=1,
            )

    def display_all_trail_segments(self):
        for j, (line, line_color, line_thickness) in enumerate(
            zip(self.trail, self.trail_colors, self.trail_thicknesses)
        ):
            if j:
                cv2.line(self.actual_frame, line, self.trail[j - 1], line_color, thickness=line_thickness, lineType=16)

    def get_new_trail_segment(self, i):
        time_to_get_new_trail_segment = self.num_frames_past_stim % 10 and (
            (self.stim_type in ["audio", "homing", "threshold_crossing"] and self.stim_status[i] == 0)
            or (self.stim_type == "laser" and self.stim_status[i] > -1 and self.stim_status[i] < 3)
        )

        if time_to_get_new_trail_segment:
            trail_color = get_color_based_on_speed(
                speed=self.speed, object_to_color="trail", stim_status=self.stim_status[i], stim_type=self.stim_type
            )
            self.trail_colors.append(trail_color)
            self.trail.append(self.avg_loc)
            self.trail_thicknesses.append(int(self.stim_status[i] != 0) + int(self.stim_type == "audio") + 1)

    def print_session_details(self) -> None:
        logger.info("Commencing processing of sessions")
        for key in self.session.__dict__.keys():
            if key in ['name']:
                logger.info(" {}: {}".format(key, self.session.__dict__[key]))
        return None

    # ----SETUP FUNCTIONS-----------------------------------------------------------------------------------------------
    def set_up_videos(self, stim_type: str, trial_num: int, onset_frames: object, stimulus_durations: object):
        """
        A function that does a lot of shit
        """
        self.source_video = cv2.VideoCapture(
            self.session.video.video_file
        )  # Read the video file into a cv2 video object
        self.fps = self.session.video.fps

        self.stimulus_durations = stimulus_durations
        self.stim_type = stim_type

        self.onset_frames = onset_frames
        self.seconds_before = self.settings.__dict__["seconds_before_" + self.stim_type]
        self.seconds_after = self.settings.__dict__["seconds_after_" + self.stim_type]
        self.frames_in_this_trial = range(
            (onset_frames[-1] - onset_frames[0])
            + int((self.seconds_before + stimulus_durations[-1] + self.seconds_after) * self.session.video.fps)
        )
        minutes_into_session = np.round(onset_frames[0] / self.fps / 60)

        self.trail = []
        self.trail_colors = []
        self.trail_thicknesses = []

        self.source_video.set(
            cv2.CAP_PROP_POS_FRAMES, onset_frames[0] - self.seconds_before * self.session.video.fps
        )  # set source video to trial start
        self.stim_status = generate_stim_status_array(
            self.onset_frames, self.stimulus_durations, self.seconds_before, self.seconds_after, self.fps
        )
        # self.stim_status: 0~stimulus on, negative~pre stimulus, positive~post-stimulus

        trial_video_path = Directory(
            self.session.processed_path,
            experiment=self.session.experiment,
            stim_type=self.stim_type,
            tracking_video=self.settings.display_tracking,
            media_type="video",
        ).file_name(self.session.mouse, trial_num, minutes_into_session)

        self.trial_video = cv2.VideoWriter(
            trial_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.session.video.fps,
            (self.session.video.width, self.session.video.height),
            self.settings.display_tracking or self.settings.display_trail,
        )

    def release_video_objects(self):
        self.source_video.release()
        self.trial_video.release()
        cv2.destroyAllWindows()
