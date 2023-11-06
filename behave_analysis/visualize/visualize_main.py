# Custom classes

from behave_analysis.track.register import load_fisheye_correction_map, correct_and_register_frame
from behave_analysis.utils.color_funcs import get_color_based_on_speed, get_colormap
from behave_analysis.utils.generate_stim_status_array import generate_stim_status_array
from behave_analysis.utils.directory import Directory
# from behave_analysis.visualize.visualize_behave import Correlations

# Import custom settings

from settings.settings_visualize import defined_settings_visualize as settings

# OS libaries

from loguru import logger
import cv2
import numpy as np
import os
import dill as pickle

class Visualize:
    """
    A class that visualizes the tracking data of a session used to ensure that the tracking is working. It also does other random shit and needs to be refactored.
    As it also handles efizz data.
    """
    
    def __init__(self, session: object):
        self.session = session        
        self.settings = settings
        self.fisheye_correction_map = load_fisheye_correction_map(session.video)
        self.delay_between_frames = int(1000 / self.session.video.fps * (not self.settings.rapid) + self.settings.rapid)
        self.kalman = open_kalman_tracking_data(os.path.join(self.session.base_path,self.session.processed_path))
        self.print_session_details() # let us know which session we're doing
        self.postprocessObject = open_postprocess_object(self.session)
 
    def trial_movies(self, stim_type) -> None:
        """
        A function that loops through all of the trials of a given type, and then loops through frame by frame.
        """
        logger.info(f"Starting to make movies of mousie escape")
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
        self.actual_frame = correct_and_register_frame(self.actual_frame[:, :, 0], 
                                                       self.session.video, 
                                                       self.fisheye_correction_map)
        if self.settings.display_tracking or self.settings.display_trail:
            self.actual_frame = cv2.cvtColor(self.actual_frame, cv2.COLOR_GRAY2RGB)

    def get_current_position_and_speed(self) -> None:
        """
        A function that gets the body direction, speed, and average location of the animal. Under the condition that
        the tracking data is being displayed, or the trail is displayed, or the stimulus is displayed.
        """

        if self.settings.display_tracking or self.settings.display_trail or self.settings.display_stimulus:
            self.hdir_shelt = self.postprocessObject.tracking_data["hdir_shelt"][self.frame_num]
            self.bod_shelt_dir = self.postprocessObject.tracking_data["bod_shelt_dir"][self.frame_num]
            if 'bod_barrier_dir' in self.postprocessObject.tracking_data:
                self.hdir_barrier = self.postprocessObject.tracking_data["bod_barrier_dir"][self.frame_num, :]
            else:
                self.hdir_barrier = []
            self.speed = self.postprocessObject.tracking_data["avg_Velocity"][self.frame_num]
            self.avg_loc = (int(self.postprocessObject.tracking_data["avg_loc"][self.frame_num][0]),
                            int(self.postprocessObject.tracking_data["avg_loc"][self.frame_num][1]))
            self.head_loc = (int(self.postprocessObject.tracking_data["head_loc"][self.frame_num][0]),
                             int(self.postprocessObject.tracking_data["head_loc"][self.frame_num][1]))
            self.hdir = self.postprocessObject.tracking_data["hdir"][self.frame_num]

    def display_stimulus(self, i: int) -> None:
        """ 
        Display a large exclamation mark on the screen if the stimulus is on
        """
        
        if (self.settings.display_stimulus and 
            self.stim_status[i] == 0 and 
            (self.stim_type == "audio" or (self.stim_type == "laser" and self.settings.display_tracking))):
                if self.stim_type == "laser":
                    exclamation_color = (255, 200, 0)
                else:
                    exclamation_color = (100, 200, 255)
                cv2.putText(self.actual_frame, "!", (self.avg_loc[0] - 100, self.avg_loc[1] - 40), 4, 1.5, exclamation_color, thickness=6)
                cv2.putText(self.actual_frame, "!", (self.avg_loc[0] - 100, self.avg_loc[1] - 40), 4, 1.5, (0, 0, 0), thickness=4)

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
        heading_dir_x = int(
            magnitudeOfVector * np.cos(self.hdir) # self.body_dir
        )  # Convert the angle from radians to an x component
        heading_dir_y = -int(
            magnitudeOfVector * np.sin(self.hdir)
        )  # Convert the angle from radians to an y component

        # Plot the heading direction on the frame centered at the animal's average location
        cv2.arrowedLine(
            self.actual_frame,
            self.head_loc, # self.avg_loc
            (self.head_loc[0] + heading_dir_x, self.head_loc[1] + heading_dir_y),
            (220, 220, 220),
            1,
            16,
        )

        # Plot the body direction interger on the frame (for debugging)
        cv2.putText(self.actual_frame, f"HD: {str((self.hdir))}deg", (self.actual_frame.shape[1]-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.actual_frame, f"BS: {str((self.bod_shelt_dir))}deg", (self.actual_frame.shape[1]-200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.actual_frame, f"HS: {str((self.hdir_shelt))}deg", (self.actual_frame.shape[1]-200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def display_goal_dir_on_frame(self):
        """
        This code shows you the angle between the mouse and its goals (shelter and barrier).
        """
        magnitudeOfVector = 30  # This is the length of the arrow that will be plotted on the frame

        # flip it around for arrow visualization purposes
        if self.bod_shelt_dir < 0: bs =  self.bod_shelt_dir + np.pi
        if self.bod_shelt_dir > 0: bs =  self.bod_shelt_dir - np.pi

        # plot a blue arrow in direction of shelter
        heading_dir_x = int(
            magnitudeOfVector * np.cos(bs) # self.bod_shelt_dir
        )  # Convert the angle from radians to an x component
        heading_dir_y = -int(
            magnitudeOfVector * np.sin(bs)
        )  # Convert the angle from radians to an y component
        # Plot the heading direction on the frame centered at the animal's average location
        cv2.arrowedLine(
            self.actual_frame,
            self.head_loc, # self.avg_loc
            (self.head_loc[0] + heading_dir_x, self.head_loc[1] + heading_dir_y),
            (220, 0, 0),
            1,
            16,
        )

        # plot a green and red arrow in direction to two barrier edges (no arrows if no barrier)
        if np.any(self.hdir_barrier): # bod_barr_dir
            cmap = [[0, 255, 0], [0, 0, 255]]
            for i in np.arange(2):  # assuming two edges in barrier (we're not plotting the arrow to the center of the barrier)
                # flip it wround for arrow visualization purposes
                if self.hdir_barrier[i] < 0: bs =  self.hdir_barrier[i] + np.pi
                if self.hdir_barrier[i] > 0: bs =  self.hdir_barrier[i] - np.pi
                heading_dir_x = int(
                    magnitudeOfVector * np.cos(bs) # bod_barr_dir
                )  # Convert the angle from radians to an x component
                heading_dir_y = -int(
                    magnitudeOfVector * np.sin(bs)
                )  # Convert the angle from radians to an y component
                # Plot the heading direction on the frame centered at the animal's average location
                cv2.arrowedLine(
                    self.actual_frame,
                    self.head_loc, # self.avg_loc
                    (self.head_loc[0] + heading_dir_x, self.head_loc[1] + heading_dir_y),
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
                int(self.postprocessObject.tracking_data[bodypart][self.frame_num, 0]),
                int(self.postprocessObject.tracking_data[bodypart][self.frame_num, 1]),
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
        video_file = os.path.join(self.session.base_path,self.session.file_path,self.session.video.camFilePath)
        self.source_video = cv2.VideoCapture(video_file)  # Read the video file into a cv2 video object
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
            os.path.join(self.session.base_path,self.session.processed_path),
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


    # def get_current_position_and_speed_old(self) -> None:
        """
        A function that gets the body direction, speed, and average location of the animal. Under the condition that
        the tracking data is being displayed, or the trail is displayed, or the stimulus is displayed.
        """

        # if self.settings.display_tracking or self.settings.display_trail or self.settings.display_stimulus:
        #     self.body_dir = self.tracking_data["body_dir"][self.frame_num]
        #     self.bod_shelt_dir = self.tracking_data["bod_shelt_dir"][self.frame_num]
        #     if np.any(self.tracking_data["bod_barrier_dir"]):
        #         self.bod_barr_dir = self.tracking_data["bod_barrier_dir"][self.frame_num, :]
        #     else:
        #         self.bod_barr_dir = []
        #     self.speed = self.tracking_data["avg_Velocity"][self.frame_num]
        #     self.avg_loc = (
        #         int(self.tracking_data["avg_loc"][self.frame_num][0]),
        #         int(self.tracking_data["avg_loc"][self.frame_num][1]),
        #     )
        #     self.hdir = self.tracking_data["hdir"][self.frame_num]


# ------------------------------------------------------------------Utilities ----------------------------------------------------------------

# TODO: move to new script 

# Utiliy functions for visualise class
def open_kalman_tracking_data(path):
    try:
        file = os.path.join(path, "kalman_tracking_data.pickle")
        with open(file, "rb") as dill_file:
            kalman = pickle.load(dill_file)
        return kalman
            
    except FileNotFoundError:
        logger.error(f"Kalman tracking data not found for this session")
        raise FileNotFoundError
    
def open_postprocess_object(session) -> object:
    try:
        fileObj = open(os.path.join(session.base_path, session.processed_path) + "\\" + "postprocessclass" + "_" + str(settings.cluster_type), 'rb')
        postprocessObject = pickle.load(fileObj)
        fileObj.close()
        return postprocessObject
        
    except FileNotFoundError:
        logger.error(f"Data not found for session: {session.name} - Check databank and whether you have actually run this configuration of postprocess. ")
        raise FileNotFoundError