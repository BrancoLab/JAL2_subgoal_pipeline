'''All the functions needed to make movies of all the mouse escapes in one session'''
# set up
import os
from loguru import logger
import cv2
import numpy as np

# import
from behave_analysis.utils.color_funcs import get_color_based_on_speed, get_colormap
from behave_analysis.utils.generate_stim_status_array import generate_stim_status_array
from behave_analysis.utils.directory import Directory
from behave_analysis.track.register import load_fisheye_correction_map, correct_and_register_frame

def trial_movies(tracking_data, kalman, session, settings, stim_type) -> None:
    """
    A function that loops through all of the trials of a given type, and then loops through frame by frame.
    """

    for trial_num, (onset_frames, stimulus_durations) in enumerate(
        zip(session.__dict__[stim_type].onset_frames, session.__dict__[stim_type].stimulus_durations)
    ):
        
        fisheye_correction_map = load_fisheye_correction_map(session.video)
        delay_between_frames = int(1000 / session.video.fps * (not settings.rapid) + settings.rapid)

        source_video, frames_in_this_trial, stim_status, trial_video = set_up_videos(session, settings, stim_type, trial_num, onset_frames, stimulus_durations)
        trail = []
        trail_colors = []
        trail_thicknesses = []

        # Loop through the frames in this trial
        for i in frames_in_this_trial:
            frame_num, actual_frame, num_frames_past_stim = read_frame(onset_frames, source_video)
            actual_frame = pass_correct_and_register_frame(actual_frame, settings, session, fisheye_correction_map)
            hdir_shelt, bod_shelt_dir, hdir_barrier, speed, avg_loc, head_loc, hdir = get_current_position_and_speed(tracking_data, settings, frame_num) # 
            display_stimulus(actual_frame, stim_status, settings, stim_type, avg_loc, i)
            trail, trail_colors, trail_thicknesses = display_trail(settings,actual_frame,trail, trail_colors, trail_thicknesses, avg_loc, stim_type, speed, num_frames_past_stim, stim_status[i]) # ready
            display_tracking(settings, actual_frame, avg_loc, speed, stim_type, hdir, head_loc, bod_shelt_dir, hdir_shelt, hdir_barrier, kalman, frame_num, i) # ready
            display_and_save_frames(actual_frame, stim_type, trial_video)
            key = cv2.waitKey(delay_between_frames)

            if key == ord("q") or key == ord("n"):
                break

        if key == ord("q"):
            break

    release_video_objects(source_video, trial_video)

# -----FIRST-LEVEL FUNCTIONS---------------------------------------------------------------------------------------

def read_frame(onset_frames, source_video):
    frame_num = int(source_video.get(cv2.CAP_PROP_POS_FRAMES))
    num_frames_past_stim = frame_num - onset_frames[0]
    successful_read, actual_frame = source_video.read()
    return frame_num, actual_frame, num_frames_past_stim

def pass_correct_and_register_frame(actual_frame, settings, session, fisheye_correction_map):
    actual_frame = correct_and_register_frame(
        actual_frame[:, :, 0], session.video, fisheye_correction_map
    )
    if settings.display_tracking or settings.display_trail:
        actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_GRAY2RGB)
    return actual_frame

def get_current_position_and_speed(tracking_data, settings, frame_num) -> None:
    """
    A function that gets the body direction, speed, and average location of the animal. Under the condition that
    the tracking data is being displayed, or the trail is displayed, or the stimulus is displayed.
    """

    if settings.display_tracking or settings.display_trail or settings.display_stimulus:
        hdir_shelt = tracking_data["hdir_shelt"][frame_num]
        bod_shelt_dir = tracking_data["bod_shelt_dir"][frame_num]
        if "bod_barrier_dir" in tracking_data:
            hdir_barrier = tracking_data["bod_barrier_dir"][frame_num, :]
        else:
            hdir_barrier = []
        speed = tracking_data["avg_Velocity"][frame_num]
        avg_loc = (
            int(tracking_data["avg_loc"][frame_num][0]),
            int(tracking_data["avg_loc"][frame_num][1]),
        )
        head_loc = (
            int(tracking_data["head_loc"][frame_num][0]),
            int(tracking_data["head_loc"][frame_num][1]),
        )
        hdir = tracking_data["hdir"][frame_num]
        return hdir_shelt, bod_shelt_dir, hdir_barrier, speed, avg_loc, head_loc, hdir

def display_stimulus(actual_frame, stim_status, settings, stim_type, avg_loc, i: int) -> None:
    """
    Display a large exclamation mark on the screen if the stimulus is on
    """

    if (
        settings.display_stimulus
        and stim_status[i] == 0
        and (stim_type == "audio" or (stim_type == "laser" and settings.display_tracking))
    ):
        if stim_type == "laser":
            exclamation_color = (255, 200, 0)
        else:
            exclamation_color = (100, 200, 255)
        cv2.putText(
            actual_frame,
            "!",
            (avg_loc[0] - 100, avg_loc[1] - 40),
            4,
            1.5,
            exclamation_color,
            thickness=6,
        )
        cv2.putText(
            actual_frame, "!", (avg_loc[0] - 100, avg_loc[1] - 40), 4, 1.5, (0, 0, 0), thickness=4
        )

def display_trail(settings,actual_frame,trail, trail_colors, trail_thicknesses, avg_loc, stim_type, speed, num_frames_past_stim, stim_status):
    if settings.display_trail:
        trail, trail_colors, trail_thicknesses = get_new_trail_segment(avg_loc, stim_type, speed, num_frames_past_stim, trail, trail_colors, trail_thicknesses, stim_status)
        display_all_trail_segments(actual_frame,trail, trail_colors, trail_thicknesses)
    return trail, trail_colors, trail_thicknesses

def display_tracking(settings, actual_frame, avg_loc, speed, stim_type, hdir, head_loc, bod_shelt_dir, hdir_shelt, hdir_barrier, kalman, frame_num, i):
    if settings.display_tracking:
        display_avg_location_on_frame(actual_frame, avg_loc)
        display_speed_on_frame(actual_frame, speed, stim_type)
        display_heading_dir_on_frame(actual_frame, hdir, head_loc, bod_shelt_dir, hdir_shelt)
        display_goal_dir_on_frame(actual_frame, head_loc, bod_shelt_dir, hdir_barrier)
        display_colored_dot_for_each_bodypart_on_frame(actual_frame, kalman, frame_num)
        # display_colored_dot_for_regions_on_frame(actual_frame, tracking_data, frame_num) # If you want to plot the regions of the body instead of the individual body parts

def display_and_save_frames(actual_frame, stim_type, trial_video):
    cv2.imshow("{} stimulus effect".format(stim_type), actual_frame)
    trial_video.write(actual_frame)

# -----TRACKING DISPLAY FUNCTIONS-----------------------------------------------------------------------------------

def display_avg_location_on_frame(actual_frame, avg_loc):
    cv2.circle(actual_frame, avg_loc, 3, (220, 220, 220), -1)

def display_speed_on_frame(actual_frame, speed, stim_type):
    """
    Print the speed of the animal on the displayed video.
    """
    # Set the color of the text based on the speed of the animal
    speed_text_color = get_color_based_on_speed(
        speed=speed, object_to_color="text", stim_status=None, stim_type=stim_type
    )

    # Print the speed on the frame
    cv2.putText(
        actual_frame,
        "{} cm/s".format(np.round(speed)),
        (actual_frame.shape[1] - 200, 45),
        0,
        1,
        speed_text_color,
        thickness=2,
    )

def display_heading_dir_on_frame(actual_frame, hdir, head_loc, bod_shelt_dir, hdir_shelt):
    """
    This code computes the x vector component and y vector component of an angle derived from two points on the animal.
    Currently the two points are looking at the upper and lower body and we will want to update this to have one for body
    direction and one for head direction. Or maybe just head direction.
    """
    magnitudeOfVector = 30  # This is the length of the arrow that will be plotted on the frame
    heading_dir_x = int(
        magnitudeOfVector * np.cos(hdir)  # self.body_dir
    )  # Convert the angle from radians to an x component
    heading_dir_y = -int(magnitudeOfVector * np.sin(hdir))  # Convert the angle from radians to an y component

    # Plot the heading direction on the frame centered at the animal's average location
    cv2.arrowedLine(
        actual_frame,
        head_loc,  # self.avg_loc
        (head_loc[0] + heading_dir_x, head_loc[1] + heading_dir_y),
        (220, 220, 220),
        1,
        16,
    )

    # Plot the body direction interger on the frame (for debugging)
    cv2.putText(
        actual_frame,
        f"HD: {str((hdir))}deg",
        (actual_frame.shape[1] - 200, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        actual_frame,
        f"BS: {str((bod_shelt_dir))}deg",
        (actual_frame.shape[1] - 200, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        actual_frame,
        f"HS: {str((hdir_shelt))}deg",
        (actual_frame.shape[1] - 200, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

def display_goal_dir_on_frame(actual_frame, head_loc, bod_shelt_dir, hdir_barrier):
    """
    This code shows you the angle between the mouse and its goals (shelter and barrier).
    """
    magnitudeOfVector = 30  # This is the length of the arrow that will be plotted on the frame

    # flip it around for arrow visualization purposes
    if bod_shelt_dir < 0:
        bs = bod_shelt_dir + np.pi
    if bod_shelt_dir > 0:
        bs = bod_shelt_dir - np.pi

    # plot a blue arrow in direction of shelter
    heading_dir_x = int(
        magnitudeOfVector * np.cos(bs)  # self.bod_shelt_dir
    )  # Convert the angle from radians to an x component
    heading_dir_y = -int(magnitudeOfVector * np.sin(bs))  # Convert the angle from radians to an y component
    # Plot the heading direction on the frame centered at the animal's average location
    cv2.arrowedLine(
        actual_frame,
        head_loc,  # self.avg_loc
        (head_loc[0] + heading_dir_x, head_loc[1] + heading_dir_y),
        (220, 0, 0),
        1,
        16,
    )

    # plot a green and red arrow in direction to two barrier edges (no arrows if no barrier)
    if np.any(hdir_barrier):  # bod_barr_dir
        cmap = [[0, 255, 0], [0, 0, 255]]
        for i in np.arange(
            2
        ):  # assuming two edges in barrier (we're not plotting the arrow to the center of the barrier)
            # flip it wround for arrow visualization purposes
            if hdir_barrier[i] < 0:
                bs = hdir_barrier[i] + np.pi
            if hdir_barrier[i] > 0:
                bs = hdir_barrier[i] - np.pi
            heading_dir_x = int(
                magnitudeOfVector * np.cos(bs)  # bod_barr_dir
            )  # Convert the angle from radians to an x component
            heading_dir_y = -int(magnitudeOfVector * np.sin(bs))  # Convert the angle from radians to an y component
            # Plot the heading direction on the frame centered at the animal's average location
            cv2.arrowedLine(
                actual_frame,
                head_loc,  # self.avg_loc
                (head_loc[0] + heading_dir_x, head_loc[1] + heading_dir_y),
                cmap[i],
                1,
                16,
            )

def display_colored_dot_for_each_bodypart_on_frame(actual_frame, kalman, frame_num) -> None:
    """
    A function that uses the individual bodypart tracking data from the kalman filter and plots it on the frame.
    """
    for j, (bodypart, color) in enumerate(zip(kalman, get_colormap())):
        bodypart_loc = (
            int(kalman[bodypart]["x"][frame_num]),
            int(kalman[bodypart]["y"][frame_num]),
        )
        cv2.circle(actual_frame, bodypart_loc, 1, color, -1)
        cv2.putText(
            actual_frame,
            bodypart,
            (actual_frame.shape[0] - 85, actual_frame.shape[1] - 280 + j * 20),
            0,
            0.4,
            color,
            thickness=1,
        )

def display_colored_dot_for_regions_on_frame(actual_frame, tracking_data, frame_num):
    """
    A function that loads the aggregated bodyparts from the tracking data from the kalman filter and plots it on the frame. Currently
    not used but leaving in for debugging purposes. NOTE you will need to change the test variable to include all the aggregated regions
    if interested in using this function. Remember that the kalman filter data and the tracking data have different formats.
    """

    test = ["head_loc", "upper_body_loc"]

    for j, (bodypart, color) in enumerate(zip(test, get_colormap())):
        bodypart_loc = (
            int(tracking_data[bodypart][frame_num, 0]),
            int(tracking_data[bodypart][frame_num, 1]),
        )
        cv2.circle(actual_frame, bodypart_loc, 1, color, -1)
        cv2.putText(
            actual_frame,
            bodypart,
            (actual_frame.shape[0] - 85, actual_frame.shape[1] - 280 + j * 20),
            0,
            0.4,
            color,
            thickness=1,
        )

def display_all_trail_segments(actual_frame,trail, trail_colors, trail_thicknesses):
    for j, (line, line_color, line_thickness) in enumerate(
        zip(trail, trail_colors, trail_thicknesses)
    ):
        if j:
            cv2.line(actual_frame, line, trail[j - 1], line_color, thickness=line_thickness, lineType=16)

def get_new_trail_segment(avg_loc, stim_type, speed, num_frames_past_stim, trail, trail_colors, trail_thicknesses, stim_status):
    time_to_get_new_trail_segment = num_frames_past_stim % 10 and (
        (stim_type in ["audio", "homing", "threshold_crossing"] and stim_status == 0)
        or (stim_type == "laser" and stim_status > -1 and stim_status < 3)
    )

    if time_to_get_new_trail_segment:
        trail_color = get_color_based_on_speed(
            speed=speed, object_to_color="trail", stim_status=stim_status, stim_type=stim_type
        )
        trail_colors.append(trail_color)
        trail.append(avg_loc)
        trail_thicknesses.append(int(stim_status != 0) + int(stim_type == "audio") + 1)
    return trail, trail_colors, trail_thicknesses

# ----SETUP FUNCTIONS-----------------------------------------------------------------------------------------------
def set_up_videos(session, settings,stim_type: str, trial_num: int, onset_frames: object, stimulus_durations: object):
    """
    A function that does a lot of shit
    """
    video_file = os.path.join(session.base_path, session.file_path, session.video.camFilePath)
    source_video = cv2.VideoCapture(video_file)  # Read the video file into a cv2 video object
    fps = session.video.fps

    seconds_before = settings.__dict__["seconds_before_" + stim_type]
    seconds_after = settings.__dict__["seconds_after_" + stim_type]
    frames_in_this_trial = range(
        (onset_frames[-1] - onset_frames[0])
        + int((seconds_before + stimulus_durations[-1] + seconds_after) * fps)
    )
    minutes_into_session = np.round(onset_frames[0] / fps / 60)

    source_video.set(
        cv2.CAP_PROP_POS_FRAMES, onset_frames[0] - seconds_before * fps
    )  # set source video to trial start
    stim_status = generate_stim_status_array(
        onset_frames, stimulus_durations, seconds_before, seconds_after, fps
    )
    # self.stim_status: 0~stimulus on, negative~pre stimulus, positive~post-stimulus

    trial_video_path = Directory(
        os.path.join(session.base_path, session.processed_path),
        experiment=session.experiment,
        stim_type=stim_type,
        tracking_video=settings.display_tracking,
        media_type="video",
    ).file_name(session.mouse, trial_num, minutes_into_session)

    trial_video = cv2.VideoWriter(
        trial_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        session.video.fps,
        (session.video.width, session.video.height),
        settings.display_tracking or settings.display_trail,
    )
    return source_video, frames_in_this_trial, stim_status, trial_video

def release_video_objects(source_video, trial_video):
    source_video.release()
    trial_video.release()
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

