from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.track.register import load_fisheye_correction_map, correct_and_register_frame
from behave_analysis.utils.color_funcs import get_color_based_on_speed, get_colormap
from behave_analysis.utils.generate_stim_status_array import generate_stim_status_array
from behave_analysis.utils.directory import Directory
import cv2
import numpy as np

class Visualize():
    def __init__(self, session: object, settings: object):
        self.session = session
        self.settings = settings
        self.fisheye_correction_map = load_fisheye_correction_map(session.video)
        self.delay_between_frames = int(1000/self.session.video.fps*(not self.settings.rapid)+self.settings.rapid)
        open_tracking_data(self)

    def trials(self, stim_type):
        print("\nPress 'q' to quit and 'n' to move to the next video")
        for trial_num, (onset_frames, stimulus_durations) in enumerate(zip(self.session.__dict__[stim_type].onset_frames, self.session.__dict__[stim_type].stimulus_durations)):
            self.set_up_videos(stim_type, trial_num, onset_frames, stimulus_durations)            
            for i in self.frames_in_this_trial:
                self.read_frame(onset_frames)
                self.correct_and_register_frame()
                self.get_current_position_and_speed() 
                self.display_stimulus(i)
                self.display_trail(i)
                self.display_tracking(i)
                self.display_and_save_frames()
                key = cv2.waitKey(self.delay_between_frames)
                if key == ord('q') or key==ord('n'): break
            if key == ord('q'): break
        self.release_video_objects()

# -----FIRST-LEVEL FUNCTIONS---------------------------------------------------------------------------------------

    def read_frame(self, onset_frames):
        self.frame_num = int(self.source_video.get(cv2.CAP_PROP_POS_FRAMES))
        self.num_frames_past_stim = self.frame_num - onset_frames[0]
        self.successful_read, self.actual_frame = self.source_video.read()

    def correct_and_register_frame(self):
        self.actual_frame = correct_and_register_frame(self.actual_frame[:, :, 0], self.session.video, self.fisheye_correction_map)
        if self.settings.display_tracking or self.settings.display_trail: self.actual_frame = cv2.cvtColor(self.actual_frame, cv2.COLOR_GRAY2RGB)
    
    def get_current_position_and_speed(self):
        if self.settings.display_tracking or self.settings.display_trail or self.settings.display_stimulus:
            self.body_dir = self.tracking_data['body_dir'][self.frame_num]
            self.speed = self.tracking_data['speed'][self.frame_num]
            self.avg_loc = (int(self.tracking_data['avg_loc'][self.frame_num, 0]), int(self.tracking_data['avg_loc'][self.frame_num, 1]))

    def display_stimulus(self, i: int) -> None:
        if self.settings.display_stimulus and self.stim_status[i]==0 and (self.stim_type=='audio' or \
                                            (self.stim_type=='laser' and self.settings.display_tracking)):
            if self.stim_type == 'laser': exclamation_color = (255, 200, 0)
            else: exclamation_color = (100,200,255)
            cv2.putText(self.actual_frame, "!", (self.avg_loc[0] - 100, self.avg_loc[1] - 40), 4, 1.5, exclamation_color, thickness=6)
            cv2.putText(self.actual_frame, "!", (self.avg_loc[0] - 100, self.avg_loc[1] - 40), 4, 1.5, (0,0,0), thickness=4)

    def display_trail(self, i):
        if self.settings.display_trail:
            self.get_new_trail_segment(i)
            self.display_all_trail_segments()           

    def display_tracking(self, i):
        if self.settings.display_tracking:
            self.display_avg_location_on_frame()
            self.display_speed_on_frame()
            self.display_heading_dir_on_frame()
            self.display_colored_dot_for_each_bodypart_on_frame()

    def display_and_save_frames(self):
        cv2.imshow('{} stimulus effect'.format(self.stim_type), self.actual_frame)
        self.trial_video.write(self.actual_frame)

# -----TRACKING DISPLAY FUNCTIONS-----------------------------------------------------------------------------------

    def display_avg_location_on_frame(self):
        cv2.circle(self.actual_frame, self.avg_loc, 3, (220,220,220), -1)

    def display_speed_on_frame(self):
        speed_text_color = get_color_based_on_speed(speed=self.speed, object_to_color='text', stim_status=None, stim_type=self.stim_type)
        cv2.putText(self.actual_frame, '{} cm/s'.format(np.round(self.speed)), (self.actual_frame.shape[1]-200, 45), 0, 1, speed_text_color, thickness=2)

    def display_heading_dir_on_frame(self):
        heading_dir_x =  int(30*np.cos(np.deg2rad(self.body_dir)))
        heading_dir_y = -int(30*np.sin(np.deg2rad(self.body_dir)))
        cv2.arrowedLine(self.actual_frame, self.avg_loc, (self.avg_loc[0] + heading_dir_x, self.avg_loc[1] + heading_dir_y), (220,220,220), 1, 16)

    def display_colored_dot_for_each_bodypart_on_frame(self):
        for j, (bodypart, color) in enumerate(zip(self.tracking_data['bodyparts'], get_colormap())):
            bodypart_loc = (int(self.tracking_data[bodypart][self.frame_num, 0]), int(self.tracking_data[bodypart][self.frame_num, 1]))
            cv2.circle(self.actual_frame, bodypart_loc, 1, color, -1)
            cv2.putText(self.actual_frame, bodypart, (self.actual_frame.shape[0] - 85, self.actual_frame.shape[1] - 280 + j * 20), 0, .4, color, thickness=1)

    def display_all_trail_segments(self):
        for j, (line, line_color, line_thickness) in enumerate(zip(self.trail, self.trail_colors, self.trail_thicknesses)):
            if j: cv2.line(self.actual_frame, line, self.trail[j-1], line_color, thickness=line_thickness, lineType=16)
            
    def get_new_trail_segment(self, i):             
        time_to_get_new_trail_segment=self.num_frames_past_stim % 10 and \
                                    ((self.stim_type in ['audio','homing','threshold_crossing'] and self.stim_status[i]==0) or \
                                     (self.stim_type=='laser' and self.stim_status[i] > -1 and self.stim_status[i] < 3))
        if time_to_get_new_trail_segment:
            trail_color = get_color_based_on_speed(speed=self.speed, object_to_color='trail', stim_status=self.stim_status[i], stim_type=self.stim_type)
            self.trail_colors.append(trail_color)
            self.trail.append(self.avg_loc)
            self.trail_thicknesses.append(int(self.stim_status[i]!=0)+int(self.stim_type=='audio')+1)
     
# ----SETUP FUNCTIONS-----------------------------------------------------------------------------------------------
    def set_up_videos(self, stim_type: str, trial_num: int, onset_frames: object, stimulus_durations: object):
        self.stimulus_durations   = stimulus_durations
        self.stim_type            = stim_type
        self.onset_frames         = onset_frames
        self.fps                  = self.session.video.fps
        self.seconds_before       = self.settings.__dict__['seconds_before_' + self.stim_type]
        self.seconds_after        = self.settings.__dict__['seconds_after_' + self.stim_type]
        self.source_video         = cv2.VideoCapture(self.session.video.video_file)
        self.frames_in_this_trial = range((onset_frames[-1]-onset_frames[0])+int((self.seconds_before+stimulus_durations[-1]+self.seconds_after)*self.session.video.fps))
        minutes_into_session      = np.round(onset_frames[0]/self.fps/60)
        self.trail                = []
        self.trail_colors         = []
        self.trail_thicknesses    = []    
        self.source_video.set(cv2.CAP_PROP_POS_FRAMES, onset_frames[0]-self.seconds_before*self.session.video.fps) # set source video to trial start
        self.stim_status = generate_stim_status_array(self.onset_frames, self.stimulus_durations, self.seconds_before, self.seconds_after, self.fps)  
        #self.stim_status: 0~stimulus on, negative~pre stimulus, positive~post-stimulus

        trial_video_path = Directory(self.settings.save_folder, experiment=self.session.experiment, stim_type=self.stim_type, tracking_video=self.settings.display_tracking, media_type='video').file_name(self.session.mouse, trial_num, minutes_into_session)
        self.trial_video = cv2.VideoWriter(trial_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.session.video.fps, (self.session.video.width, self.session.video.height), self.settings.display_tracking or self.settings.display_trail)

    def release_video_objects(self):
        self.source_video.release()
        self.trial_video.release()
        cv2.destroyAllWindows()
