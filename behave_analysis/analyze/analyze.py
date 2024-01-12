# Custom libs
# from behave_analysis.utils.mat_to_python import convert_matlab_struct
from behave_analysis.process.camera_trigger import get_num_frames_expected, get_Camera_trigger
from behave_analysis.process.process import Process
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.analyze.plot_funcs import *
from behave_analysis.analyze.data_extraction_funcs import *
from behave_analysis.analyze.stats_funcs import permutation_test, print_stat_test_results
from behave_analysis.analyze.trial_eligibility_funcs import trial_is_eligible
from behave_analysis.utils.directory import Directory
from behave_analysis.analyze.grid_cells.grid_cell_funcs import mother_plot

#Custom libs
from settings.settings_process import settings_process as settings_p # So to check if pipeline includes efizz

# # Additional libraries if running with efizz
# if settings_p.efizz:
#     from behave_analysis.utils.downsample_AI_data import remove_idx_as_per_bonsai_ttl_resample

# OS libs
import mpl_toolkits.axisartist.floating_axes as floating_axes
from ephysiopy.common.binning import RateMap # Trying robin's rate map function
from matplotlib.transforms import Affine2D
from ephysiopy.visualise.plotting import FigureMaker # Trying robin's rate map function
from matplotlib.backends.backend_pdf import PdfPages # For saving to a pdf
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np
import resampy
from loguru import logger
import pickle
import os
import time

class Analyze():
    def __init__(self, session_IDs, settings, analysis_type):
        self.settings = settings
        self.session_IDs = session_IDs
        self.session_count = -1
        self.trial_num = 0
        self.avg_pos = []
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.data_session_num = np.array([])
        self.trial_colors   = []
        self.trials_to_plot = []
        self.group_nums = np.unique(session_IDs[:, 5])
        self.num_of_groups = np.ptp(self.group_nums)+1 
        self.analysis_type = analysis_type   
        self.title = self.settings.analysis.title
        self.color_by = self.settings.color_by
        if 'traject' in analysis_type and not self.color_by in ['speed', 'speed+RT','time','target', 'session','trial','']:
            if 'escape' in analysis_type:  self.color_by = 'target'
            if 'homing' in analysis_type:  self.color_by = 'speed'
            if 'laser'  in analysis_type:  self.color_by = 'time'
            if 't xing'  in analysis_type: self.color_by = 'speed'
            if 'trial'   in analysis_type: self.color_by = 'speed'
            if 'explor' in analysis_type: self.color_by = 'speed'
        if 'target'  in analysis_type and not self.color_by in ['target', 'session','trial','']:
            self.color_by = 'target'
        if self.settings.leftside_only:  self.title += " (leftside)"
        if self.settings.rightside_only: self.title += " (rightside)"   
        if 'traject' in analysis_type and self.settings.reflect_trajectories: self.title += " (reflect)"
        if 'escape' in analysis_type: self.stim_type='audio'
        if 'homing' in analysis_type: self.stim_type='homing'
        if 'laser' in analysis_type:  self.stim_type='laser'
        if 't xing' in analysis_type: self.stim_type='threshold_crossing'
        if 'explor' in analysis_type: self.stim_type='explore'

# ----MAIN METHODS------------------------------------------------------
    def trajectories(self):
        self.extract_data()
        self.initialize_trajectory_plot()
        for trial in self.trials_to_plot:
            print(len(trial['trajectory x']))
            self.plot_trajectory(trial)
        self.save_plot()

    def single_trial(self):
        self.extract_data()
        for trial in self.trials_to_plot:
            self.initialize_trajectory_plot()
            self.plot_single_trial(trial)
            self.save_plot()

    def distribution(self):
        self.extract_data()
        self.do_statistics()
        self.initialize_data_plot()
        self.plot_boxplot()
        self.plot_scatterplot()
        self.save_plot()

    def exploration(self):
        self.extract_data()
        position_data = self.tracking_data
        self.avg_pos = position_data['avg_loc'] # The average position of the animal across the session
        self.interp_position() # Interpolate posiiton data and extend to fs of 30khz
        mother_plot(self)
        # for trial in self.trials_to_plot:
        #     print(trial['trajectory x'])
        #     self.initialize_trajectory_plot() # plot an empty circle w/wo an obstacle
        #     self.plot_trajectory(trial)
        #     # self.overlay_spikes() #Use this function to plot spikes onto trajectory
        #     self.save_plot()

# ----DATA EXTRACTION FUNCS---------------------------------------------
    def extract_data(self):
        for session_ID in self.session_IDs:
            self.trial_num = 0
            self.minutes_into_session = None
            self.open_session_data(session_ID) 
            if not 'explor' in self.analysis_type:
                self.get_data_on_each_trial()
                return
            self.get_start_and_end_frames()

    def open_session_data(self, session_ID):
        self.session = Process(session_ID).load_session()
        self.group_num = session_ID[5]
        self.fps = self.session.video.fps
        self.session_count += 1
        self.num_successful_escapes_this_session = 0
        self.tracking_data = open_tracking_data(self.session)

    def get_data_on_each_trial(self):
        for onset_frames, stim_durations in zip(self.session.__dict__[self.stim_type].onset_frames, \
                                                self.session.__dict__[self.stim_type].stimulus_durations):
            if not trial_is_eligible(self, onset_frames): 
                continue
            self.generate_trial_dict(onset_frames, stim_durations)
            self.trial_num+=1

    def generate_trial_dict(self, onset_frames: list, stim_durations: list):
        trial_start_idx, trial_end_idx = get_trial_start_and_end(self, onset_frames)
        trial                          = create_trial_dict(self, trial_start_idx, trial_end_idx)
        self.trials_to_plot.append(trial)

    def get_start_and_end_frames(self):
        session_start = 0
        camera_trigger_data = get_Camera_trigger(self.session, 
                                                 self.session.ttl.choose_index, 
                                                 self.session.ttl.temporal_difference,
                                                 self.session.ttl.bonsai_TTL)[1]
        session_end = get_num_frames_expected(self.session, camera_trigger_data)[0]
        trial = create_trial_dict(self, session_start, session_end)
        self.trials_to_plot.append(trial)

        
# ----STATISTICS FUNCS--------------------------------------------------
    def do_statistics(self):
        self.compile_all_trials_data()
        self.run_permutation_tests()

    def compile_all_trials_data(self):
        for trial in self.trials_to_plot:
            self.data_x           = np.append(self.data_x, trial['group number'])
            self.data_y           = np.append(self.data_y, trial['escape target score'])
            self.data_session_num = np.append(self.data_session_num, trial['session count'])
            self.trial_colors.         append(get_plot_color(self, trial, plot_type='scatter'))

    def run_permutation_tests(self):
        for group_num in self.group_nums:
            if group_num <= 1: continue
            if self.settings.binarize_statistics:     data_for_stat_test = self.data_y > self.settings.edge_vector_threshold
            if not self.settings.binarize_statistics: data_for_stat_test = self.data_y
            p = permutation_test(data_for_stat_test,self.data_x,self.data_session_num,group_1=1,group_2=group_num,iterations=1000,two_tailed=self.settings.two_tailed_test)
            print_stat_test_results(p, self.analysis_type, self.settings.two_tailed_test, self.settings.binarize_statistics, group_1=1, group_2=group_num,)
       
# ----PLOTTING DATA-----------------------------------------------------
    def initialize_data_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.num_of_groups*2, 9))
        self.fig.canvas.set_window_title(self.title) 
        self.ax.set_ylim([-.2, 1.2])
        self.x_range = [min(self.group_nums)-.6, max(self.group_nums)+.6]
        self.ax.set_xlim(self.x_range)
        plt.plot(self.x_range, [0,0],     color=(.9,.9,.9), linestyle='--', zorder=-1)
        plt.plot(self.x_range, [1,1],     color=(.9,.9,.9), linestyle='--', zorder=-1)
        plt.plot(self.x_range, [self.settings.edge_vector_threshold,self.settings.edge_vector_threshold], color=(.9,.9,.9), linestyle='--', zorder=-1)
        format_axis(self)

    def plot_scatterplot(self):
        apply_x_jitter(self, offset_x=0, min_distance_y=0.01, jitter_distance_x=0.01 + .002 * self.num_of_groups)
        self.ax.scatter(self.jittered_data_x, self.data_y, color=self.trial_colors, linewidth=0, s=35, zorder=99)

    def plot_boxplot(self, width=.25):
        for group_num in self.group_nums:
            group_data_y = self.data_y[self.data_x==group_num]
            quartile_1, median, quartile_3 = np.percentile(group_data_y, [25, 50, 75])
            iqr = quartile_3 - quartile_1
            lower_range = max(min(group_data_y), quartile_1-1.5*iqr)
            upper_range = min(max(group_data_y), quartile_3+1.5*iqr)
            color = (.85,.85,.85)

            whiskers = self.ax.plot([group_num,group_num], [lower_range, upper_range], color=color, linewidth=1)
            boxplot = plt.Rectangle((group_num-width/2, quartile_1), width, iqr, color=color, edgecolor=None, fill=True)
            self.ax.add_artist(boxplot)

            median_line = self.ax.plot([group_num-width/2.15, group_num+width/2.1], [median, median], color=(0,0,0), linewidth=3)

    def save_plot(self):
        plt.show()
        for file_extension in ['.png', '.eps']:
            plot_path = Directory(self.settings.save_folder, experiment=self.session.experiment, analysis_type=self.analysis_type, stim_type = self.stim_type, media_type='plot').\
                        file_name(self.mouse, self.trial_num, self.minutes_into_session, self.title, self.color_by, file_extension)
            self.fig.savefig(plot_path, bbox_inches='tight', pad_inches=0) 

# ----PLOTTING TRAJECTORIES---------------------------------------------
    def initialize_trajectory_plot(self):
        """A function that inits a matplotlib fig, plots an outer black circle and an obstacle based on
        the registration size. Obstacle is only plotted if defined by settings.
        """
        size = self.session.video.registration_size # Define circle plot size from registration
        self.fig, self.ax = plt.subplots(figsize=(9,9)) # Instantiate trajectory plot
        self.ax.set_xlim([0, size[0]]) # Define x size of circle
        self.ax.set_ylim([0, size[1]]) # Define y size of circle 
        if self.stim_type in ['laser', 'homing', 'threshold_crossing']: # If settings chosen
            self.ax.plot([size[0]/2-250, size[1]/2+250], [size[0]/2, size[1]/2], color=[0, 0, 0], linewidth=5) # plot obstacle
        circle = plt.Circle((size[0]/2, size[1]/2), radius=460, color=[0, 0, 0], linewidth=1, fill=False) # set circle parameters
        self.ax.add_artist(circle) # add the circle border to the plot
        self.ax.invert_yaxis()
        format_axis(self) # formatting func found in plot_funcs

    def plot_single_trial(self, trial):
        self.plot_trajectory(trial)
        self.plot_silhouettes(trial)
        self.trial_num            = trial['trial count']
        self.minutes_into_session = np.round(trial['trial start'] / self.session.video.fps / 60) 

    def plot_trajectory(self, trial):

        # if color_by in settings_analyze is defined have a gradient line
        if self.color_by in ['speed', 'speed+RT','time']: 
            gradient_line(self, trial) # A func found in plot_funcs
        # else color the trajectory with a solid line
        else:                                             
            solid_line(self, trial)   
        self.mouse = trial['mouse']
        self.session.experiment = trial['experiment']   

    def plot_silhouettes(self, trial, mouse_size: float=38, color: tuple=(.7,.7,.7), num_silhouettes=6):

        colors = generate_list_of_colors(self.color_by, self.stim_type, trial['speed'], RT=trial['escape initiation idx'], object_to_color='trial')
        frames_to_illustrate = np.concatenate((np.zeros(1, dtype=int), np.linspace(trial['escape initiation idx'], trial['escape end idx'] - trial['trial start'] - 2, num=num_silhouettes, dtype=int)))

        for i, idx in enumerate(frames_to_illustrate):

            color = colors[idx] 
            if i==0: color = (.7,.7,.7)

            head_ellipse       = ptch.Ellipse(trial['head_loc'][idx, :],       width = int(mouse_size*.60), height = int(mouse_size * .23), \
                                      angle = trial['head_dir'][idx],       color=color, alpha=1, edgecolor=None)
            neck_ellipse       = ptch.Ellipse(trial['neck_loc'][idx, :],       width = int(mouse_size*.80), height = int(mouse_size * .38), \
                                      angle = trial['neck_dir'][idx],       color=color, alpha=1, edgecolor=None)
            shoulder_ellipse   = ptch.Ellipse(trial['shoulder_loc'][idx, :],   width = int(mouse_size*.50), height = int(mouse_size * .45), \
                                      angle = trial['shoulder_dir'][idx],   color=color, alpha=1, edgecolor=None)
            upper_body_ellipse = ptch.Ellipse(trial['upper_body_loc'][idx, :], width = int(mouse_size*.50), height = int(mouse_size * .45), \
                                      angle = trial['upper_body_dir'][idx], color=color, alpha=1, edgecolor=None)
            lower_body_ellipse = ptch.Ellipse(trial['lower_body_loc'][idx, :], width = int(mouse_size*.50), height = int(mouse_size * .45), \
                                      angle = trial['lower_body_dir'][idx], color=color, alpha=1, edgecolor=None)
            body_ellipse       = ptch.Ellipse(trial['body_loc'][idx, :],       width = int(mouse_size*.95), height = int(mouse_size * .58), \
                                      angle = trial['body_dir'][idx],       color=color, alpha=1, edgecolor=None)

            self.ax.add_artist(head_ellipse)
            self.ax.add_artist(neck_ellipse)
            self.ax.add_artist(shoulder_ellipse)
            self.ax.add_artist(upper_body_ellipse)
            self.ax.add_artist(lower_body_ellipse)
            self.ax.add_artist(body_ellipse)

# -------------- Interpolate behavioural data to match efiz

    # Interpolate the position data so it's the same length as the ephys 
    def interp_position(self):
        """A function that takes in the fps of the camera, position data, and fs of signal and interpolates
        both x and y positions and then saves it as a pickle file

        #Note!! if you want to re-do the interpolation you will need to delete the pickle file
        """

        # Time interpolation
        start_time = time.time()

        # Retrieve paths and set new path
        self.interp_path = self.settings.efiz_file_path + "interpolated_data.pkl"

        # See if interp dic already exsists and if so break
        if os.path.isfile(self.interp_path) == True:
            logger.info("Interpolation file already exsists")
            return

        #Params
        fps = self.session.video.fps
        desired_fs = 30000

        # Data
        speed = self.tracking_data['speed']
        position = self.avg_pos
        hdir = self.tracking_data['neck_dir'] # two ears and upper back to use as hdir

        # Interp
        logger.info("Commencing interpolation, processing may hang. Should take 5-10 minutes")
        self.interp_x     = resampy.resample(position[:,0], fps, desired_fs)
        self.interp_y     = resampy.resample(position[:,1], fps, desired_fs)
        self.interp_speed = resampy.resample(speed, fps, desired_fs)
        self.hdir         = resampy.resample(hdir, fps, desired_fs)
        
        interp_dic = {"x": self.interp_x,
                      "y": self.interp_y,
                      "speed" : self.interp_speed,
                      "hdir"  : self.hdir}
        
        logger.info("Interpolation took: {} minutes".format((time.time() - start_time) / 60))

        # Test save func
        self.save_dictionary(interp_dic)

        # Assertions
        assert len(self.interp_x) == len(self.interp_y), "Interpolations should be the same length"
    
    # Save the interpolated data to pickle rick dictionary
    def save_dictionary(self, dic_to_save):
        """A function that saves the interpolated data into a dictionary for loading in the future as the files are big to
        speed up coding
        """
        
        file = open(self.interp_path, "wb")
        pickle.dump(dic_to_save, file)
        file.close()

