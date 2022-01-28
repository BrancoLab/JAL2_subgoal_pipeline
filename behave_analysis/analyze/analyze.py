from behave_analysis.process.camera_trigger import get_num_frames_expected, get_Camera_trigger
from behave_analysis.process.process import Process
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.analyze.plot_funcs import *
from behave_analysis.analyze.data_extraction_funcs import *
from behave_analysis.analyze.stats_funcs import permutation_test, print_stat_test_results
from behave_analysis.analyze.trial_eligibility_funcs import trial_is_eligible
from behave_analysis.utils.directory import Directory
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np

class Analyze():
    def __init__(self, session_IDs, settings, analysis_type):
        self.settings = settings
        self.session_IDs = session_IDs
        self.session_count = -1
        self.trial_num = 0
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
        for trial in self.trials_to_plot:
            self.initialize_trajectory_plot()
            self.plot_trajectory(trial)
            self.save_plot()

# ----DATA EXTRACTION FUNCS---------------------------------------------
    def extract_data(self):
        for session_ID in self.session_IDs:
            self.trial_num = 0
            self.minutes_into_session = None
            self.open_session_data(session_ID) 
            if not 'explor' in self.analysis_type: self.get_data_on_each_trial()
            self.get_start_and_end_frames()

    def open_session_data(self, session_ID):
        self.session = Process(session_ID).load_session()
        self.group_num = session_ID[5]
        self.fps = self.session.video.fps
        self.session_count += 1
        self.num_successful_escapes_this_session = 0
        open_tracking_data(self) # generates self.tracking_data

    def get_data_on_each_trial(self):
        for onset_frames, stim_durations in zip(self.session.__dict__[self.stim_type].onset_frames, \
                                                self.session.__dict__[self.stim_type].stimulus_durations):
            if not trial_is_eligible(self, onset_frames): 
                continue
            self.generate_trial_dict(onset_frames, stim_durations)
            self.trial_num+=1

    def generate_trial_dict(self, onset_frames: list, stim_durations: list):
        if self.stim_type in ['audio', 'homing', 'threshold_crossing']: epochs = ['stimulus']
        if self.stim_type=='laser':                                     epochs = ['stimulus', 'post-laser']
        for epoch in epochs:
            trial_start_idx, trial_end_idx = get_trial_start_and_end(self, onset_frames, stim_durations, epoch)
            trial                          = create_trial_dict(self, trial_start_idx, trial_end_idx, epoch)
            self.trials_to_plot.append(trial)

    def get_start_and_end_frames(self):
        session_start = 0
        camera_trigger_data = get_Camera_trigger(self.session)[1]
        session_end = get_num_frames_expected(self.session, camera_trigger_data)[0]
        epoch = 'exploration'
        trial = create_trial_dict(self, session_start, session_end, epoch)
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
        size = self.session.video.registration_size
        self.fig, self.ax = plt.subplots(figsize=(9,9))
        self.ax.set_xlim([0, size[0]])
        self.ax.set_ylim([0, size[1]])
        if self.stim_type in ['laser', 'homing', 'threshold_crossing']: 
            self.ax.plot([size[0]/2-250, size[1]/2+250], [size[0]/2, size[1]/2], color=[0, 0, 0], linewidth=5) #obstacle
        circle = plt.Circle((size[0]/2, size[1]/2), radius=460, color=[0, 0, 0], linewidth=1, fill=False)
        self.ax.add_artist(circle)
        self.ax.invert_yaxis()
        format_axis(self) 

    def plot_single_trial(self, trial):
        self.plot_trajectory(trial)
        self.plot_silhouettes(trial)
        self.trial_num            = trial['trial count']
        self.minutes_into_session = np.round(trial['trial start'] / self.session.video.fps / 60) 

    def plot_trajectory(self, trial):
        if self.color_by in ['speed', 'speed+RT','time']: gradient_line(self, trial)
        else:                                             solid_line(self, trial)   
        self.mouse = trial['mouse']
        self.session.experiment = trial['experiment']   

    def plot_silhouettes(self, trial, mouse_size: float=38, color: tuple=(.7,.7,.7), num_silhouettes=6):

        colors = generate_list_of_colors(self.color_by, self.stim_type, trial['epoch'], trial['speed'], RT=trial['escape initiation idx'], object_to_color='trial')
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
