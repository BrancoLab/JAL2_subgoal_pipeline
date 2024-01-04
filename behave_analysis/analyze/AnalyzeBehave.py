# Custom classes
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.utils.color_funcs import get_color_based_on_speed
from settings.settings_analyze import settings_analyze as settings

# OS Lib
from loguru import logger
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt

class AnalyzeBehave:
    """
    A class that analyzes mouse behavior in a session
    """
    def __init__(self,session):
        logger.info('Initializing AnalyzeBehave')
        self.dir = os.path.join(session.base_path,session.processed_path) + "\\" + 'analyze_behave' 
        self.session = session
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self.show_plots = settings.show_plots
        self.settings = settings
        open_tracking_data(self)
        """Load in video df"""
        video_df = os.path.join(session.base_path,session.processed_path) + "\\" + "full_video_dataframe.csv"
        if os.path.isfile(video_df):
            self.video_df = pl.read_csv(video_df)
        else:
            raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
        self.spatial_efficiency()
 
    def spatial_efficiency(self):
        """ 
        Plot escape trajectories as well as optimal path
        """
        
        # set up figure and number of rows and calculate number of columns
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        ntrial = len(self.session.__dict__[self.settings.stim_type].onset_frames)
        nrows = 3
        ncols = ntrial // nrows + (ntrial % nrows > 0)
        
        # plot trajectory
        self.trajectory_length = np.empty(len(self.session.__dict__[self.settings.stim_type].onset_frames))
        self.optimal_trajectory_length = np.empty(len(self.session.__dict__[self.settings.stim_type].onset_frames))
        self.spatial_efficiency_value = np.empty(len(self.session.__dict__[self.settings.stim_type].onset_frames))
        for trial_num, (onset_frames, stimulus_durations) in enumerate(
            zip(self.session.__dict__[self.settings.stim_type].onset_frames, 
                self.session.__dict__[self.settings.stim_type].stimulus_durations)):
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            # set up axes with shelt and barrier locations
            condition = identify_condition_escape(self.video_df.filter(self.video_df['frames'] == onset_frames),self.session)
            base_plotting(ax,self.tracking_data,condition)
            self.trajectory_length[trial_num] = self.plot_escape_trajectories(onset_frames[0],stimulus_durations[0],ax)
            self.optimal_trajectory_length[trial_num] = self.plot_optimal_trajectories(onset_frames[0],ax,condition)
            self.spatial_efficiency_value[trial_num] = self.optimal_trajectory_length[trial_num]/self.trajectory_length[trial_num]
            ax.set_title('spatial efficiency = ' + str(self.spatial_efficiency_value[trial_num]))
        
        # save figure
        filename = str(self.dir) + "/" + "SpatialEfficiency" + ".png"
        plt.savefig(filename)
        if self.show_plots: plt.show()
        plt.close()

    def plot_escape_trajectories(self, onset_frames,stimulus_durations,ax):
        """ 
        Plot escape trajectories
        """
        # compute and plot each trajectory
        x_loc = self.tracking_data['head_loc'][onset_frames:onset_frames + int(stimulus_durations*40),0]
        y_loc = self.tracking_data['head_loc'][onset_frames:onset_frames + int(stimulus_durations*40),1]
        speed = self.tracking_data["avg_Velocity"][onset_frames:onset_frames + int(stimulus_durations*40)]
        trail_color = np.empty((len(speed),3))
        distance_travelled = []
        for i,stim_status in enumerate(np.arange(0,stimulus_durations,0.025)):
            trail_color[i,:] = get_color_based_on_speed(speed=speed[i], 
                                                object_to_color="trail", 
                                                stim_status=stim_status, 
                                                stim_type=self.settings.stim_type)
            if i > 0:
                distance_travelled = np.append(distance_travelled,
                                               np.sqrt((x_loc[i] - x_loc[i-1])**2 + (y_loc[i] - y_loc[i-1])**2))
        ax.scatter(x_loc,y_loc,s=5,c=trail_color/255)
        return np.sum(distance_travelled)

    def plot_optimal_trajectories(self, onset_frames, ax, condition):
        """ Plot optimal escape path"""
        x_loc = self.tracking_data['head_loc'][onset_frames,0]
        y_loc = self.tracking_data['head_loc'][onset_frames,1]
        c_line = [1,0,0]
        # compute and plot each optimal trajectory to barrier
        trjectory_to_barrier = 0
        if not(condition == 'shelter_only'):
            if condition == 'barrier_present': # double sided barrier 
                nearest_barrier_edge = np.argmin([np.sqrt((x_loc - self.tracking_data["barrier_loc"][0][0])**2 +
                                                        (y_loc - self.tracking_data["barrier_loc"][0][1])**2),
                                                np.sqrt((x_loc - self.tracking_data["barrier_loc"][1][0])**2 +
                                                        (y_loc - self.tracking_data["barrier_loc"][1][1])**2)])
            elif condition == 'barrier_pre_flip':
                nearest_barrier_edge = 0
            elif condition == 'barrier_post_flip':
                nearest_barrier_edge = 1
            ax.plot([x_loc,self.tracking_data["barrier_loc"][nearest_barrier_edge][0]],
                        [y_loc,self.tracking_data["barrier_loc"][nearest_barrier_edge][1]],
                        color = c_line)
            trjectory_to_barrier = np.sqrt((x_loc - self.tracking_data["barrier_loc"][nearest_barrier_edge][0])**2
                                    + (y_loc - self.tracking_data["barrier_loc"][nearest_barrier_edge][1])**2)
            x_loc = self.tracking_data["barrier_loc"][nearest_barrier_edge][0]
            y_loc = self.tracking_data["barrier_loc"][nearest_barrier_edge][1]
            
        # compute and plot each optimal trajectory to shelter
        ax.plot([x_loc,np.mean([self.tracking_data['shelter_loc'][0][0],self.tracking_data['shelter_loc'][1][0]])],
                [y_loc,np.mean([self.tracking_data['shelter_loc'][0][1],self.tracking_data['shelter_loc'][1][1]])],
                color = c_line)
        trajectory_to_shelter = np.sqrt((np.mean([self.tracking_data['shelter_loc'][0][0],self.tracking_data['shelter_loc'][1][0]]) - x_loc)**2 
                                        + (np.mean([self.tracking_data['shelter_loc'][0][1],self.tracking_data['shelter_loc'][1][1]]) - y_loc)**2)
        return np.sum([trjectory_to_barrier,trajectory_to_shelter])

###--------------UTILS

def identify_condition_escape(video_df,session):
    """Which condition did the escape happen in?"""
    if np.logical_and(video_df['shelter'].to_numpy() == True, video_df['barrier_present'].to_numpy() == False):
        condition = 'shelter_only'
    elif np.logical_and(video_df['shelter'].to_numpy() == True, video_df['barrier_present'].to_numpy() == True):
        if session.barrier_flip_time:
            if video_df['barrier_flipped'].to_numpy() == False:
                condition = 'barrier_pre_flip'
            else: condition = 'barrier_post_flip'
        else: condition = 'barrier_present'
    return condition

def base_plotting(ax,tracking,condition):
    arena_radius = 460
    # draw shelter
    if 'shelter_loc' in tracking.keys():
        for i in [0,1]:
            plt.plot([tracking["shelter_loc"][0][0],tracking["shelter_loc"][1][0]],
                    [tracking["shelter_loc"][i][1],tracking["shelter_loc"][i][1]],
                    color = [1,0,0])
            plt.plot([tracking["shelter_loc"][i][0],tracking["shelter_loc"][i][0]],
                    [tracking["shelter_loc"][0][1],tracking["shelter_loc"][1][1]],
                    color = [0,0,0])
    
    if not(condition == 'shelter_only'):
        if condition == 'barrier_present':
            # draw old two-sided barrier
            bar_loc = [tracking["barrier_loc"][0][0],tracking["barrier_loc"][1][0]]
        
        if condition == 'barrier_pre_flip':
            # draw barrier from first point to the edge
            if tracking["barrier_loc"][0][0] < 512: bar_loc = [tracking["barrier_loc"][0][0],512+arena_radius]
            else: bar_loc = [512-arena_radius,tracking["barrier_loc"][0][0]]
        
        if condition == 'barrier_post_flip':
            # draw barrier from second point to the edge
            if tracking["barrier_loc"][1][0] < 512: bar_loc = [tracking["barrier_loc"][1][0],512+arena_radius]
            else: bar_loc = [512-arena_radius,tracking["barrier_loc"][1][0]]
        
        plt.plot([bar_loc[0],bar_loc[1]],
                [tracking["barrier_loc"][0][1],tracking["barrier_loc"][1][1]],
                color = [0,0,0])
    
    # draw arena edge
    a = 512 + ( arena_radius * np.cos( np.linspace( 0 , 2 * np.pi , 150 ) ) )
    b = 512 + ( arena_radius * np.sin( np.linspace( 0 , 2 * np.pi , 150 ) ) )

    ax.plot( a, b , color = [0,0,0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')