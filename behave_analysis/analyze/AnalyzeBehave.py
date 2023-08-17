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
        self.dir = session.processed_path + "\\" + 'analyze_behave' 
        self.session = session
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self.show_plots = settings.show_plots
        self.settings = settings
        open_tracking_data(self)
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
            self.trajectory_length[trial_num] = self.plot_escape_trajectories(onset_frames[0],stimulus_durations[0],ax)
            self.optimal_trajectory_length[trial_num] = self.plot_optimal_trajectories(onset_frames[0],ax)
            self.spatial_efficiency_value[trial_num] = self.optimal_trajectory_length[trial_num]/self.trajectory_length[trial_num]
            ax.title('spatial efficiency = ' + str(self.spatial_efficiency_value[trial_num]))
        
        # save figure
        filename = str(self.dir) + "/" + "SpatialEfficiency" + ".png"
        plt.savefig(filename)
        if self.show_plots: plt.show()
        plt.close()

    def plot_escape_trajectories(self, onset_frames,stimulus_durations,ax):
        """ 
        Plot escape trajectories
        """
        # set up axes with shelt and barrier locations
        base_plotting(ax,self.tracking_data)
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

    def plot_optimal_trajectories(self, onset_frames, ax):
        """ Plot optimal escape path"""
        x_loc = self.tracking_data['head_loc'][onset_frames,0]
        y_loc = self.tracking_data['head_loc'][onset_frames,1]
        c_line = [1,0,0]
        # compute and plot each optimal trajectory to barrier
        trjectory_to_barrier = []
        if len(self.tracking_data['barrier_loc']) > 0:
            nearest_barrier_edge = np.argmin([np.sqrt((x_loc - self.tracking_data["barrier_loc"][0][0])**2 +
                                                    (y_loc - self.tracking_data["barrier_loc"][0][1])**2),
                                            np.sqrt((x_loc - self.tracking_data["barrier_loc"][1][0])**2 +
                                                    (y_loc - self.tracking_data["barrier_loc"][1][1])**2)])
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

def base_plotting(ax,tracking):
    # draw barrier
    if len(tracking['barrier_loc']) > 0: # 'barrier_loc' in tracking.keys():
        plt.plot([tracking["barrier_loc"][0][0],tracking["barrier_loc"][1][0]],
                [tracking["barrier_loc"][0][1],tracking["barrier_loc"][1][1]],
                color = [0,0,0])
    # draw shelter
    if 'shelter_loc' in tracking.keys():
        for i in [0,1]:
            plt.plot([tracking["shelter_loc"][0][0],tracking["shelter_loc"][1][0]],
                    [tracking["shelter_loc"][i][1],tracking["shelter_loc"][i][1]],
                    color = [1,0,0])
            plt.plot([tracking["shelter_loc"][i][0],tracking["shelter_loc"][i][0]],
                    [tracking["shelter_loc"][0][1],tracking["shelter_loc"][1][1]],
                    color = [0,0,0])
    
    # draw arena edge
    a = 512 + ( 460 * np.cos( np.linspace( 0 , 2 * np.pi , 150 ) ) )
    b = 512 + ( 460 * np.sin( np.linspace( 0 , 2 * np.pi , 150 ) ) )

    ax.plot( a, b , color = [0,0,0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')