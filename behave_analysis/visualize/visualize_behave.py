#TODO update visualize behaviour to use new preprocess object as attributes will not be found now

# OS libaries
import numpy as np
from glob import glob
import polars as pl
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

class Visualize_behave():
    """
    A class for some sanity check behavior plots 
    to get a sense for what the mouse was doing in the session
    """
    def __init__(self, Visualize):
        self.Visualize = Visualize

    def position_by_bsa(self):
        """
        Make a scatter plot of position in arena colored by angle between body and shelter
        """
        # remove times when mouse is inside shelter
        outofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0],
            self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]),
            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1],
            self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1])))
        # color position by their shelter angle
        mass = self.Visualize.tracking_data['avg_loc'][outofShelterIdx,:]
        ang_color = np.digitize(np.rad2deg(self.Visualize.tracking_data['bod_shelt_dir'][outofShelterIdx]),np.arange(-180,180))
        phi = np.linspace(0, 2*np.pi, len(np.arange(360)))
        rgb_cycle = np.vstack((            # Three sinusoids
            .5*(1.+np.cos(phi          )), # scaled to [0,1]
            .5*(1.+np.cos(phi+2*np.pi/3)), # 120° phase shifted.
            .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (60,3)
        bsa_rgb_cycle = np.zeros(shape = (len(ang_color),3))
        for i in np.arange(360):
            bsa_rgb_cycle[ang_color == i+1,:] = rgb_cycle[i,:]
        plt.figure()
        plt.scatter(mass[:,0],mass[:,1],s=5,c=bsa_rgb_cycle,linewidths=0,marker='.')
        plt.title ('position coloured by angle to shelter')
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.savefig(os.path.join(self.Visualize.session.processed_path, "arena_position.png"))
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()
    
    def location_occupancy(self):
        """
        Make plots showing time in shelter, near barrier edged and in 4 quadrants of arena over the course of the session
        """
        # look in a 3 minute window
        w = 3
        # x axis values for plotting, in minutes
        x = np.arange(w/2,(self.Visualize.session.video.num_frames/(self.Visualize.session.video.fps*60))-(w/2)+1/(self.Visualize.session.video.fps*60),1/(self.Visualize.session.video.fps*60))
        figg, axs = plt.subplots(1,3)
        figg.set_figwidth(15)
        # time in shelter
        if len(self.Visualize.session.shelter_time) > 0:
            if 'mushroom' in self.Visualize.session.experiment:
                extra = 50 # in the mushroom session extend what the shelter is beyond the base
            else: extra = 0
            InShelterIdx = np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0]-extra,
                self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]+extra),
                np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1]-extra,
                self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1]+extra))
            axs[0].plot(x,np.convolve(InShelterIdx.astype(int), np.ones(self.Visualize.session.video.fps*60*w), 'valid') / (self.Visualize.session.video.fps*60*w))
            axs[0].plot([self.Visualize.sheltertime[0]/60,self.Visualize.sheltertime[0]/60],[0, 1],'-k')
            axs[0].title.set_text('In shelter')
            axs[0].set_xlabel('time (mins)')
            axs[0].set_ylabel('fraction occupancy')

        # time in 4 quadrants
        cc = matplotlib.cm.Set1
        center = [self.Visualize.session.video.width/2, self.Visualize.session.video.height/2]
        Q = np.vstack((np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] < center[0],self.Visualize.tracking_data['avg_loc'][:, 1] < center[1]), # upper_left
                            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > center[0],self.Visualize.tracking_data['avg_loc'][:, 1] < center[1]), # upper_right
                            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > center[0],self.Visualize.tracking_data['avg_loc'][:, 1] > center[1]), # lower_right
                            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] < center[0],self.Visualize.tracking_data['avg_loc'][:, 1] > center[1]))) # lower_left
        for i in np.arange(4):
            axs[1].plot(x, np.convolve(Q[i,:].astype(int), np.ones(self.Visualize.session.video.fps*60*w), 'valid') / (self.Visualize.session.video.fps*60*w), color = cc(i))
        axs[1].title.set_text('In quadrants')
        axs[1].legend(['upper_left','upper_right','lower_right','lower_left'])
        axs[1].set_xlabel('time (mins)')

        # time near barrier edge
        if len(self.Visualize.session.barrier_time) > 0:
            for i, c in enumerate(self.Visualize.tracking_data['barrier_loc']):
                extra = 35 # 
                NearBarrier = np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > c[0]-extra,
                    self.Visualize.tracking_data['avg_loc'][:, 0] < c[0]+extra),
                    np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > c[1]-extra,
                    self.Visualize.tracking_data['avg_loc'][:, 1] < c[1]+extra))
                axs[2].plot(x,np.convolve(NearBarrier.astype(int), np.ones(self.Visualize.session.video.fps*60*w), 'valid') / (self.Visualize.session.video.fps*60*w), color = cc(i))
            axs[2].plot([self.Visualize.barriertime[0]/60,self.Visualize.barriertime[0]/60],[0, 1],'-k')    
        axs[2].set_xlabel('time (mins)')
        axs[2].legend(['left_edge','right_edge'])
        axs[2].title.set_text('Near barrier edge')

        plt.savefig(os.path.join(self.Visualize.session.processed_path, "arena_occupancy_vs_time.png"))
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()

    def angle_histograms(self):
        """
        Make histograms of head direction, head shelter angle and barrier shelter angle to ensure good sampling
        """
        figg, axs = plt.subplots(1,3)
        figg.set_figwidth(15)
        
        # time in shelter (we're excluding this from our histograms)
        if 'mushroom' in self.Visualize.session.experiment:
            extra = 50 # in the mushroom session extend what the shelter is beyond the base
        else: extra = 0
        OutofShelterIdx = np.logical_not(np.logical_and(np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 0] > self.Visualize.tracking_data['shelter_loc'][0][0]-extra,
            self.Visualize.tracking_data['avg_loc'][:, 0] < self.Visualize.tracking_data['shelter_loc'][1][0]+extra),
            np.logical_and(self.Visualize.tracking_data['avg_loc'][:, 1] > self.Visualize.tracking_data['shelter_loc'][0][1]-extra,
            self.Visualize.tracking_data['avg_loc'][:, 1] < self.Visualize.tracking_data['shelter_loc'][1][1]+extra)))
        # head direction
        axs[0].hist(self.Visualize.tracking_data['hdir'][OutofShelterIdx],np.arange(-np.pi,np.pi,np.pi/10),density = 'stacked')
        axs[0].set_ylabel('fraction of frames')
        axs[0].title.set_text('head dir')

        # head shelter angle
        if len(self.Visualize.session.shelter_time) > 0:
            # only for times when there is a shelter-only
            frames_with_shelter = np.zeros_like(self.Visualize.tracking_data['hdir_shelt'])
            if self.Visualize.sheltertime[1] == -60: frames_with_shelter[self.Visualize.sheltertime[0]*self.Visualize.session.video.fps:] = 1
            else: frames_with_shelter[self.Visualize.sheltertime[0]*self.Visualize.session.video.fps:self.Visualize.sheltertime[1]*self.Visualize.session.video.fps] = 1
            axs[1].hist(self.Visualize.tracking_data['hdir_shelt'][np.logical_and(OutofShelterIdx,frames_with_shelter == 1)],np.arange(-np.pi,np.pi,np.pi/10),density = 'stacked')
            axs[1].title.set_text('head shelter angle')

        # head barrier angle
        if len(self.Visualize.session.barrier_time) > 0:
            # only for times when there is a barrier
            frames_with_barrier = np.zeros_like(self.Visualize.tracking_data['hdir_shelt'])
            if self.Visualize.barriertime[1] == -60: frames_with_barrier[self.Visualize.barriertime[0]*self.Visualize.session.video.fps:] = 1
            else: frames_with_barrier[self.Visualize.barriertime[0]*self.Visualize.session.video.fps:self.Visualize.barriertime[1]*self.Visualize.session.video.fps] = 1
            for c in np.arange(2):
                axs[2].hist(self.Visualize.tracking_data['hdir_barrier'][np.logical_and(OutofShelterIdx,frames_with_shelter == 1),c],np.arange(-np.pi,np.pi,np.pi/10),density = 'stacked')
            axs[2].title.set_text('head barrier-edge angle')
        
        plt.savefig(os.path.join(self.Visualize.session.processed_path, "distribution_head_angles.png"))
        if self.Visualize.settings.show_plots: plt.show()
        plt.close()
        
class GraphingBase:
    """
    A base parent class for all graphing functions
    """
    def __init__(self, MaxPlotsPerFigure, how_many_plots_you_need):
        self.num_cols = int(np.ceil(np.sqrt(MaxPlotsPerFigure)))
        self.num_rows = int(np.ceil(MaxPlotsPerFigure / self.num_cols))
        self.num_figures = int(np.ceil(how_many_plots_you_need / MaxPlotsPerFigure))
        self.how_many_plots_you_need = how_many_plots_you_need
        
        print(f"The number of figures created will be: {self.num_figures}")
                
    def create_figure_with_subplots(self):
        fig, axs = plt.subplots(self.num_rows, self.num_cols)
        fig.set_figwidth(15)
        fig.set_figheight(8)
        return fig, axs
    
    def save_plot(self, directoryToSaveTo, plotName):
        plt.savefig(directoryToSaveTo + "/" + plotName + ".png", dpi = 300)
         
class Correlations(GraphingBase):
    """
    A class for plotting correlations between different angle variables
    """
    def __init__(self, MaxPlotsPerFigure, how_many_plots_you_need, CleanVideoDf, directoryToSaveTo, plotName):
        super().__init__(MaxPlotsPerFigure, how_many_plots_you_need)
        self.CleanVideoDf = CleanVideoDf
        self.variable_permutations = self.extract_correlation_permutations()
        self.xsys = self.extract_xsys()
        self.directoryToSaveTo = directoryToSaveTo
        self.plotName = plotName
                
    def extract_correlation_permutations(self):
        """
        A function to extract all possible permutations of the correlation variables for plotting. There should be six permutations:
        1. head direction vs head shelter angle, south, north
        2. head shelter angl vs head barrier angle south, north
        3. north barrier vs south barrier
        """
        variable_permutation_dictionary = {"head direction VS head shelter angle": ("hdir", "hsa"),
                      "head direction VS north barrier edge angle": ("hdir", "h_bar_north_a"),
                      "head direction VS south barrier edge angle": ("hdir", "h_bar_south_a"),
                      "head shelter angle VS north barrier edge angle": ("hsa", "h_bar_north_a"),
                      "head shelter angle VS south barrier edge angle": ("hsa", "h_bar_south_a"),
                      "north barrier edge angle VS south barrier edge angle": ("h_bar_north_a", "h_bar_south_a")}
        
        permutation_tuples = {}
        for key, value in variable_permutation_dictionary.items():
            permutation_tuples[key] = (value[0], value[1])
                
        return permutation_tuples
        
    def extract_xsys(self):
        dict = {}
        for key, value in self.variable_permutations.items():
            dict[key] = self.CleanVideoDf[[value[0], value[1]]]
        return dict
                
    def create_correration_plot(self):
        total_plots = 0
        
        for figure in range(self.num_figures):
            fig, axs = self.create_figure_with_subplots()
            x = np.linspace(0, 2 * np.pi, 400)
            y = np.linspace(0, 3 * np.pi, 400)
            
            for ax in axs.flat:
                if total_plots < self.how_many_plots_you_need:
                    x = list(self.xsys.values())[total_plots][:, 0]
                    y = list(self.xsys.values())[total_plots][:, 1]
                                
                    ax.scatter(x, y, s = 0.2)
                    ax.set_title(list(self.variable_permutations.keys())[total_plots], fontsize = 8)
                    ax.set_xlabel(list(self.variable_permutations.values())[total_plots][0])
                    ax.set_ylabel(list(self.variable_permutations.values())[total_plots][1])
                    
                    total_plots += 1
                else:
                    ax.axis('off')  # hide axis if not used
            
            fig.subplots_adjust(hspace=1)
            
        self.save_plot(self.directoryToSaveTo, self.plotName)
        plt.show()



