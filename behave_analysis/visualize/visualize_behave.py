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
        plt.scatter(mass[:,0],mass[:,1],s=5,c=bsa_rgb_cycle,linewidths=0,marker='.')
        plt.title ('position coloured by angle to shelter')
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + "arena_position.png")
        if self.Visualize.settings.show_plots: plt.show()
    
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
        if 'Seq' in self.Visualize.session.experiment:
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
        plt.savefig(str(self.Visualize.session.file_path) + "/" + "arena_occupancy_vs_time.png")
        if self.Visualize.settings.show_plots: plt.show()

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
        axs[1].hist(self.Visualize.tracking_data['hdir_shelt'][OutofShelterIdx],np.arange(-np.pi,np.pi,np.pi/10),density = 'stacked')
        axs[1].title.set_text('head shelter angle')
        # head barrier angle
        for c in np.arange(2):
            axs[2].hist(self.Visualize.tracking_data['hdir_barrier'][OutofShelterIdx,c],np.arange(-np.pi,np.pi,np.pi/10),density = 'stacked')
        axs[2].title.set_text('head barrier-edge angle')
        plt.savefig(str(self.Visualize.session.file_path) + "/" + "distribution_head_angles.png")
        if self.Visualize.settings.show_plots: plt.show()