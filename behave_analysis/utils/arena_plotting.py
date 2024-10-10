import numpy as np


class Arena:
    """Arena parent class for plotting the edge of the arena. This class is inherited by the other arena classes."""

    def __init__(self, dim = 1024, ax=None, shelter_coordinates=None, condition=None, barrier_coordinates=None, no_offset = False):
        self.scaling = dim/1024 # scale for 
        radius = 460
        if no_offset:
            radius = 512
        self.arena_center = dim/2
        self.arena_radius = radius*self.scaling
        self.ax = ax
        self.create_arena()
        if not(condition == "pre_shelter"):
            if no_offset:
                shelter_coordinates = [[s+(512-460) for s in sh] for sh in shelter_coordinates]
            self.draw_shelter(shelter_coordinates)
        self.ax.set_ylim([0, dim])
        self.ax.set_xlim([0, dim])
        if not barrier_coordinates == None:
            if no_offset:
                barrier_coordinates = [[b+(512-460) for b in bc] for bc in barrier_coordinates]
            self.draw_barrier(condition, barrier_coordinates)
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()

    def create_arena(self):
        """Given an axes, draw the edge of the arena"""
        a = self.arena_center + (self.arena_radius * np.cos(np.linspace(0, 2 * np.pi, 150)))
        b = self.arena_center + (self.arena_radius * np.sin(np.linspace(0, 2 * np.pi, 150)))
        self.ax.plot(a, b, color=[0, 0, 0], linewidth=2)

    def draw_shelter(self, shelter_coordinates):
        """Draws the shelter on the arena plot"""
        top_left_coordinates = [b*self.scaling for b in shelter_coordinates[0]]
        bottom_right_coordinates = [b*self.scaling for b in shelter_coordinates[1]]
        self.ax.plot([top_left_coordinates[0], bottom_right_coordinates[0]], [top_left_coordinates[1], top_left_coordinates[1]], color=[0, 0, 0])
        self.ax.plot([top_left_coordinates[0], top_left_coordinates[0]], [top_left_coordinates[1], bottom_right_coordinates[1]], color=[0, 0, 0])
        self.ax.plot(
            [top_left_coordinates[0], bottom_right_coordinates[0]], [bottom_right_coordinates[1], bottom_right_coordinates[1]], color=[0, 0, 0]
        )
        self.ax.plot(
            [bottom_right_coordinates[0], bottom_right_coordinates[0]], [top_left_coordinates[1], bottom_right_coordinates[1]], color=[0, 0, 0]
        )

    def draw_barrier(self, condition, barrier_coordinates):
        if not np.logical_or(condition == "shelter_only", condition == "pre_shelter"):
            if np.logical_or(np.logical_or(condition == 'barrier_present',condition == 'all_time'),condition == 'shelter_present'):
                # draw old two-sided barrier
                bar_loc = [barrier_coordinates[0][0]*self.scaling, barrier_coordinates[1][0]*self.scaling]
                if np.logical_and(np.diff(bar_loc) < self.arena_radius,bar_loc[1]>self.arena_center): # if we didn't flip the barrier and only have one side the "two sides will appear as one"
                    bar_loc[0] = self.arena_center - self.arena_radius
                elif np.logical_and(np.diff(bar_loc) < self.arena_radius,bar_loc[1]<self.arena_center):
                    bar_loc[0] = self.arena_center + self.arena_radius
            
            if condition == "barrier_pre_flip":
                # draw barrier from first point to the edge
                if barrier_coordinates[0][0] < self.arena_center:
                    bar_loc = [barrier_coordinates[0][0]*self.scaling, self.arena_center + self.arena_radius]
                else:
                    bar_loc = [self.arena_center - self.arena_radius, barrier_coordinates[0][0]*self.scaling]

            if condition == "barrier_post_flip":
                # draw barrier from second point to the edge
                if barrier_coordinates[1][0] < self.arena_center:
                    bar_loc = [barrier_coordinates[1][0]*self.scaling, self.arena_center + self.arena_radius]
                else:
                    bar_loc = [self.arena_center + self.arena_radius, barrier_coordinates[1][0]*self.scaling]
                    
            if not condition == "barrier_removed":
                # draw barrier location onto the arena base
                self.ax.plot([bar_loc[0], bar_loc[1]], [barrier_coordinates[0][1]*self.scaling, barrier_coordinates[1][1]*self.scaling], color=[0, 0, 0])
    
