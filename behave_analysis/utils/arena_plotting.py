import numpy as np
from typing import List, Optional
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Arena:
    def __init__(
        self, ax, dim=1024, shelter_coordinates=[[437, 886],[587, 976]], condition=None, barrier_coordinates: Optional[List[List[float]]] = None, full_image=True
    ):
        """Draws the arena with the shelter and barrier if present.

        Args:
            dim (int, optional): Defaults to 104 assuming a 1024x1024 pixel image. If your image is smaller (e.g. your data points are binned), you can pass the sim of your image and this will scale the arena plotting.
            ax (_type_, optional): plotting axes to draw the arena on. Necessary.
            shelter_coordinates (_type_, optional): In the format [[x1, y1], [x2, y2]] where the first sublist is the top left corner of the shelter and the second sublist is the bottom right corner of the shelter. Defaults to [[437, 886],[587, 976]] which are the coordinates for the shelter in our 1024x1024 pixel images.
            condition (_type_, optional): Experimental condition which determines whether shelter or barrier will be drawn on.
            barrier_coordinates (List[List], optional): The first sublist is one edge, the second the other. The third is the center point. Defaults to None.
                                                        If you pass barrier coordinate we assume you want them plotted!
            full_image (bool, optional): If False, coordinates are in 920-pixel arena space. If True, coordinates are in full 1024-pixel image space. Defaults to True.
        """
        radius = 460 if full_image else 512 # a funny trick that makes the radius the full image, assuming a 1024 size
        self.scaling = dim / 1024  # scale for
        self.arena_center = dim / 2
        self.ax = ax
        self.arena_radius = radius * self.scaling
        self.create_arena()
        self.ax.set_ylim([0, dim])
        self.ax.set_xlim([0, dim])
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()

        if not (condition == "pre_shelter"):
            if not full_image:
                shelter_coordinates = [[s + (512 - 460) for s in sh] for sh in shelter_coordinates]
            self.draw_shelter(shelter_coordinates)

        if barrier_coordinates is not None:
            if np.logical_or("barrier" in condition, condition == "all_time"):
                if not full_image:
                    barrier_coordinates = [[b + (512 - 460) for b in bc] for bc in barrier_coordinates]
                self.draw_barrier(condition, barrier_coordinates)

    def create_arena(self):
        """Given an axes, draw the edge of the arena"""
        circle = plt.Circle((self.arena_center, self.arena_center), self.arena_radius, color='k', fill=False, linewidth=2)
        self.ax.add_patch(circle)

    def draw_shelter(self, shelter_coordinates):
        """Draws the shelter on the arena plot"""
        top_left_coordinates = [b * self.scaling for b in shelter_coordinates[0]]
        bottom_right_coordinates = [b * self.scaling for b in shelter_coordinates[1]]
        width = bottom_right_coordinates[0] - top_left_coordinates[0]
        height = bottom_right_coordinates[1] - top_left_coordinates[1]
        
        square = patches.Rectangle((top_left_coordinates[0], top_left_coordinates[1]), width, height, color='r', alpha=0.4)
        self.ax.add_patch(square)

    def draw_barrier(self, condition, barrier_coordinates):
        """Draws a barrier on the arena plot, the orientation of which depends on the condition.
        We are assuming a barrier that varies in the x direction, so the two points should have the same y coordinate. 
        If this is not the case, the barrier will still be plotted but it may look weird!"""

        if "barrier_removed" in condition:
            return
        
        bar_loc = [] 

        if (len(barrier_coordinates) >= 2) & (isinstance(barrier_coordinates[0], list) or isinstance(barrier_coordinates[0], np.ndarray)):
            # barrier_coordinates needs to be a list of two points, each with x and y coordinates, e.g. [[x1, y1], [x2, y2]]
            if np.logical_or(np.logical_or("barrier_present" in condition, "all_time" in condition), "shelter_present" in condition):
                # draw old two-sided barrier
                bar_loc = [barrier_coordinates[0][0] * self.scaling, barrier_coordinates[1][0] * self.scaling] # assuming barrier varies in x direction
            elif "barrier_pre_flip" in condition:
                # take the first point
                barrier_coordinates = barrier_coordinates[0]
            elif "barrier_post_flip" in condition:
                # take the second point
                barrier_coordinates = barrier_coordinates[1]

        if len(bar_loc) == 0:
            # draw from the barrier point to the edge of the arena in cases where we only got one point
            if np.logical_or(np.logical_and("tiny" not in condition, (barrier_coordinates[0] * self.scaling) < self.arena_center),
                             np.logical_and("tiny" in condition, (barrier_coordinates[0] * self.scaling) > self.arena_center)):  # if 224 < 512
                bar_loc = [barrier_coordinates[0] * self.scaling, self.arena_center + self.arena_radius]  # [224 * 1, 512 + 460]
            elif np.logical_or(np.logical_and("tiny" not in condition, (barrier_coordinates[0] * self.scaling) > self.arena_center),
                               np.logical_and("tiny" in condition, (barrier_coordinates[0] * self.scaling) < self.arena_center)):  # if 800 > 512
                bar_loc = [self.arena_center - self.arena_radius, barrier_coordinates[0] * self.scaling]  # [512 - 460, 797 * 1]
        self.ax.plot(
                [bar_loc[0], bar_loc[1]], [self.arena_center, self.arena_center], color=[0, 0, 0]
            )