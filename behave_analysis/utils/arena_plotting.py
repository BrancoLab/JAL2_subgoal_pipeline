import numpy as np
from typing import List, Optional


class Arena:
    def __init__(
        self, dim=1024, ax=None, shelter_coordinates=[[437, 886],[587, 976]], condition=None, barrier_coordinates: Optional[List[List[float]]] = None, no_offset=False
    ):
        """Draws the arena with the shelter and barrier if present.

        Args:
            dim (int, optional): _description_. Defaults to 1024.
            ax (_type_, optional): _description_. Defaults to None.
            shelter_coordinates (_type_, optional): _description_. Defaults to None.
            condition (_type_, optional): _description_. Defaults to None.
            barrier_coordinates (List[List], optional): The first sublist is one edge, the second the other. The third is the center point. Defaults to None.
            no_offset (bool, optional): _description_. Defaults to False.
        """
        radius = 460 if not no_offset else 512
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
            if no_offset:
                shelter_coordinates = [[s + (512 - 460) for s in sh] for sh in shelter_coordinates]
            self.draw_shelter(shelter_coordinates)

        if barrier_coordinates:
            if no_offset:
                barrier_coordinates = [[b + (512 - 460) for b in bc] for bc in barrier_coordinates]
            self.draw_barrier(condition, barrier_coordinates)

    def create_arena(self):
        """Given an axes, draw the edge of the arena"""
        a = self.arena_center + (self.arena_radius * np.cos(np.linspace(0, 2 * np.pi, 150)))
        b = self.arena_center + (self.arena_radius * np.sin(np.linspace(0, 2 * np.pi, 150)))
        self.ax.plot(a, b, color=[0, 0, 0], linewidth=2)

    def draw_shelter(self, shelter_coordinates):
        """Draws the shelter on the arena plot"""
        top_left_coordinates = [b * self.scaling for b in shelter_coordinates[0]]
        bottom_right_coordinates = [b * self.scaling for b in shelter_coordinates[1]]
        self.ax.plot([top_left_coordinates[0], bottom_right_coordinates[0]], [top_left_coordinates[1], top_left_coordinates[1]], color=[0, 0, 0])
        self.ax.plot([top_left_coordinates[0], top_left_coordinates[0]], [top_left_coordinates[1], bottom_right_coordinates[1]], color=[0, 0, 0])
        self.ax.plot(
            [top_left_coordinates[0], bottom_right_coordinates[0]], [bottom_right_coordinates[1], bottom_right_coordinates[1]], color=[0, 0, 0]
        )
        self.ax.plot(
            [bottom_right_coordinates[0], bottom_right_coordinates[0]], [top_left_coordinates[1], bottom_right_coordinates[1]], color=[0, 0, 0]
        )

    def draw_barrier(self, condition, barrier_coordinates):
        """Draws a barrier on the arena plot, the orientation of which depends on the condition"""
        if not np.logical_or(condition == "shelter_only", condition == "pre_shelter"):

            if np.logical_or(np.logical_or(condition == "barrier_present", condition == "all_time"), condition == "shelter_present"):
                # draw old two-sided barrier
                bar_loc = [barrier_coordinates[0][0] * self.scaling, barrier_coordinates[1][0] * self.scaling]
                if np.logical_and(
                    np.diff(bar_loc) < self.arena_radius, bar_loc[1] > self.arena_center
                ):  # if we didn't flip the barrier and only have one side the "two sides will appear as one"
                    bar_loc[0] = self.arena_center - self.arena_radius
                elif np.logical_and(np.diff(bar_loc) < self.arena_radius, bar_loc[1] < self.arena_center):
                    bar_loc[0] = self.arena_center + self.arena_radius

            if condition == "barrier_pre_flip":
                # draw barrier from first point to the edge
                if barrier_coordinates[0][0] < self.arena_center:  # if 224 < 512
                    bar_loc = [barrier_coordinates[0][0] * self.scaling, self.arena_center + self.arena_radius]  # [224 * 1, 512 + 460]
                else:
                    bar_loc = [self.arena_center - self.arena_radius, barrier_coordinates[0][0] * self.scaling]

            if condition == "barrier_post_flip":
                # draw barrier from second point to the edge
                if barrier_coordinates[1][0] < self.arena_center:  # if 797 < 512
                    bar_loc = [barrier_coordinates[1][0] * self.scaling, self.arena_center + self.arena_radius]
                else:
                    bar_loc = [self.arena_center - self.arena_radius, barrier_coordinates[1][0] * self.scaling]  # [512 + 460, 797 * 1]

            if not condition == "barrier_removed":
                # draw barrier location onto the arena base
                self.ax.plot(
                    [bar_loc[0], bar_loc[1]], [barrier_coordinates[0][1] * self.scaling, barrier_coordinates[1][1] * self.scaling], color=[0, 0, 0]
                )
