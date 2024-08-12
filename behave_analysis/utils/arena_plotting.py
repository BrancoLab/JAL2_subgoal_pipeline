import numpy as np


class Arena:
    """Arena parent class for plotting the edge of the arena. This class is inherited by the other arena classes."""

    def __init__(self, arena_radius=460, arena_center=512, ax=None, shelter_coordinates=None, condition=None, barrier_coordinates=None):
        self.arena_radius = arena_radius
        self.arena_center = arena_center
        self.ax = ax
        self.create_arena()
        if shelter_coordinates is not None:
            self.draw_shelter(shelter_coordinates)
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()
        if condition is not None:
            self.draw_barrier(condition, barrier_coordinates)

    def create_arena(self):
        """Given an axes, draw the edge of the arena"""
        a = self.arena_center + (self.arena_radius * np.cos(np.linspace(0, 2 * np.pi, 150)))
        b = self.arena_center + (self.arena_radius * np.sin(np.linspace(0, 2 * np.pi, 150)))
        self.ax.plot(a, b, color=[0, 0, 0], linewidth=2)

    def draw_shelter(self, shelter_coordinates):
        """Draws the shelter on the arena plot"""
        top_left_coordinates = shelter_coordinates[0]
        bottom_right_coordinates = shelter_coordinates[1]
        self.ax.plot([top_left_coordinates[0], bottom_right_coordinates[0]], [top_left_coordinates[1], top_left_coordinates[1]], color=[1, 0, 0])
        self.ax.plot([top_left_coordinates[0], top_left_coordinates[0]], [top_left_coordinates[1], bottom_right_coordinates[1]], color=[1, 0, 0])
        self.ax.plot(
            [top_left_coordinates[0], bottom_right_coordinates[0]], [bottom_right_coordinates[1], bottom_right_coordinates[1]], color=[1, 0, 0]
        )
        self.ax.plot(
            [bottom_right_coordinates[0], bottom_right_coordinates[0]], [top_left_coordinates[1], bottom_right_coordinates[1]], color=[1, 0, 0]
        )

    def draw_barrier(self, condition, barrier_coordinates):
        if condition == "barrier_pre_flip":
            # draw barrier from first point to the edge
            if barrier_coordinates[0][0] < 512:
                bar_loc = [barrier_coordinates[0][0], 512 + self.arena_radius]
            else:
                bar_loc = [512 - self.arena_radius, barrier_coordinates[0][0]]

        if condition == "barrier_post_flip":
            # draw barrier from second point to the edge
            if barrier_coordinates[1][0] < 512:
                bar_loc = [barrier_coordinates[1][0], 512 + self.arena_radius]
            else:
                bar_loc = [512 - self.arena_radius, barrier_coordinates[1][0]]
                
        if not condition == "barrier_removed":
            if not condition == "all_time":
                if not condition == "shelter_only":
                    # draw barrier location onto the arena base
                    self.ax.plot([bar_loc[0], bar_loc[1]], [barrier_coordinates[0][1], barrier_coordinates[1][1]], color=[0, 0, 0])
    
