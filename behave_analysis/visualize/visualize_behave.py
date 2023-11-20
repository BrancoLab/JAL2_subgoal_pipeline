"""A module that organizes the visualization of the behavior data"""

import os

import seaborn as sns
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, filter_video_dataframe
from behave_analysis.visualize.behaviour.circular_coeff_of_angles import plot_the_circular_rho
from behave_analysis.visualize.behaviour.behaviour_coverage_metrics import CoverageStatistics
from behave_analysis.visualize.behaviour.heat_plot import plot_heat_map_of_position
from behave_analysis.visualize.behaviour.angle_distributions import plot_angle_distributions
from settings.settings_visualize import defined_settings_visualize as settings_v
from behave_analysis.analyze.AnalyzeBehave import identify_condition, base_plotting

matplotlib.use("TKAgg")


class Visualize_behave:
    """
    A class for some sanity check behavior plots
    to get a sense for what the mouse was doing in the session
    """

    def __init__(self, session, postprocessingObj):
        self.session = session
        self.behave_path = os.path.join(self.session.base_path, self.session.processed_path, "behaviour")
        self.tracking_data = postprocessingObj.tracking_data
        self.video_df = pl.read_csv(
            os.path.join(self.session.base_path, self.session.processed_path, "full_video_dataframe.csv")
        )
        self.post_process_obj = postprocessingObj
        if not os.path.exists(self.behave_path):
            os.makedirs(self.behave_path)

    def plot_behavioral_stats(self):
        """Excute behaviour plotting functions"""

        logger.info("Making plots summarizing the exploratory behavior of the mouse ")
        self.shelter_occupancy()
        self.position_by_bsa()
        self.location_occupancy()

        # Plotting angle distriubtions must go before plotting the circular rho
        plot_angle_distributions(
            session=self.session,
            trackingData=self.tracking_data,
            videoDf=self.video_df,
            sessionHeight=self.session.video.height,
            save_path=self.behave_path,
        )

        # Circular rho depends on angle distributions
        plot_the_circular_rho(self.video_df, save_path=self.behave_path)

        plot_heat_map_of_position(
            session=self.session,
            video_data_frame=self.video_df,
            save_path=self.behave_path,
            session_height=self.session.video.height,
        )

        CoverageStatistics(video_data_frame=self.video_df, session=self.session, behave_path=self.behave_path)

    def escape_plotting(self):
        logger.info("Making plots of mouse escape trajectories")
        self.escape_trajectory_and_shelter_exits()

    ## ------------- PLOTTING FUNCTIONS
    def position_by_bsa(self):
        """Make a scatter plot of position in arena colored by angle between body and shelter"""
        outofShelterIdx = np.array(
            self.video_df["OutofshelterIdx"].to_numpy()
        )  # remove times when mouse is inside shelter

        # color position by their shelter angle
        mass = self.tracking_data["avg_loc"][outofShelterIdx, :]
        ang_color = np.digitize(np.rad2deg(self.tracking_data["bod_shelt_dir"][outofShelterIdx]), np.arange(-180, 180))
        bsa_rgb_cycle = hsv_hdir_colormap(ang_color)
        plt.figure()
        plt.scatter(mass[:, 0], mass[:, 1], s=5, c=bsa_rgb_cycle, linewidths=0, marker=".")
        plt.title("position coloured by angle to shelter")
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect("equal")
        plt.savefig(os.path.join(self.behave_path, "arena_position.png"))
        if settings_v.show_plots:
            plt.show()
        plt.close()

    def shelter_occupancy(self):
        """Make a bar plot of minutes in and out of shelter per condition in each session"""

        if settings_v.user_defined_conditions:
            conditions = settings_v.conditions
        else:
            conditions = identify_conditions(self.session)

        _, ax = plt.subplots(figsize=(10, 5))

        for x, c in enumerate(conditions):
            # Plot out of shelter
            time_out_of_shelter = (
                len(filter_video_dataframe(self.video_df, c, outofshelter=True, exclude_escape=False))
                / self.session.video.fps
            )
            plt.bar(
                x + 0.9,
                time_out_of_shelter / 60,
                width=0.2,
                color="dimgrey",
            )

            # Plot in shelter
            time_in_shelter = (
                len(filter_video_dataframe(self.video_df, c, outofshelter=False, exclude_escape=False))
                / self.session.video.fps
            )
            plt.bar(
                x + 1.1,
                time_in_shelter / 60,
                width=0.2,
                color="lightgrey",
            )

        # Plot settings
        # sns.set()
        plt.xticks(np.arange(len(conditions)) + 1, conditions, rotation=35)
        plt.ylabel("Time (mins)")
        plt.legend(["out of shelter", "in shelter"], facecolor="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.behave_path, "shelter_occupancy.png"))
        if settings_v.show_plots:
            plt.show()
        plt.close()

    def location_occupancy(self):
        """
        Make plots showing time in shelter, near barrier edged and in 4 quadrants of arena over the course of the session
        """
        # look in a 3 minute window
        w = 3
        # x axis values for plotting, in minutes
        x = np.arange(
            w / 2,
            (self.session.video.num_frames / (self.session.video.fps * 60))
            - (w / 2)
            + 1 / (self.session.video.fps * 60),
            1 / (self.session.video.fps * 60),
        )
        figg, axs = plt.subplots(1, 3)
        figg.set_figwidth(15)
        # time in shelter
        if len(self.session.shelter_time) > 0:
            if "mushroom" in self.session.experiment:
                extra = 50  # in the mushroom session extend what the shelter is beyond the base
            else:
                extra = 0
            InShelterIdx = np.logical_and(
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 0] > self.tracking_data["shelter_loc"][0][0] - extra,
                    self.tracking_data["avg_loc"][:, 0] < self.tracking_data["shelter_loc"][1][0] + extra,
                ),
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 1] > self.tracking_data["shelter_loc"][0][1] - extra,
                    self.tracking_data["avg_loc"][:, 1] < self.tracking_data["shelter_loc"][1][1] + extra,
                ),
            )
            axs[0].plot(
                x,
                np.convolve(InShelterIdx.astype(int), np.ones(self.session.video.fps * 60 * w), "valid")
                / (self.session.video.fps * 60 * w),
            )
            axs[0].plot(
                [self.post_process_obj.sheltertime[0] / 60, self.post_process_obj.sheltertime[0] / 60], [0, 1], "-k"
            )
            axs[0].title.set_text("In shelter")
            axs[0].set_xlabel("time (mins)")
            axs[0].set_ylabel("fraction occupancy")

        # time in 4 quadrants
        cc = matplotlib.cm.Set1
        center = [self.session.video.width / 2, self.session.video.height / 2]
        Q = np.vstack(
            (
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 0] < center[0], self.tracking_data["avg_loc"][:, 1] < center[1]
                ),  # upper_left
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 0] > center[0], self.tracking_data["avg_loc"][:, 1] < center[1]
                ),  # upper_right
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 0] > center[0], self.tracking_data["avg_loc"][:, 1] > center[1]
                ),  # lower_right
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 0] < center[0], self.tracking_data["avg_loc"][:, 1] > center[1]
                ),
            )
        )  # lower_left
        for i in np.arange(4):
            axs[1].plot(
                x,
                np.convolve(Q[i, :].astype(int), np.ones(self.session.video.fps * 60 * w), "valid")
                / (self.session.video.fps * 60 * w),
                color=cc(i),
            )
        axs[1].title.set_text("In quadrants")
        axs[1].legend(["upper_left", "upper_right", "lower_right", "lower_left"])
        axs[1].set_xlabel("time (mins)")

        # time near barrier edge
        if len(self.session.barrier_time) > 0:
            for i, c in enumerate(self.tracking_data["barrier_loc"]):
                extra = 35  #
                NearBarrier = np.logical_and(
                    np.logical_and(
                        self.tracking_data["avg_loc"][:, 0] > c[0] - extra,
                        self.tracking_data["avg_loc"][:, 0] < c[0] + extra,
                    ),
                    np.logical_and(
                        self.tracking_data["avg_loc"][:, 1] > c[1] - extra,
                        self.tracking_data["avg_loc"][:, 1] < c[1] + extra,
                    ),
                )
                axs[2].plot(
                    x,
                    np.convolve(NearBarrier.astype(int), np.ones(self.session.video.fps * 60 * w), "valid")
                    / (self.session.video.fps * 60 * w),
                    color=cc(i),
                )
            axs[2].plot(
                [self.post_process_obj.barriertime[0] / 60, self.post_process_obj.barriertime[0] / 60], [0, 1], "-k"
            )
        axs[2].set_xlabel("time (mins)")
        axs[2].legend(["left_edge", "right_edge"])
        axs[2].title.set_text("Near barrier edge")

        plt.savefig(os.path.join(self.behave_path, "arena_occupancy_vs_time.png"))
        if settings_v.show_plots:
            plt.show()
        plt.close()

    def angle_histograms(self):
        """
        Make histograms of head direction, head shelter angle and barrier shelter angle to ensure good sampling
        """
        figg, axs = plt.subplots(1, 3)
        figg.set_figwidth(15)

        # time in shelter (we're excluding this from our histograms)
        if "mushroom" in self.session.experiment:
            extra = 50  # in the mushroom session extend what the shelter is beyond the base
        else:
            extra = 0
        OutofShelterIdx = np.logical_not(
            np.logical_and(
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 0] > self.tracking_data["shelter_loc"][0][0] - extra,
                    self.tracking_data["avg_loc"][:, 0] < self.tracking_data["shelter_loc"][1][0] + extra,
                ),
                np.logical_and(
                    self.tracking_data["avg_loc"][:, 1] > self.tracking_data["shelter_loc"][0][1] - extra,
                    self.tracking_data["avg_loc"][:, 1] < self.tracking_data["shelter_loc"][1][1] + extra,
                ),
            )
        )
        # head direction
        axs[0].hist(
            self.tracking_data["hdir"][OutofShelterIdx], np.arange(-np.pi, np.pi, np.pi / 10), density="stacked"
        )
        axs[0].set_ylabel("fraction of frames")
        axs[0].title.set_text("head dir")

        # head shelter angle
        if len(self.session.shelter_time) > 0:
            # only for times when there is a shelter-only
            frames_with_shelter = np.zeros_like(self.tracking_data["hdir_shelt"])
            if self.post_process_obj.sheltertime[1] == -60:
                frames_with_shelter[self.post_process_obj.sheltertime[0] * self.session.video.fps :] = 1
            else:
                frames_with_shelter[
                    self.post_process_obj.sheltertime[0]
                    * self.session.video.fps : self.post_process_obj.sheltertime[1]
                    * self.session.video.fps
                ] = 1
            axs[1].hist(
                self.tracking_data["hdir_shelt"][np.logical_and(OutofShelterIdx, frames_with_shelter == 1)],
                np.arange(-np.pi, np.pi, np.pi / 10),
                density="stacked",
            )
            axs[1].title.set_text("head shelter angle")

        # head barrier angle
        if len(self.session.barrier_time) > 0:
            # only for times when there is a barrier
            frames_with_barrier = np.zeros_like(self.tracking_data["hdir_shelt"])
            if self.post_process_obj.barriertime[1] == -60:
                frames_with_barrier[self.post_process_obj.barriertime[0] * self.session.video.fps :] = 1
            else:
                frames_with_barrier[
                    self.post_process_obj.barriertime[0]
                    * self.session.video.fps : self.post_process_obj.barriertime[1]
                    * self.session.video.fps
                ] = 1
            for c in np.arange(2):
                axs[2].hist(
                    self.tracking_data["hdir_barrier"][np.logical_and(OutofShelterIdx, frames_with_shelter == 1), c],
                    np.arange(-np.pi, np.pi, np.pi / 10),
                    density="stacked",
                )
            axs[2].title.set_text("head barrier-edge angle")

        plt.savefig(os.path.join(self.behave_path, "distribution_head_angles.png"))
        if settings_v.show_plots:
            plt.show()
        plt.close()

    def escape_trajectory_and_shelter_exits(self):
        """
        Plot escape trajectories as well as the path by which the mouse last left the shelter
        """

        # set up figure and number of rows and calculate number of columns
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        ntrial = len(self.session.__dict__[settings_v.stim_type].onset_frames)
        nrows = 3
        ncols = ntrial // nrows + (ntrial % nrows > 0)

        for trial_num, (onset_frames, stimulus_durations) in enumerate(
            zip(
                self.session.__dict__[settings_v.stim_type].onset_frames,
                self.session.__dict__[settings_v.stim_type].stimulus_durations,
            )
        ):
            ax = plt.subplot(nrows, ncols, trial_num + 1)
            # set up axes with shelt and barrier locations
            condition = identify_condition(self.video_df.filter(self.video_df["frames"] == onset_frames), self.session)
            base_plotting(ax, self.tracking_data, condition)
            # plot escape trajectory
            plot_trajectories(
                self.tracking_data["head_loc"],
                self.tracking_data["avg_Velocity"],
                onset_frames[0],
                stimulus_durations[0],
                ax,
            )
            # plot shelter exit trajectory
            exits = np.where(np.diff(self.video_df["OutofshelterIdx"].to_numpy().astype(int)) == 1)[0]
            exits = exits[exits < onset_frames[0]]
            plot_trajectories(
                self.tracking_data["head_loc"],
                self.tracking_data["avg_Velocity"],
                exits[-1] - self.session.video.fps,
                3,
                ax,
                colors="Blues",
            )

        filename = str(self.behave_path) + "/" + "Escape_and_Exit" + ".png"
        plt.savefig(filename)
        if settings_v.show_plots:
            plt.show()
        plt.close()


def plot_trajectories(head_loc, velocity, onset_frames, stimulus_durations, ax, colors="Reds"):
    """
    Plot escape trajectories
    """
    # compute and plot each trajectory
    x_loc = head_loc[onset_frames : onset_frames + int(stimulus_durations * 40), 0]
    y_loc = head_loc[onset_frames : onset_frames + int(stimulus_durations * 40), 1]
    speed = velocity[onset_frames : onset_frames + int(stimulus_durations * 40)]
    ax.scatter(x_loc, y_loc, s=5, c=speed, cmap=colors)

def hsv_hdir_colormap(angles):
    """
    Make a colormap for circular variables like hdir
    input is array of angles you need to assign a colour to"""
    phi = np.linspace(0, 2 * np.pi, len(np.arange(360)))
    rgb_cycle = np.vstack(
        (  # Three sinusoids
            0.5 * (1.0 + np.cos(phi)),  # scaled to [0,1]
            0.5 * (1.0 + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
            0.5 * (1.0 + np.cos(phi - 2 * np.pi / 3)),
        )
    ).T  # Shape = (60,3)
    bsa_rgb_cycle = np.zeros(shape=(len(angles), 3))
    for i in np.arange(360):
        bsa_rgb_cycle[angles == i + 1, :] = rgb_cycle[i, :]
    return bsa_rgb_cycle