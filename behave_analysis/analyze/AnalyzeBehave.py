from loguru import logger
import polars as pl
import os
import numpy as np

# Custom classes
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.behaviour.plot_homings import (
    plot_homings,
    plot_the_start_of_each_run,
    plot_the_probability_of_start_locations,
    trajectory_by_target,
    hist_initial_heading_angle,
    trial_initial_heading_angle,
    trial_speed_hist,
)
from behave_analysis.utils.data_loading import load_or_extract_homings, load_or_extract_escapes
from settings.settings_analyze import settings_analyze as settings


class AnalyzeBehave:
    """
    A class that analyzes mouse behavior in a session
    """

    def __init__(self, session):
        logger.info("Initializing AnalyzeBehave")
        self.dir = make_directory(os.path.join(session.base_path, session.processed_path) + "\\" + "analyze_behave")
        self.session = session
        self.settings = settings
        self.tracking_data = open_tracking_data(self.session)

    def behaviour_analyses(self):
        esc_obj = load_or_extract_escapes(self.session)
        assert esc_obj is not None, "Failed to load homing data."
        assert hasattr(esc_obj, "escape_onset_frames") and hasattr(
            esc_obj, "stimulus_durations"
        ), "Escape object must have 'onset_frames' and 'stimulus_durations'."

        if len(esc_obj.escape_onset_frames) > 0:
            logger.info(f"Making plots of spatial effciency in escape")
            spatial_efficiency(
                esc_obj.escape_onset_frames,
                esc_obj.stimulus_durations,
                self.session,
                settings,
                esc_obj.escape_condition,
                self.tracking_data,
                trial_type="Escapes",
                plotting=True,
                save_dir=self.dir,
            )

            plot_the_start_of_each_run(session=self.session,
                                        onsets=esc_obj.escape_onset_frames,
                                        hdir_at_start=esc_obj.start_head_ori,
                                        all_conditions=esc_obj.escape_condition,
                                        tracking_data=self.tracking_data,
                                        title="Escapes")

            plot_the_probability_of_start_locations(
                session=self.session, 
                onset_frames=esc_obj.escape_onset_frames, 
                all_conditions=esc_obj.escape_condition, 
                tracking_data=self.tracking_data, 
                title="Escapes",
            )

            trial_speed_hist(session=self.session, 
                            avg_speed=esc_obj.avg_speed, 
                            title="Escapes")
            
            trial_initial_heading_angle(
                session=self.session,
                onsets=esc_obj.escape_onset_frames,
                offsets=esc_obj.escape_end_frames,
                head_angle=esc_obj.head_orientation["avg_hdir"],
                hdir_at_start=esc_obj.start_head_ori,
                all_conditions=esc_obj.escape_condition,
                tracking_data=self.tracking_data,
                title="Escapes",
            )

            trajectory_by_target(
                session=self.session,
                onsets=esc_obj.escape_onset_frames,
                offsets=esc_obj.escape_end_frames,
                head_angle=esc_obj.head_orientation["avg_hdir"],
                all_conditions=esc_obj.escape_condition,
                tracking_data=self.tracking_data,
                title="Escapes",
            )

            hist_initial_heading_angle(session=self.session,
                onsets=esc_obj.escape_onset_frames,
                offsets=esc_obj.escape_end_frames,
                head_angle=esc_obj.head_orientation["avg_hdir"],
                all_conditions=esc_obj.escape_condition,
                tracking_data=self.tracking_data,
                title="Escapes",
            )

        homings_obj = load_or_extract_homings(self.session)
        assert homings_obj is not None, "Failed to load homing data."
        assert hasattr(homings_obj, "onset_frames") and hasattr(
            homings_obj, "stimulus_durations"
        ), "Homings object must have 'onset_frames' and 'stimulus_durations'."

        logger.info(f"Making plots of homing trajectories")
        spatial_efficiency(
            homings_obj.onset_frames,
            homings_obj.stimulus_durations,
            self.session,
            settings,
            homings_obj.homing_condition,
            self.tracking_data,
            trial_type="Homing",
            plotting=True,
            save_dir=self.dir,
        )

        plot_the_start_of_each_run(
            session=self.session,
            onsets=homings_obj.onset_frames,
            hdir_at_start=homings_obj.hdir_at_start,
            all_conditions=homings_obj.homing_condition,
            tracking_data=self.tracking_data,
            title="Homing",
        )

        plot_the_probability_of_start_locations(
            session=self.session, 
            onset_frames=homings_obj.onset_frames, 
            all_conditions=homings_obj.homing_condition, 
            tracking_data=self.tracking_data, 
            title="Homing",
        )

        trial_initial_heading_angle(
            session=self.session,
            onsets=homings_obj.onset_frames,
            offsets=homings_obj.offset_frames,
            head_angle=homings_obj.homing_angles_dic["avg_hdir"],
            hdir_at_start=homings_obj.hdir_at_start,
            all_conditions=homings_obj.homing_condition,
            tracking_data=self.tracking_data,
            title="Homing",
        )

        trajectory_by_target(
            session=self.session,
            onsets=homings_obj.onset_frames,
            offsets=homings_obj.offset_frames,
            head_angle=homings_obj.homing_angles_dic["avg_hdir"],
            all_conditions=homings_obj.homing_condition,
            tracking_data=self.tracking_data,
            title="Homing",
        )

        hist_initial_heading_angle(
            session=self.session,
            onsets=homings_obj.onset_frames,
            offsets=homings_obj.offset_frames,
            head_angle=homings_obj.homing_angles_dic["avg_hdir"],
            all_conditions=homings_obj.homing_condition,
            tracking_data=self.tracking_data,
            title="Homing",
        )

        trial_speed_hist(self.session, homings_obj.avg_speed, title="Homing")

        # this one is kind of redundant with the spatial efficiency plots
        # plot_homings(self.session, self.tracking_data, homings_obj, settings.show_plots)

        # this one doesn't really work yet
        # homing_head_angle_trajectory(self.session, homings_obj, self.tracking_data)
