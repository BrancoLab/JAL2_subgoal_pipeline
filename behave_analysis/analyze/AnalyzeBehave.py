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
from settings.settings_analyze_behave import settings_analyze_behave as settings
from behave_analysis.analyze.homings_escapes.homings import get_Homings
from behave_analysis.analyze.homings_escapes.escapes import get_Escapes

class AnalyzeBehave:
    """
    A class that analyzes mouse behavior in a session
    """

    def __init__(self, session):
        logger.info("Initializing AnalyzeBehave")
        self.dir = make_directory(os.path.join(session.base_path, session.processed_path) + "\\" + "analyze_behave")
        self.session = session
        self.settings = settings

    def load_data(self, analysis_name):
        self.tracking_data = open_tracking_data(self.session)

        if analysis_name == "escape_plots":
            self.esc_obj = load_or_extract_escapes(self.session)
            assert self.esc_obj is not None, "Failed to load homing data."
            assert hasattr(self.esc_obj, "escape_onset_frames") and hasattr(
                self.esc_obj, "stimulus_durations"
            ), "Escape object must have 'onset_frames' and 'stimulus_durations'."
            assert len(self.esc_obj.escape_onset_frames) > 0, "No escape trials found for this session."
            self.onsets = self.esc_obj.escape_onset_frames
            self.conditions = self.esc_obj.escape_condition
            self.offsets = self.esc_obj.escape_end_frames
            self.starting_hdir = self.esc_obj.start_head_ori
            self.head_angles_dic = self.esc_obj.head_orientation

        if analysis_name == "homings_plots":
            self.homings_obj = load_or_extract_homings(self.session)
            assert self.homings_obj is not None, "Failed to load homing data."
            assert hasattr(self.homings_obj, "onset_frames") and hasattr(
                self.homings_obj, "stimulus_durations"
            ), "Homings object must have 'onset_frames' and 'stimulus_durations'."
            self.onsets = self.homings_obj.onset_frames
            self.conditions = self.homings_obj.homing_condition
            self.offsets = self.homings_obj.offset_frames
            self.starting_hdir = self.homings_obj.hdir_at_start
            self.head_angles_dic = self.homings_obj.homing_angles_dic

        if analysis_name == 'homings&escape':
            # load behavioral data
            self.video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv")
            

    def behaviour_analyses(self, analysis_name):
        
        
        # ----------------------------- Find Homings ----------------------------------

        if analysis_name == 'homings&escape':
            """Let's check out some homings and threshold crossings."""
            logger.info("The homings pipeline has started")
            homings_obj = get_Homings(settings=self.settings, session=self.session)
            get_Escapes(settings=self.settings, session=self.session, tracking_data = self.tracking_data, video_df = self.video_df, homings = homings_obj.session.homing)
            logger.success("Homing & escapes pipeline complete")


        if analysis_name in ["escape_plots", "homings_plots"]:

            if analysis_name == "escape_plots":
                trials = "Escapes"
                trial_obj = self.esc_obj
            elif analysis_name == "homings_plots":
                trials = "Homings"
                trial_obj = self.homings_obj

            logger.info(f"Making plots of {trials}")
            spatial_efficiency(self.onsets,
                                trial_obj.stimulus_durations,
                                self.session,
                                self.settings,
                                self.conditions,
                                self.tracking_data,
                                trial_type=trials,
                                plotting=True,
                                save_dir=self.dir,
                            )

            plot_the_start_of_each_run(session=self.session,
                                        onsets=self.onsets,
                                        hdir_at_start=self.starting_hdir,
                                        all_conditions=self.conditions,
                                        tracking_data=self.tracking_data,
                                        title=trials)

            plot_the_probability_of_start_locations(session=self.session,
                                                    onset_frames=self.onsets,
                                                    all_conditions=self.conditions,
                                                    tracking_data=self.tracking_data,
                                                    title=trials,
                                                )

            trial_speed_hist(session=self.session, 
                            avg_speed=trial_obj.avg_speed, 
                            title=trials)

            trial_initial_heading_angle(session=self.session,
                                        onsets=self.onsets,
                                        offsets=self.offsets,
                                        head_angle=self.head_angles_dic["avg_hdir"],
                                        hdir_at_start=self.starting_hdir,
                                        all_conditions=self.conditions,
                                        tracking_data=self.tracking_data,
                                        title=trials,
                                    )

            trajectory_by_target(session=self.session,
                                onsets=self.onsets,
                                offsets=self.offsets,
                                head_angle=self.head_angles_dic["avg_hdir"],
                                all_conditions=self.conditions,
                                tracking_data=self.tracking_data,
                                title=trials,
                            )

            hist_initial_heading_angle(session=self.session,
                                        onsets=self.onsets,
                                        offsets=self.offsets,
                                        head_angle=self.head_angles_dic["avg_hdir"],
                                        all_conditions=self.conditions,
                                        tracking_data=self.tracking_data,
                                        title=trials,
                                    )

        # this one is kind of redundant with the spatial efficiency plots
        # plot_homings(self.session, self.tracking_data, homings_obj, settings.show_plots)

        # this one doesn't really work yet
        # homing_head_angle_trajectory(self.session, homings_obj, self.tracking_data)
