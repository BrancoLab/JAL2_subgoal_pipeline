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
from behave_analysis.analyze.behaviour.homings_escapes.homings import get_Homings
from behave_analysis.analyze.behaviour.homings_escapes.escapes import get_Escapes

class AnalyzeBehave:
    """
    A class that analyzes mouse behavior in a session
    """

    def __init__(self, session, settings):
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

        if analysis_name == "homings_plots":
            self.homings_obj = load_or_extract_homings(self.session)
            assert self.homings_obj is not None, "Failed to load homing data."
            assert hasattr(self.homings_obj, "onset_frames") and hasattr(
                self.homings_obj, "stimulus_durations"
            ), "Homings object must have 'onset_frames' and 'stimulus_durations'."

        if analysis_name == 'homings&escape':
            # load behavioral data
            self.video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv")
            

    def behaviour_analyses(self, analysis_name):
        
        
        # ----------------------------- Find Homings ----------------------------------

        if analysis_name == 'homings&escape':
            """Let's check out some homings and threshold crossings."""
            logger.info("The homings pipeline has started")
            homings_obj = get_Homings(settings=self.settings, session=self.session).get_homings()
            get_Escapes(settings=self.settings, session=self.session, tracking_data = self.tracking_data, video_df = self.video_df, homings = homings_obj).get_escape()
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
                                trial_obj.condition,
                                self.tracking_data,
                                trial_type=trials,
                                plotting=True,
                                save_dir=self.dir,
                            )

            plot_the_start_of_each_run(session=self.session,
                                        onsets=trial_obj.onset_frames,
                                        hdir_at_start=self.starting_hdir,
                                        all_conditions=trial_obj.condition,
                                        tracking_data=self.tracking_data,
                                        title=trials)

            plot_the_probability_of_start_locations(session=self.session,
                                                    onset_frames=trial_obj.onset_frames,
                                                    all_conditions=trial_obj.condition,
                                                    tracking_data=self.tracking_data,
                                                    title=trials,
                                                )

            trial_speed_hist(session=self.session, 
                            avg_speed=trial_obj.avg_speed, 
                            title=trials)

            trial_initial_heading_angle(session=self.session,
                                        onsets=trial_obj.onset_frames,
                                        offsets=trial_obj.offset_frames,
                                        head_angle=self.head_angles_dic["avg_hdir"],
                                        hdir_at_start=trial_obj.hdir_at_start,
                                        all_conditions=trial_obj.condition,
                                        tracking_data=self.tracking_data,
                                        title=trials,
                                    )

            trajectory_by_target(session=self.session,
                                onsets=trial_obj.onset_frames,
                                offsets=trial_obj.offset_frames,
                                head_angle=trial_obj.head_orientation_dic["avg_hdir"],
                                all_conditions=trial_obj.condition,
                                tracking_data=self.tracking_data,
                                title=trials,
                            )

            hist_initial_heading_angle(session=self.session,
                                        onsets=trial_obj.onset_frames,
                                        offsets=trial_obj.offset_frames,
                                        head_angle=trial_obj.head_orientation_dic["avg_hdir"],
                                        all_conditions=trial_obj.condition,
                                        tracking_data=self.tracking_data,
                                        title=trials,
                                    )

        # this one is kind of redundant with the spatial efficiency plots
        # plot_homings(self.session, self.tracking_data, homings_obj, settings.show_plots)

        # this one doesn't really work yet
        # homing_head_angle_trajectory(self.session, homings_obj, self.tracking_data)
