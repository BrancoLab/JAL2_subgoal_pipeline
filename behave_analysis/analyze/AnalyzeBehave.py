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
from behave_analysis.utils.data_loading import load_or_extract_escapes
from behave_analysis.analyze.behaviour.homings_escapes.homings import add_homie_to_video_df, get_Homings
from behave_analysis.analyze.behaviour.homings_escapes.escapes import get_Escapes
from behave_analysis.analyze.behaviour.correlation_matrix import compute_correlation_matrix, plot_correlation_matrix, circular_linear_corr
from settings.settings_overrides import settings_overrides

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

        if analysis_name == "escape_plots" or analysis_name == 'correlations':
            self.escape_object = load_or_extract_escapes(self.session)
            assert self.escape_object is not None, "Failed to load escape data."
            assert len(self.escape_object["onset_frames"]) > 0, "No escape trials found for this session."

        if analysis_name == "homings_plots" or analysis_name == "correlations":
            from settings.settings_analyze_behave import settings_ab
            settings_ab = settings_overrides(settings_ab, {"redo_compute": False})
            self.homings = get_Homings({**settings_ab, "homings_curated": True}, self.session).get_homings()
            assert self.homings is not None, "Failed to load homing data."

        if analysis_name in ['homings&escape', 'correlations']:
            # load behavioral data
            self.video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv")
            self.video_df = add_homie_to_video_df(self.session, self.video_df, self.tracking_data)

    def behaviour_analyses(self, analysis_name, variables=None):
        
        
        # ----------------------------- Find Homings ----------------------------------

        if analysis_name == 'homings&escape':
            """Let's check out some homings and threshold crossings."""
            logger.info("The homings pipeline has started")
            homings_dict = get_Homings(settings=self.settings, session=self.session).get_homings(video_df = self.video_df, tracking_data = self.tracking_data)
            get_Escapes(settings=self.settings, session=self.session, tracking_data = self.tracking_data, video_df = self.video_df, homings = homings_dict).get_escape()
            logger.success("Homing & escapes pipeline complete")


        if analysis_name in ["escape_plots", "homings_plots"]:

            if analysis_name == "escape_plots":
                trials = "Escapes"
                trial_obj = self.escape_object
            elif analysis_name == "homings_plots":
                trials = "Homings"
                trial_obj = self.homings

            logger.info(f"Making plots of {trials}")
            spatial_efficiency(self.onsets,
                                trial_obj["stimulus_durations"],
                                self.session,
                                self.settings,
                                trial_obj["condition"],
                                self.tracking_data,
                                trial_type=trials,
                                plotting=True,
                                save_dir=self.dir,
                            )

            plot_the_start_of_each_run(session=self.session,
                                        onsets=trial_obj["onset_frames"],
                                        hdir_at_start=trial_obj["starting_hdir"],
                                        all_conditions=trial_obj["condition"],
                                        tracking_data=self.tracking_data,
                                        title=trials)

            plot_the_probability_of_start_locations(session=self.session,
                                                    onset_frames=trial_obj["onset_frames"],
                                                    all_conditions=trial_obj["condition"],
                                                    tracking_data=self.tracking_data,
                                                    title=trials,
                                                )

            trial_speed_hist(session=self.session, 
                            avg_speed=trial_obj["avg_speed"], 
                            title=trials)

            trial_initial_heading_angle(session=self.session,
                                        onsets=trial_obj["onset_frames"],
                                        offsets=trial_obj["offset_frames"],
                                        head_angle=trial_obj["head_orientation_dic"]["avg_hdir"],
                                        hdir_at_start=trial_obj["starting_hdir"],
                                        all_conditions=trial_obj["condition"],
                                        tracking_data=self.tracking_data,
                                        title=trials,
                                    )

            trajectory_by_target(session=self.session,
                                onsets=trial_obj["onset_frames"],
                                offsets=trial_obj["offset_frames"],
                                head_angle=trial_obj["head_orientation_dic"]["avg_hdir"],
                                all_conditions=trial_obj["condition"],
                                tracking_data=self.tracking_data,
                                title=trials,
                            )

            hist_initial_heading_angle(session=self.session,
                                        onsets=trial_obj["onset_frames"],
                                        offsets=trial_obj["offset_frames"],
                                        head_angle=trial_obj["head_orientation_dic"]["avg_hdir"],
                                        all_conditions=trial_obj["condition"],
                                        tracking_data=self.tracking_data,
                                        title=trials,
                                    )
        
        if analysis_name == 'correlations':
            """Let's build correlation matrices between behavioral variables"""
            assert variables is not None, "Please provide a list of variables to compute correlations between."
            corr_matrix = compute_correlation_matrix(self, variables)
            plot_correlation_matrix(corr_matrix, variables)
            

        # this one is kind of redundant with the spatial efficiency plots
        # plot_homings(self.session, self.tracking_data, homings_obj, settings.show_plots)

        # this one doesn't really work yet
        # homing_head_angle_trajectory(self.session, homings_obj, self.tracking_data)
