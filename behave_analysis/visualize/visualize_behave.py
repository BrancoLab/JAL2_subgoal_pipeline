"""A module that organizes the visualization of the behavior data"""

import os
from loguru import logger
import numpy as np
import polars as pl

from behave_analysis.visualize.behaviour.circular_coeff_of_angles import plot_the_circular_rho
from behave_analysis.visualize.behaviour.behaviour_coverage_metrics import CoverageStatistics
from behave_analysis.visualize.behaviour.heat_plot import plot_heat_map_of_position
from behave_analysis.visualize.behaviour.angle_distributions import plot_angle_distributions
from behave_analysis.visualize.behaviour.behavioral_stats import shelter_occupancy,position_by_bsa,location_occupancy
from behave_analysis.visualize.behaviour.escape_trajectory import escape_trajectory_and_shelter_exits
from behave_analysis.visualize.behaviour.movies import trial_movies
from behave_analysis.visualize.visualize_utils import open_tracking_data, open_kalman_tracking_data
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import extract_all_or_custom_conditions
from settings.settings_visualize import defined_settings_visualize as settings_v

class Visualize_behave:
    """
    A class for some sanity check behavior plots
    to get a sense for what the mouse was doing in the session
    """

    def __init__(self, session):
        self.session = session
        self.behave_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "behaviour"))
        self.tracking_data = open_tracking_data(session)
        self.kalman = open_kalman_tracking_data(os.path.join(self.session.base_path, self.session.processed_path))
        self.video_df = pl.read_csv(
            os.path.join(self.session.base_path, self.session.processed_path, "full_video_dataframe.csv")
        )

##---------PLOT BEHAVIORAL STATS
    def plot_behavioral_stats(self):
        """Excute behaviour plotting functions"""

        logger.info("Making plots summarizing the exploratory behavior of the mouse ")
        shelter_occupancy(video_df = self.video_df, 
                          session = self.session, 
                          settings = settings_v, 
                          conditions = extract_all_or_custom_conditions(settings_v, self.session),
                          save_path = self.behave_path)
        position_by_bsa(tracking_data = self.tracking_data,
                        outofShelterIdx = np.array(self.video_df["OutofshelterIdx"].to_numpy()),
                        settings = settings_v, 
                        save_path = self.behave_path)
        location_occupancy(tracking_data = self.tracking_data,
                           session = self.session, 
                           settings = settings_v, 
                           save_path = self.behave_path)

        # Plotting angle distriubtions must go before plotting the circular rho
        plot_angle_distributions(session=self.session,
                                 settings = settings_v,
                                 trackingData=self.tracking_data,
                                 video_data=self.video_df,
                                 conditions = extract_all_or_custom_conditions(settings_v, self.session),
                                 sessionHeight=self.session.video.height,
                                 save_path=self.behave_path)

        # Circular rho depends on angle distributions
        plot_the_circular_rho(self.session, 
                              settings_v,
                              self.video_df, 
                              conditions = extract_all_or_custom_conditions(settings_v, self.session),
                              save_path=self.behave_path)

        plot_heat_map_of_position(session=self.session,
                                  settings = settings_v,
                                  video_data_frame=self.video_df,
                                  conditions = extract_all_or_custom_conditions(settings_v, self.session),
                                  save_path=self.behave_path,
                                  session_height=self.session.video.height)

        CoverageStatistics(video_data_frame=self.video_df, 
                           settings = settings_v,
                           behave_path=self.behave_path)

##--------PLOT TRAJECTORIES OF ESCAPE
    def escape_plotting(self):
        logger.info("Making plots of mouse escape trajectories")
        escape_trajectory_and_shelter_exits(tracking_data = self.tracking_data,
                                            video_df = self.video_df, 
                                            session = self.session, 
                                            settings = settings_v, 
                                            save_path = self.behave_path)

##--------MAKE MOVIS OF ESCAPE WITH DLC TRACKING
    def make_movies(self, stim_type):
        """Make movies for homings or escapes"""
        logger.info(f"Starting to make behaviour movies for trials")
        trial_movies(tracking_data = self.tracking_data, 
                     kalman = self.kalman, 
                     session = self.session, 
                     settings = settings_v, 
                     stim_type = stim_type)