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
from behave_analysis.visualize.behaviour.behavioral_stats import shelter_occupancy,position_by_bsa,location_occupancy
from behave_analysis.visualize.behaviour.escape_trajectory import escape_trajectory_and_shelter_exits
from behave_analysis.visualize.behaviour.escape_movies import trial_movies
from settings.settings_visualize import defined_settings_visualize as settings_v

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
        shelter_occupancy(video_df = self.video_df, 
                          session = self.session, 
                          settings = settings_v, 
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
                                trackingData=self.tracking_data,
                                video_data=self.video_df,
                                sessionHeight=self.session.video.height,
                                save_path=self.behave_path)

        # Circular rho depends on angle distributions
        plot_the_circular_rho(self.session, 
                              self.video_df, 
                              save_path=self.behave_path)

        plot_heat_map_of_position(session=self.session,
                                video_data_frame=self.video_df,
                                save_path=self.behave_path,
                                session_height=self.session.video.height)

        CoverageStatistics(video_data_frame=self.video_df, 
                           session=self.session, 
                           behave_path=self.behave_path)

    def escape_plotting(self):
        logger.info("Making plots of mouse escape trajectories")
        escape_trajectory_and_shelter_exits(tracking_data = self.tracking_data,
                                            video_df = self.video_df, 
                                            session = self.session, 
                                            settings = settings_v, 
                                            save_path = self.behave_path)

    def escape_movies(self, kalman):
        logger.info(f"Starting to make movies of mousie escape")
        print("\nPress 'q' to quit and 'n' to move to the next video")
        trial_movies(tracking_data = self.tracking_data, 
                     kalman = kalman, 
                     session = self.session, 
                     settings = settings_v, 
                     stim_type = settings_v.stim_type)