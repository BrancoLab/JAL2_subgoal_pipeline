from loguru import logger
import polars as pl
import os
import numpy as np

# Custom classes
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.behaviour.plot_homings import plot_homings
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
        """Load in video df"""
        video_df = os.path.join(session.base_path, session.processed_path) + "\\" + "full_video_dataframe.csv"
        if os.path.isfile(video_df):
            self.video_df = pl.read_csv(video_df)
        else:
            raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")

    def behaviour_analyses(self):
        logger.info(f"Making plots of spatial effciency in escape")
        spatial_efficiency(
            np.array(self.session.__dict__[settings.stim_type].onset_frames),
            np.array(self.session.__dict__[settings.stim_type].stimulus_durations),
            self.session,
            settings,
            self.video_df,
            self.tracking_data,
            plotting=True,
            save_dir=self.dir,
        )

        logger.info(f"Making plots of homing trajectories")
        plot_homings(self.session, self.tracking_data, self.video_df)
