import os

from loguru import logger
import cv2
import numpy as np
import dill as pickle

from settings.settings_visualize import defined_settings_visualize as settings

class Visualize:
    """
    Create visualize object, loading in processed data for plotting
    #TODO: Make this class have one responsbiility"""

    def __init__(self, session: object):
        self.session = session
        self.settings = settings
        self.kalman = open_kalman_tracking_data(os.path.join(self.session.base_path, self.session.processed_path))
        self.postprocessObject = open_postprocess_object(self.session)



