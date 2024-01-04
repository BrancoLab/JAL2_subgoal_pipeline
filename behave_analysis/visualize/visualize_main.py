import os

from loguru import logger
import cv2
import numpy as np
import dill as pickle

from settings.settings_visualize import defined_settings_visualize as settings

class Visualize:
    """Visualise escape trials AND open postprocess object

    #TODO: Make this class have one responsbiility"""

    def __init__(self, session: object):
        self.session = session
        self.settings = settings
        self.kalman = open_kalman_tracking_data(os.path.join(self.session.base_path, self.session.processed_path))
        self.postprocessObject = open_postprocess_object(self.session)
# ------------------------------------------------------------------Utilities ----------------------------------------------------------------

# TODO: move to new script
# Utiliy functions for visualise class
def open_kalman_tracking_data(path):
    try:
        file = os.path.join(path, "kalman_tracking_data.pickle")
        with open(file, "rb") as dill_file:
            kalman = pickle.load(dill_file)
        return kalman

    except FileNotFoundError:
        logger.error(f"Kalman tracking data not found for this session")
        raise FileNotFoundError


def open_postprocess_object(session) -> object:
    try:
        fileObj = open(
            os.path.join(session.base_path, session.processed_path)
            + "\\"
            + "postprocessclass"
            + "_"
            + str(settings.cluster_type),
            "rb",
        )
        postprocessObject = pickle.load(fileObj)
        fileObj.close()
        return postprocessObject

    except FileNotFoundError:
        logger.error(
            f"Data not found for session: {session.name} - Check databank and whether you have actually run this configuration of postprocess. "
        )
        raise FileNotFoundError
