'''Utiliy functions for visualise class'''

# set up
import os
from loguru import logger
import cv2
import numpy as np
import dill as pickle

# import
from settings.settings_visualize import defined_settings_visualize as settings

def open_kalman_tracking_data(path):
    try:
        file = os.path.join(path, "kalman_tracking_data.pickle")
        with open(file, "rb") as dill_file:
            kalman = pickle.load(dill_file)
        return kalman

    except FileNotFoundError:
        logger.error(f"Kalman tracking data not found for this session")
        raise FileNotFoundError

def open_tracking_data(session):
    """ 
    This function opens the tracking data and appends it to the postprocessing object.
    
    NOTE: This function consider whether this function is out of place. 
    It is not really the responsibility of the postprocessing class to open the tracking data.
    """
    
    file = os.path.join(session.base_path,session.processed_path, "fully_processed_tracking_data.pickle")
    
    try:
        with open(file, "rb") as dill_file:
            tracking_data = pickle.load(dill_file)
    
    except FileNotFoundError:
        logger.error(f"Tracking data not found for session: {session.name}")
        raise FileNotFoundError
    
    return tracking_data

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