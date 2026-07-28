"""This script contains lots of useful data loading function"""

import dill as pickle
import os
from loguru import logger
import numpy as np

from behave_analysis.analyze.behaviour.homings_escapes.homings import get_Homings
from settings.settings_overrides import settings_overrides
from settings.settings_analyze_behave import settings_ab as settings
from behave_analysis.analyze.behaviour.homings_escapes.escapes import get_Escapes

def load_or_extract_homings(session):
    """Check if homings object pickle is saved, if not extract homings
    Return the homing data object for a given session.

    This function retrieves homing data, which typically includes information about specific behavioral trials,
    from a pre-processed pickle file. The file's location is determined based on the session's base path and
    processed path attributes. The function asserts the existence of the file before attempting to load it,
    ensuring that the necessary data is available and has been generated prior to this function call.

    Parameters:
    - session (SessionType): An object representing the session, which should contain attributes `base_path`
      and `processed_path` used to construct the file path to the homing data.

    Returns:
    - object: The homing data object loaded from the pickle file.

    Raises:
    - AssertionError: If the homing data file does not exist at the expected path.
    """
    logger.warning("This homing loading function is deprecated! Use getHomings class with redo_compute set to False!")
    # homie_path = os.path.join(session.base_path, session.processed_path, "homings", "homings_obj.pkl")
    # if np.logical_and(os.path.exists(homie_path), not(settings.redo_homings)):
    #     logger.info("Homings object found. Loading...")
    #     with open(homie_path, "rb") as dill_file:
    #         homings = pickle.load(dill_file)
    # else:
    #     logger.info("Homings object not found. Extracting homings now...")
    #     homings_obj = get_Homings(settings=settings, session=session)
    # return homings_obj


def load_or_extract_escapes(session):
    """Check if homings object pickle is saved, if not extract homings
    Return the homing data object for a given session.

    This function retrieves homing data, which typically includes information about specific behavioral trials,
    from a pre-processed pickle file. The file's location is determined based on the session's base path and
    processed path attributes. The function asserts the existence of the file before attempting to load it,
    ensuring that the necessary data is available and has been generated prior to this function call.

    Parameters:
    - session (SessionType): An object representing the session, which should contain attributes `base_path`
      and `processed_path` used to construct the file path to the homing data.

    Returns:
    - object: The homing data object loaded from the pickle file.

    Raises:
    - AssertionError: If the homing data file does not exist at the expected path.
    """
    esc_path = os.path.join(session.base_path, session.processed_path, "escapes", "escapes.npy")
    if os.path.exists(esc_path):
        logger.info("Escape dict found. Loading...")
        escapes = np.load(esc_path, allow_pickle=True).item()
    else:
        logger.info("Escape dict not found. Extracting escapes now...")
        settings_ab = settings_overrides(settings_ab, {"redo_compute": False})
        homings = get_Homings({**settings_ab, "homings_curated": True}, session).get_homings()
        escapes = get_Escapes(settings, session, tracking_data = [], video_df = [], homings = homings)
    return escapes
