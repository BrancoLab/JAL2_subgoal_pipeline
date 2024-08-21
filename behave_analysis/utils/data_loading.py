"""This script contains lots of useful data loading function"""
import dill as pickle
import os
from loguru import logger

from behave_analysis.homings.homings import get_Homings
from settings.settings_homings import settings_homings as settings_h


def load_or_extract_homings(session, video_df):
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
    homie_path = os.path.join(session.base_path, session.processed_path, "homings", "homings_obj.pkl")
    if os.path.exists(homie_path):
        logger.info("Homings object found. Loading...")
        with open(homie_path, "rb") as dill_file:
            homings = pickle.load(dill_file)
    else:
        logger.info("Homings object not found. Extracting homings now...")
        homings_obj = get_Homings(settings_h, session, video_df)
        homings = homings_obj.session.homing
    return homings
