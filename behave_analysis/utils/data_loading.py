"""This script contains lots of useful data loading function"""
import dill as pickle
import os
from loguru import logger

from behave_analysis.homings.homings import get_Homings
from settings.settings_homings import settings_homings as settings_h


def load_or_extract_homings(session):
    """Check if homings object pickle is saved, if not extract homings"""
    homie_path = os.path.join(session.base_path, session.processed_path, "homings", "homings_obj.pkl")
    if os.path.exists(homie_path):
        logger.info("Homings object found. Loading...")
        with open(homie_path, "rb") as dill_file:
            homings = pickle.load(dill_file)
    else:
        homings = get_Homings(settings_h, session)
    return homings
