"""Debugging homings or regenerate if changed homing logic

BUG - iF you redo homings escapes are made from homings so you need to redo pp as well"""

from loguru import logger

from databank import experiments_objects
from settings.settings_homings import settings_homings as settings_h
from behave_analysis.process.process import Process
from behave_analysis.homings.homings import get_Homings


def homings():
    """Let's check out some homings and threshold crossings."""
    logger.info("The homings pipeline has started")
    for session_id in experiments_objects:
        logger.info("Loaded a session with the following details: {}".format(session_id))
        session = Process(session_id).load_session()
        get_Homings(settings_h, session, use_boris=True)
    logger.success("Homing pipeline complete")


homings()
