"""Debugging homings or regenerate if changed homing logic

BUG - iF you redo homings escapes are made from homings so you need to redo pp as well"""

from loguru import logger

from databank import experiments_objects
from settings.settings_homings import settings_homings as settings_h
from behave_analysis.process.process import Process
from behave_analysis.homings.homings import get_Homings
from behave_analysis.postprocess.trials.escapes import get_Escapes
from settings.settings_postprocess import defined_settings_postprocess as settings


def homings():
    """Let's check out some homings and threshold crossings."""
    logger.info("The homings pipeline has started")
    for session_id in experiments_objects:
        logger.info("Loaded a session with the following details: {}".format(session_id))
        session = Process(session_id).load_session()
        homings_obj = get_Homings(settings=settings_h, session=session)
        escapes = get_Escapes(settings, session, tracking_data = [], video_df = [], homings = homings_obj.session.homing)
    logger.success("Homing pipeline complete")

homings()
