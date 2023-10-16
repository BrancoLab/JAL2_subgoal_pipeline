from settings.settings_visualize import defined_settings_visualize as settings_v
from loguru import logger
from behave_analysis.process.process import Process
from databank import experiments_objects
from behave_analysis.visualize.visualize_main import Visualize

def visualize():
    """
    A function that visualising the mouse's behaviour in a session by trial, and looks at how well the efizz has synced.
    """
    
    logger.info("Visualisation started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        if settings_v.laser_trials:
            Visualize(session).trials(stim_type="laser")
        if settings_v.escape_trials:
            Visualize(session).trials(stim_type="audio")
        if settings_v.homing_trials:
            Visualize(session).trials(stim_type="homing")
        if settings_v.t_xing_trials:
            Visualize(session).trials(stim_type="threshold_crossing")
        if settings_v.explore_trial:
            Visualize(session).trials(stim_type="audio")
    logger.success("Visualisation complete")

visualize()
