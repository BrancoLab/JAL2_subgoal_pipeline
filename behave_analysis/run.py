from settings.settings_process import settings_process as settings_p
from settings.settings_track import settings_track as settings_t
from settings.settings_visualize import settings_visualize as settings_v
from settings.settings_analyze import settings_analyze as settings_a
from settings.settings_homings import settings_homings as settings_h
from behave_analysis.process.process import Process
from behave_analysis.track.track import Track
from behave_analysis.homings.homings import get_Homings
from behave_analysis.homings.threshold_crossings import get_Threshold_crossings
from behave_analysis.visualize.visualize import Visualize
from behave_analysis.analyze.analyze import Analyze
from behave_analysis.utils.print_settings import print_settings, print_settings_analysis
from behave_analysis.utils.collect_session_IDs import collect_session_IDs, collect_session_IDs_analysis

# Testing 
from databank import experiments_objects

# OS Libaries
from loguru import logger

def process():
    """
    A function that collects sessions from the databank, puts the sessions through a processing
    pipeline and then saves the sessions to a metadata file. This metadata file is then loaded and used
    by subsequent track, homing, visualize and analyze functions.
    Returns: Nothing, data is saved to a metadata file.
    """
    logger.info("Processing started")
    assert len(experiments_objects) != 0, "Session list should not be empty"
    for session_ID in experiments_objects:
        processObject = Process(session_ID)
        processObject.create_session(settings_p)
    logger.success("Processing complete")
 
def track():
    """
    A function that collects sessions from the databank, puts the sessions through a tracking
    in Deep lab cut.
    """
    logger.info("The tracking pipeline has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        Track(settings_t, session)
    logger.success("Tracking complete")

def homings():
    # TODO: Update to new databank
    session_IDs = collect_session_IDs(settings_h, databank)
    for session_ID in session_IDs:
        session = Process(session_ID).load_session()
        get_Homings(settings_h, session)
        get_Threshold_crossings(settings_h, session)

def visualize():
    """
    A function that visualising the mouse's behaviour in a session by trial, and looks at how well the efizz has synced.
    """
    logger.info("Visualisation started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        if settings_v.laser_trials:  Visualize(session, settings_v).trials(stim_type = 'laser')
        if settings_v.escape_trials: Visualize(session, settings_v).trials(stim_type = 'audio')
        if settings_v.homing_trials: Visualize(session, settings_v).trials(stim_type = 'homing')
        if settings_v.t_xing_trials: Visualize(session, settings_v).trials(stim_type = 'threshold_crossing')
        if settings_v.explore_trial: Visualize(session, settings_v).trials(stim_type = 'audio')
    logger.success("Visualisation complete")

def analyze():
    # print("\n------ ANALYZING DATA ------"); print_settings_analysis(settings_a);
    # TODO: update this to use the new databank 
    session_IDs = collect_session_IDs_analysis(settings_a.analysis, databank)
    if settings_a.analysis.plot_escape:  Analyze(session_IDs, settings_a, 'escape trajectories'    ).trajectories()
    if settings_a.analysis.plot_laser:   Analyze(session_IDs, settings_a, 'laser trajectories'     ).trajectories()
    if settings_a.analysis.plot_homings: Analyze(session_IDs, settings_a, 'homing trajectories'    ).trajectories()
    if settings_a.analysis.plot_t_xings: Analyze(session_IDs, settings_a, 't xing trajectories'    ).trajectories()
    if settings_a.analysis.plot_trial:   Analyze(session_IDs, settings_a, 'escape trial trajectory').single_trial()
    if settings_a.analysis.plot_homing:  Analyze(session_IDs, settings_a, 'homing trial trajectory').single_trial()
    if settings_a.analysis.plot_targets: Analyze(session_IDs, settings_a, 'escape targets'         ).distribution()
    if settings_a.analysis.plot_explore: Analyze(session_IDs, settings_a, 'exploration'            ).exploration() 