import os
import dill as pickle
# Custom libs

from settings.settings_process import settings_process as settings_p
from settings.settings_track import settings_track as settings_t
from settings.settings_visualize import Settings_visualize as settings_v
from settings.settings_analyze import settings_analyze as settings_a
from settings.settings_analyze_efizz import Settings_analyze_efizz as settings_a_e
from settings.settings_homings import settings_homings as settings_h
from behave_analysis.process.process import Process
from behave_analysis.track.track import Track
from behave_analysis.homings.homings import get_Homings
from behave_analysis.homings.threshold_crossings import get_Threshold_crossings
from behave_analysis.visualize.visualize_main import Visualize
from behave_analysis.visualize.visualize_efizz import Visualize_efizz
from behave_analysis.visualize.visualize_behave import Visualize_behave
from behave_analysis.analyze.analyze import Analyze
from behave_analysis.utils.print_settings import print_settings, print_settings_analysis
from behave_analysis.utils.collect_session_IDs import collect_session_IDs, collect_session_IDs_analysis
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from behave_analysis.analyze.AnalyzeBehave import AnalyzeBehave
from databank import experiments_objects
from behave_analysis.postprocess.pp_main import Postprocessor

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
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        get_Homings(settings_h, session)
        get_Threshold_crossings(settings_h, session)
        
def postprocess():
    """ 
    A function that outputs and saves a postprocessed object as a pickle file in the processed data folder.
    """
    logger.info("The post processing of the data has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        Postprocessor(session)
    logger.success("The post processing of the data has finished and the postprocessed object has been saved to a pickle file")


def visualize():
    """
    A function that visualising the mouse's behaviour in a session by trial, and looks at how well the efizz has synced.
    """
    logger.info("Visualisation started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        visual_object = Visualize(session)
        Visualize_behave(session,visual_object.postprocessObject).plot_behavioral_stats()
        if settings_v.stim_type != '':
            visual_object.trial_movies(settings_v.stim_type)
            Visualize_behave(session,visual_object.postprocessObject).escape_plotting()
        if settings_v.efizz:
            Visualize_efizz(visual_object.postprocessObject, session).run_tuning_functions()
            if settings_v.stim_type != '':
                Visualize_efizz(visual_object.postprocessObject, session).run_stim_resp_plotting()
    logger.success("Visualisation complete")

def analyze():
    """
    A function that calls all the analysis modules and is designed to be run last and for the whole dataset.
    """
    logger.info("The analysis pipeline has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        AnalyzeBehave(session)
        if settings_a.efizz:
            AnalyzeEfizz(session)
    
    logger.success("Analysis pipeline complete")
        
    # # print("\n------ ANALYZING DATA ------"); print_settings_analysis(settings_a);
    # # TODO: update this to use the new databank 
    # session_IDs = collect_session_IDs_analysis(settings_a.analysis, databank)
    # if settings_a.analysis.plot_escape:  Analyze(session_IDs, settings_a, 'escape trajectories'    ).trajectories()
    # if settings_a.analysis.plot_laser:   Analyze(session_IDs, settings_a, 'laser trajectories'     ).trajectories()
    # if settings_a.analysis.plot_homings: Analyze(session_IDs, settings_a, 'homing trajectories'    ).trajectories()
    # if settings_a.analysis.plot_t_xings: Analyze(session_IDs, settings_a, 't xing trajectories'    ).trajectories()
    # if settings_a.analysis.plot_trial:   Analyze(session_IDs, settings_a, 'escape trial trajectory').single_trial()
    # if settings_a.analysis.plot_homing:  Analyze(session_IDs, settings_a, 'homing trial trajectory').single_trial()
    # if settings_a.analysis.plot_targets: Analyze(session_IDs, settings_a, 'escape targets'         ).distribution()
    # if settings_a.analysis.plot_explore: Analyze(session_IDs, settings_a, 'exploration'            ).exploration() 