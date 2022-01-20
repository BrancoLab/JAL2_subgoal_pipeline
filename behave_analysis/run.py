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
from databank import databank


def process():
    print("\n------ PROCESSING DATA ------".format(settings_p)); print_settings(settings_p)
    session_IDs = collect_session_IDs(settings_p, databank)
    for session_ID in session_IDs:
        Process(session_ID).create_session(settings_p)

def track():
    print("\n------ TRACKING VIDEOS ------"); print_settings(settings_t)
    session_IDs = collect_session_IDs(settings_t, databank)
    for session_ID in session_IDs:
        session = Process(session_ID).load_session()
        Track(settings_t).run_deeplabcut_tracking(session)
        Track(settings_t).process_tracking_data  (session)

def homings():
    print("\n------ EXTRACTING HOMINGS ------"); print_settings(settings_h)
    session_IDs = collect_session_IDs(settings_h, databank)
    for session_ID in session_IDs:
        session = Process(session_ID).load_session()
        get_Homings(settings_h, session)
        get_Threshold_crossings(settings_h, session)

def visualize():
    print("\n------ VISUALIZING DATA ------"); print_settings(settings_v)
    session_IDs = collect_session_IDs(settings_v, databank)
    for session_ID in session_IDs:
        session = Process(session_ID).load_session()
        if settings_v.laser_trials:  Visualize(session, settings_v).trials(stim_type = 'laser')
        if settings_v.escape_trials: Visualize(session, settings_v).trials(stim_type = 'audio')
        if settings_v.homing_trials: Visualize(session, settings_v).trials(stim_type = 'homing')
        if settings_v.t_xing_trials: Visualize(session, settings_v).trials(stim_type = 'threshold_crossing')

def analyze():
    print("\n------ ANALYZING DATA ------"); print_settings_analysis(settings_a); 
    session_IDs = collect_session_IDs_analysis(settings_a.analysis, databank)
    if settings_a.analysis.plot_escape:  Analyze(session_IDs, settings_a, 'escape trajectories'    ).trajectories()
    if settings_a.analysis.plot_laser:   Analyze(session_IDs, settings_a, 'laser trajectories'     ).trajectories()
    if settings_a.analysis.plot_homings: Analyze(session_IDs, settings_a, 'homing trajectories'    ).trajectories()
    if settings_a.analysis.plot_t_xings: Analyze(session_IDs, settings_a, 't xing trajectories'    ).trajectories()
    if settings_a.analysis.plot_trial:   Analyze(session_IDs, settings_a, 'escape trial trajectory').single_trial()
    if settings_a.analysis.plot_homing:  Analyze(session_IDs, settings_a, 'homing trial trajectory').single_trial()
    if settings_a.analysis.plot_targets: Analyze(session_IDs, settings_a, 'escape targets'         ).distribution()
    if settings_a.analysis.plot_exploration: Analyze(session_IDs, settings_a, 'escape targets'     ).distribution() # TODO: update this line