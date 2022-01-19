from behave_analysis.homings.homings import get_Homings
from behave_analysis.process.process import Process
from behave_analysis.run import collect_session_IDs
from behave_analysis.run import collect_session_IDs
from sample_data.sample_databank import databank
from sample_data.sample_settings.sample_settings_homings import settings_homings
import numpy as np

def test_homings():
    selected_session_IDs = collect_session_IDs(settings_homings, databank)
    session = Process(selected_session_IDs[0]).load_session()
    get_Homings(settings_homings, session)
    session = Process(selected_session_IDs[0]).load_session()
    assert isinstance(session.homings.onset_frames, np.ndarray)
    assert isinstance(session.homings.stimulus_durations, np.ndarray)
    assert len(session.homings.onset_frames)
    assert len(session.homings.stimulus_durations)