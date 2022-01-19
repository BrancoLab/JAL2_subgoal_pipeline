from behave_analysis.process.process import Process
from behave_analysis.run import collect_session_IDs
from sample_data.sample_settings.sample_settings_process import settings_process
from sample_data.sample_databank import databank
import numpy as np
import cv2

def test_process():
    selected_session_IDs = collect_session_IDs(settings_process, databank)
    collect_session_IDs_assertions(selected_session_IDs)
    session = Process(selected_session_IDs[0]).create_session(settings_process)
    session_assertions(session)
    camera_trigger_assertions(session.camera_trigger)
    laser_assertions(session.laser)
    audio_assertions(session.audio)
    video_assertions(session.video)

    session_loaded = Process(selected_session_IDs[0]).load_session()
    load_session_assertions(session, session_loaded)

def collect_session_IDs_assertions(selected_session_IDs: object):
    assert selected_session_IDs.shape == (2, 5)
    assert selected_session_IDs[0,0]==0
    assert np.all(selected_session_IDs[:,2]=="block edge vectors")
    assert selected_session_IDs[0, 4] == ".\\sample_data\\21MAR16_9718_block evs"

def session_assertions(session):
    assert session.mouse == '9718'
    assert session.experiment == 'block edge vectors'
    assert session.daq_sampling_rate == 15000
    assert session.previous_sessions == 0
    assert session.file_path == ".\\sample_data\\21MAR16_9718_block evs"
    assert session.metadata_file == ".\\sample_data\\21MAR16_9718_block evs\\metadata"

def camera_trigger_assertions(camera_trigger):
    assert camera_trigger.num_samples == 54010500
    assert camera_trigger.num_frames == 143956
    assert camera_trigger.fps == 40

def laser_assertions(laser):
    onset_frames = np.array([np.array([2692, 2828]), np.array([4996, 5120, 5248])], dtype=object)
    stimulus_durations = np.array([np.array([2., 2.]), np.array([2., 2., 2.])], dtype=object)   

    assert laser.num_samples == 54004500
    assert np.all([np.all(l) for l in [x == y for x, y in zip(laser.onset_frames, onset_frames)]])
    assert np.all([np.all(l) for l in [x == y for x, y in zip(laser.stimulus_durations, stimulus_durations)]])
    assert np.all(laser.frequency == np.array([20., 20.]))

def audio_assertions(audio):
    assert audio.num_samples == 54010500
    assert np.all(audio.onset_frames == np.array([ [54473], [97113], [103703], [113945]]))
    assert np.all(audio.stimulus_durations == np.array([[4.5], [4.5] , [4.5], [3.] ]))

def video_assertions(video):
    assert video.num_frames == 143956
    assert video.fps == 40
    assert video.height == 1024
    assert video.width == 1024
    video_object = cv2.VideoCapture(video.video_file)
    video_object.set(cv2.CAP_PROP_POS_FRAMES, 0)
    successful_read, _ = video_object.read()
    assert successful_read

def load_session_assertions(created_session, loaded_session):
    equal_dict_contents_assertions(created_session.__dict__, loaded_session.__dict__)

def equal_dict_contents_assertions(dict1, dict2):
    for entry1, entry2 in zip(dict1, dict2):
        data1 = dict1[entry1]
        data2 = dict2[entry2]
        if 'behave_analysis' in str(data1.__class__):
            equal_dict_contents_assertions(data1.__dict__, data2.__dict__)
        elif isinstance(data1, (list, tuple, np.ndarray)) and isinstance(data1[0], (list, tuple, np.ndarray)):
            assert np.concatenate([[a==b for a, b in zip(x, y)] for x, y in zip(data1, data2)]).all()
        else:
            assert (np.array(data1==data2)).all()
