from behave_analysis.process.process import Process
from behave_analysis.track.track import Track
from behave_analysis.track.register import Register
from behave_analysis.run import collect_session_IDs
from sample_data.sample_databank import databank
from sample_data.sample_settings.sample_settings_track import settings_track
from sample_data.sample_settings.sample_settings_process import settings_process
import numpy as np
import dill as pickle
import cv2
import os

def test_register():
    selected_session_IDs = collect_session_IDs(settings_process, databank)

    session = Process(selected_session_IDs[0]).load_session()
    video = session.video
    video_object = cv2.VideoCapture(video.video_file)

    registration_transform = Register(session, video, video_object).transform
    assert registration_transform.__class__ == np.ndarray
    assert registration_transform.shape == (2,3)

def test_track():
    selected_session_IDs = collect_session_IDs(settings_process, databank)

    session = Process(selected_session_IDs[0]).load_session()
    tracking = Track(settings_track)
    tracking.tracking_data = {}

    extract_data_from_dlc_file_assertions(tracking, session)
    create_array_with_dlc_tracking_data_assertions(tracking, session)
    replace_low_confidence_points_with_nan_assertions(tracking)
    interpolate_nan_values_assertions(tracking)
    apply_median_filter_assertions(tracking)
    replace_points_far_from_median_bodypart_with_nan_assertions(tracking)
    interpolate_nan_values_assertions(tracking)
    fisheye_correct_tracking_data_assertions(tracking, session)
    register_tracking_data_assertions(tracking, session)
    plot_filtered_and_registered_tracking_assertions(tracking)
    compute_avg_bodypart_locations_assertions(tracking)
    compute_angles_assertions(tracking)
    compute_speed_assertions(tracking, session)
    save_tracking_data_assertions(tracking, session)
    
# -------------------------------------------------------------------------------

def extract_data_from_dlc_file_assertions(tracking, session):
    tracking.extract_data_from_dlc_file(session)
    assert tracking.tracking_data['bodyparts'] == ['nose','L eye','R eye','L ear','neck','R ear','L shoulder','upper back','R shoulder','L hind limb','Lower back','R hind limb','derriere']
    assert tracking.dlc_network_name =='DeepCut_resnet101_BarnesDec7shuffle1_600000'

def create_array_with_dlc_tracking_data_assertions(tracking, session):
    tracking.create_array_with_dlc_tracking_data(session)
    assert tracking.tracking_data_array.shape == (143956, 13, 3)
    assert np.sum(tracking.tracking_data_array) == 1233733422.4502664

def replace_low_confidence_points_with_nan_assertions(tracking):
    tracking.replace_low_confidence_points_with_nan()
    assert np.sum(np.isnan(tracking.tracking_data_array[:,:,:2])>0)
    assert np.sum(np.isnan(tracking.tracking_data_array[:,:,2])==0)

def interpolate_nan_values_assertions(tracking):
    tracking.interpolate_nan_values()
    assert np.sum(np.isnan(tracking.tracking_data_array)==0)

def apply_median_filter_assertions(tracking):
    stdev_before = np.std(tracking.tracking_data_array)
    tracking.apply_median_filter(filter_length=7)
    stdev_after = np.std(tracking.tracking_data_array)
    assert np.sum(tracking.tracking_data_array) == 1229537249.5783331
    assert stdev_after < stdev_before

def replace_points_far_from_median_bodypart_with_nan_assertions(tracking):
    tracking.replace_points_far_from_median_bodypart_with_nan()
    assert np.sum(np.isnan(tracking.tracking_data_array[:,:,:2])>0)
    assert np.sum(np.isnan(tracking.tracking_data_array[:,:,2])==0)

def fisheye_correct_tracking_data_assertions(tracking, session):
    tracking.fisheye_correct_tracking_data(session)
    assert np.sum(tracking.fisheye_corrected_tracking_data_array) == 1307205748

def register_tracking_data_assertions(tracking, session):
    tracking.register_tracking_data(session)
    assert True # use the non-negative coordinates assertion within the method as the test

def plot_filtered_and_registered_tracking_assertions(tracking):
    tracking.plot_filtered_and_registered_tracking()
    assert True # check visually if it looks ok

def compute_avg_bodypart_locations_assertions(tracking):
    tracking.compute_avg_bodypart_locations()
    assert np.sum(tracking.tracking_data['avg_loc']) in [195460552.11858884, 195515748.5779112]

def compute_angles_assertions(tracking):
    tracking.compute_angles()
    assert np.sum(tracking.tracking_data['body_dir']) == -3325857.2246120735

def compute_speed_assertions(tracking, session):
    tracking.compute_speed(session)
    
def save_tracking_data_assertions(tracking, session):
    tracking.save_tracking_data(session)
    assert os.path.isfile(session.video.tracking_data_file)
    with open(session.video.tracking_data_file, "rb") as dill_file: tracking_data_loaded = pickle.load(dill_file)
    assert np.sum(tracking_data_loaded['body_dir']) == -3325857.2246120735
    assert np.sum(tracking_data_loaded['speed']) == 629190.3495970555
