# Import custom libaries
from behave_analysis.utils.creating_directories import make_directory
from settings.settings_process import settings_process as settings_p
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video, Video
from behave_analysis.process.photoresistor import get_Photoresistor
from behave_analysis.process.electrophysiology.ttl_sync import get_TTL
from behave_analysis.process.verify import Verifications
from behave_analysis.process.electrophysiology.load_electrophysiology import LoadEfizz
from behave_analysis.process.electrophysiology.process_electrophysiology import ProcessedEfizz
from behave_analysis.process.session import NEW_Session, get_experiment
from behave_analysis.database.computer_ID import get_computer_specific_paths
from behave_analysis.utils.AI_dataClass_objects import Audio

# Import OS libraries
import json
import os
import numpy as np
import dill as pickle
from loguru import logger
import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass


class Process:
    """
    A class that holds the processing part of the data pipeline. It is the first part of the pipeline that should be run.
    This stage also includes verifications of data
    """

    def __init__(self, session_id):
        self.session = get_experiment(session_id)  # Retrieve experimental data
        self.sesion_id = session_id
        self.processed_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path))

    def _session_metadata_path(self) -> str:
        return os.path.join(self.processed_path, "metadata.json")

    @staticmethod
    def _json_compatible(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        if is_dataclass(data):
            return {key: Process._json_compatible(value) for key, value in asdict(data).items()}
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if isinstance(data, list):
            return [Process._json_compatible(item) for item in data]
        if isinstance(data, tuple):
            return [Process._json_compatible(item) for item in data]
        if isinstance(data, dict):
            return {key: Process._json_compatible(value) for key, value in data.items()}
        return data

    def _serialize_session_payload(self, session: NEW_Session) -> dict:
        """Serialize only fields that are not already reconstructed by get_experiment."""
        return {
            "shelter_location": session.shelter_location,
            "barrier_location": session.barrier_location,
            "audio": session.audio,
            "video": session.video,
        }

    def _apply_session_payload(self, session: NEW_Session, payload: dict) -> None:
        """Apply JSON metadata fields onto a freshly reconstructed session object."""
        session.shelter_location = payload.get("shelter_location")
        session.barrier_location = payload.get("barrier_location")

        audio_payload = payload.get("audio")
        if isinstance(audio_payload, dict):
            session.audio = Audio(
                num_samples = int(audio_payload.get("num_samples", 0)),
                onset_frames = np.asarray(audio_payload.get("onset_frames", [])),
                stimulus_durations = np.asarray(audio_payload.get("stimulus_durations", [])),
            )

        video_payload = payload.get("video")
        if isinstance(video_payload, dict):
            registration_size = video_payload.get("registration_size")
            session.video = Video(
                num_frames = int(video_payload.get("num_frames", 0)),
                camFilePath = video_payload.get("camFilePath"),
                fps = int(video_payload.get("fps", 0)),
                height = int(video_payload.get("height", 0)),
                width = int(video_payload.get("width", 0)),
                fisheye_correction_file = video_payload.get("fisheye_correction_file"),
                registration_type = video_payload.get("registration_type"),
                registration_size = tuple(registration_size) if registration_size is not None else tuple(),
                pixels_per_cm = int(video_payload.get("pixels_per_cm", 0)),
                radius = int(video_payload.get("radius", 0)),
                x_offset = int(video_payload.get("x_offset", 128)),
                y_offset = int(video_payload.get("y_offset", 0)),
            )

        return session

    def create_session(self, settings) -> NEW_Session:
        """
        A function that creates the session, and saves the metadata file. It also runs the quality checks on the session.
        Resamples and aligns signals etc. Need to refactor as a lot is happening.
        """
        self.__check_files_exist()  # Check that all behavioural files exist

        if settings_p.efizz:
            self.efizzDataLoaded = LoadEfizz(self.session, settings).select_and_load_efizz_files()
            self.ttl = get_TTL(self.session, self.efizzDataLoaded.imec_sync_path)

        # Retrieve Dev 3 NIDAQ signals
        self.camera_trigger = get_Camera_trigger(self.session, drop_frames=True)[0]
        self.session.audio = get_Audio(self.session)
        self.photo_resistor = get_Photoresistor(self.session)

        # load or perform registration transform
        self.loaded_registration_transform = None
        if not settings.create_new_registration:
            self.loaded_registration_transform, self.session.shelter_location, self.session.barrier_location = self.load_registration_transform()

        self.session.video, registration_transform = get_Video(self.session, settings, self.loaded_registration_transform)
        self.save_registration_transform(registration_transform, self.session)

        # quality check the session
        if settings_p.efizz:
            _, slope, intercept, lastPulse, firstPulse = self.quality_check_new_sessions()
        elif settings_p.efizz == False:
            self.quality_check_new_sessions()

        logger.info("Saving session metadata - building polars df next")
        self.save_session(self.session)

        if settings_p.efizz:
            ProcessedEfizz(
                efizzDataLoaded=self.efizzDataLoaded,
                slope=slope,
                intercept=intercept,
                samplingRate=self.ttl.sampling_rate,
                filePath=self.processed_path,
                camera_trigger=self.camera_trigger.frame_trigger_onsets_idx,
                lastPulse=lastPulse,
                firstPulse=firstPulse,
            )

        return self.session

    def quality_check_new_sessions(self) -> tuple:
        """
        A function that runs veritifcation checks on a new session that has been recorded but not processed
        """
        Verifications(self).verify_all_frames_saved()

        if settings_p.efizz:
            Verifications(self).verify_check_for_abberant_signals_in_bonsai()
            Verifications(self).verify_aligned_data_streams()
            Verifications(self).verify_check_means()
            Verifications(self).verify_onsets_and_offsets()
            Verifications(self).verify_ttl_len_with_frame_duration()
            (r2_value, slope, intercept), lastPulse, firstPulse = Verifications(self).visulize_sync_output()
            Verifications(self).verify_clock_drift(r2_value)
            Verifications(self).plot_residuals(show=False)

            logger.success("All verifications steps passed")
            return r2_value, slope, intercept, lastPulse, firstPulse

        return None

    def save_session(self, session, overwrite=True) -> None:
        """
        A function that saves the processes session to a metadata file
        """
        meta_file = self._session_metadata_path()
        assert not os.path.isfile(meta_file) or overwrite, "Permission to save not granted"
        session_dict = self._json_compatible(self._serialize_session_payload(session))
        os.makedirs(os.path.dirname(meta_file), exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as json_file:
            json.dump(session_dict, json_file)
        return None

    def load_session(self) -> NEW_Session:
        """
        Load a previously exsisting file. If the file does not exsist and the settings process
        is set to skip process. Then an error may occur
        """
        try:
            # if session dict json does not exist,load legacy pickled metadata and migrate to JSON
            meta_json_path = self._session_metadata_path()
            if not os.path.isfile(meta_json_path):
                # Legacy pickle migration path.
                legacy_meta_path = os.path.join(self.processed_path, "metadata")
                with open(legacy_meta_path, "rb") as dill_file:
                    legacy_session = pickle.load(dill_file)

                legacy_transform = getattr(getattr(legacy_session, "video", None), "registration_transform", None)
                if isinstance(legacy_transform, np.ndarray):
                    self.save_registration_transform(legacy_transform, session=legacy_session)

                # turn legacy session obj into a dict
                legacy_payload = self._json_compatible(self._serialize_session_payload(legacy_session))
                # use the dict to fill a new session object
                legacy_session = self._apply_session_payload(self.session, legacy_payload)

                # save the session object as a JSON dict
                self.save_session(session=legacy_session, overwrite=True)

                # delete legacy files
                # os.unlink(os.path.join(self.processed_path, "metadata.pkl"))
                # os.unlink(os.path.join(self.processed_path, "photoresistor.pkl"))
                # os.unlink(os.path.join(self.processed_path, "TTL_file.pkl"))

            # merge session and session dict json
            session = asdict(self.session)
            with open(meta_json_path, "r", encoding="utf-8") as json_file:
                session_dict = json.load(json_file)
            for key, val in session_dict.items():
                session[key] = val

        except FileNotFoundError:
            print(f"Meta data file for path {self._session_metadata_path(get_experiment(self.sesion_id))} not found, aborting script")
            sys.exit()

        return session

    def __check_files_exist(self) -> None:
        """
        Private method to check that all the files exist for the session.
        In case of user error where the path has been set up incorrectly. And also to check for missing data.
        Run checks that all the behavioural files exist that are needed for processing.
        Check for camera data, frames and analog data. NOTE this was originally introduced
        because a bug where the path root was incorrect and the program wouldn't inform the user.
        """
        full_file_path = Path(os.path.join(self.session.base_path, self.session.file_path))
        datapath_parts = (full_file_path / full_file_path.name).parts

        camlastpath = datapath_parts[-1] + "_cam.avi"
        cam_path = Path(*datapath_parts[:-1]) / camlastpath
        assert os.path.exists(cam_path), "No camera data found. Check file path"

        frameslastpath = datapath_parts[-1] + "_frames.csv"
        frames_path = Path(*datapath_parts[:-1]) / frameslastpath
        assert os.path.exists(frames_path), "No frames data found. Check file path"

        analoglastpath = datapath_parts[-1] + "_analog.bin"
        analog_path = Path(*datapath_parts[:-1]) / analoglastpath
        assert os.path.exists(analog_path), "No analog data found. Check file path"

    def save_registration_transform(self, registration_transform: object, session: NEW_Session = None) -> None:
        """
        Save registration transform and manual arena locations to a separate JSON sidecar.
        """
        reg_dict = {
            "registration_transform": registration_transform,
            "shelter_location": session.shelter_location,
            "barrier_location": session.barrier_location,
        }

        os.makedirs(os.path.dirname(self._registration_metadata_path()), exist_ok=True)
        with open(self._registration_metadata_path(), "w", encoding="utf-8") as f:
            json.dump(self._json_compatible(reg_dict), f)

    def _registration_metadata_path(self) -> str:

        return os.path.join(self.processed_path, "registration_data.json")
    
    def load_registration_transform(self):
        """
        Load registration sidecar and return transform + manual arena locations.
        Returns none if failed to load registration
        """

        registration_path = self._registration_metadata_path()
        if not os.path.isfile(registration_path):
            # Backward compatibility: try loading from legacy pickled metadata.
            legacy_metadata_path = os.path.join(self.processed_path, "metadata")
            if os.path.isfile(legacy_metadata_path):
                try:
                    with open(legacy_metadata_path, "rb") as dill_file:
                        legacy_session = pickle.load(dill_file)

                    legacy_transform = getattr(getattr(legacy_session, "video", None), "registration_transform", None)
                    legacy_shelter_location = getattr(legacy_session, "shelter_location")
                    legacy_barrier_location = getattr(legacy_session, "barrier_location")

                    if isinstance(legacy_transform, np.ndarray):
                        logger.info("Loaded registration transform from legacy metadata file")
                        return legacy_transform, legacy_shelter_location, legacy_barrier_location
                except Exception as e:
                    logger.warning(f"Legacy metadata exists but could not be loaded for registration fallback: {e}")

            logger.info("Registration transform file does not exist yet")
            return None, None, None

        with open(registration_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        transform = payload.get("registration_transform")
        loaded_registration_transform = np.array(transform) if transform is not None else None
        shelter_location = payload.get("shelter_location")
        barrier_location = payload.get("barrier_location")

        logger.info("Loaded registration transform and locations from file JSON")
        return loaded_registration_transform, shelter_location, barrier_location
