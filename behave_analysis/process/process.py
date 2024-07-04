#Import custom libaries
from settings.settings_process import settings_process as settings_p
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video
from behave_analysis.process.photoresistor import get_Photoresistor
from behave_analysis.process.electrophysiology.ttl_sync import get_TTL
from behave_analysis.process.verify import Verifications
from behave_analysis.process.electrophysiology.load_electrophysiology import LoadEfizz
from behave_analysis.process.electrophysiology.process_electrophysiology import ProcessedEfizz
from behave_analysis.process.session import NEW_Session, get_experiment # Testing refactored dataclass structure
from behave_analysis.database.computer_ID import get_computer_specific_paths


#Import OS libraries
import os
import numpy as np
import dill as pickle
from loguru import logger
import sys
from pathlib import Path

class Process():
    """
    A class that holds the processing part of the data pipeline. It is the first part of the pipeline that should be run.
    This stage also includes verifications of data
    """
    def __init__(self, session_id):
        self.session = get_experiment(session_id) # Retrieve experimental data
        self.sesion_id = session_id
        
    def create_session(self, settings) -> NEW_Session:
        """
        A function that creates the session, and saves the metadata file. It also runs the quality checks on the session.
        Resamples and aligns signals etc. Need to refactor as a lot is happening.
        """
        self.__check_files_exist() # Check that all behavioural files exist
        # Create processed path if it doesn't exist
        if not(os.path.exists(os.path.join(self.session.base_path,self.session.processed_path))): 
            os.makedirs(os.path.join(self.session.base_path,self.session.processed_path))
        
        if settings_p.efizz:
            self.efizzDataLoaded = LoadEfizz(self.session)
            self.ttl = get_TTL(self.session, self.efizzDataLoaded.imec_sync_path)
        
        # Retrieve Dev 3 NIDAQ signals
        self.session.camera_trigger = get_Camera_trigger(self.session, drop_frames = True)[0]
        self.session.audio = get_Audio(self.session)
        self.photo_resistor = get_Photoresistor(self.session)

        if os.path.isfile(os.path.join(self.session.base_path,self.session.metadata_file)) and not settings.create_new_registration: 
            self.load_registration_transform()
        else: 
            logger.info('Registration will be performed!')
            self.loaded_registration_transform = None
        self.session.video = get_Video(self.session, settings, self.loaded_registration_transform)
                 
        if settings_p.efizz:
            _, slope, intercept, lastPulse, firstPulse = self.quality_check_new_sessions()
        elif settings_p.efizz == False:
            self.quality_check_new_sessions()
            
        if settings_p.efizz:
            efizzDataProcessed = ProcessedEfizz(efizzDataLoaded = self.efizzDataLoaded, 
                                                             slope = slope, 
                                                             intercept = intercept,
                                                             samplingRate = self.ttl.sampling_rate,
                                                             filePath = os.path.join(self.session.base_path,self.session.processed_path),
                                                             camera_trigger = self.session.camera_trigger.frame_trigger_onsets_idx,
                                                             lastPulse = lastPulse,
                                                             firstPulse = firstPulse)
        
        logger.info("Finished process, saving session metadata")
        self.save_session()
        
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
            Verifications(self).plot_residuals(show = False)
            
            logger.success("All verifications steps passed")
            return r2_value, slope, intercept, lastPulse, firstPulse
        
        return None

    def save_session(self, overwrite=True) -> None:
        """
        A function that saves the processes session to a metadata file
        """
        meta_file = os.path.join(self.session.base_path,self.session.metadata_file)
        assert not os.path.isfile(meta_file) or overwrite, "Permission to save not granted"
        with open(meta_file, "wb") as dill_file:
            pickle.dump(self.session, dill_file)
        return None

    def load_session(self) -> NEW_Session:
        """
        Load a previously exsisting file. If the file does not exsist and the settings process
        is set to skip process. Then an error may occur
        """
        try:
            with open(os.path.join(self.session.base_path, self.session.metadata_file), "rb") as dill_file: 
                session = pickle.load(dill_file)
                session.base_path, _ = get_computer_specific_paths(session.file_path)

        except FileNotFoundError:
            print(f"Meta data file for path {os.path.join(self.session.base_path, self.session.metadata_file)} not found, aborting script")
            sys.exit()

        except EOFError:
            print(f"The file location is: {self.session.metadata_file}. Is this correct? Does a metadata file exsist here?")
            print("Delete meta file")
            return
        
        except AttributeError:
            print('Attribute Error')
            
        return session

    def __check_files_exist(self) -> None:
        """
        Private method to check that all the files exist for the session.
        In case of user error where the path has been set up incorrectly. And also to check for missing data.
        Run checks that all the behavioural files exist that are needed for processing.
        Check for camera data, frames and analog data. NOTE this was originally introduced
        because a bug where the path root was incorrect and the program wouldn't inform the user.
        """
        full_file_path = Path(os.path.join(self.session.base_path,self.session.file_path))
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

    def load_registration_transform(self) -> None:
        """
        A function that loads the registration transform if it exists, otherwise it sets it to None
        """
        import time
        start = time.time()
        old_sesh = self.load_session()
        if isinstance(old_sesh.video.registration_transform, np.ndarray):
            logger.info(f'Loading existing registration transform! It took:  {time.time()-start}')
            self.loaded_registration_transform = old_sesh.video.registration_transform
            self.session.shelter_location = old_sesh.shelter_location
            if hasattr(old_sesh,'barrier_location'):
                self.session.barrier_location = old_sesh.barrier_location
        else: 
            logger.info('Registration transform does not exist yet')
            self.loaded_registration_transform = None

