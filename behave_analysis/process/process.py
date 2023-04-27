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
from behave_analysis.process.laser_sync_test import get_dev3_signals

from behave_analysis.process.session import NEW_Session, get_experiment # Testing refactored dataclass structure

from behave_analysis.process.laser_sync_test import plot_laser_sync_test_in_process

#Import OS libraries
import os
import numpy as np
import dill as pickle
from loguru import logger

class Process():
    """
    A class that holds the processing part of the data pipeline. It is the first part of the pipeline that should be run.
    This stage also includes verifications of data
    """
    def __init__(self, session_ID):
        self.session = get_experiment(session_ID) # Session is experiment now
        
    def create_session(self, video_settings) -> NEW_Session:
        """
        A function that creates the session, and saves the metadata file. It also runs the quality checks on the session.
        Resamples and aligns signals etc. Need to refactor as a lot is happening.
        """
        self.load_registration_transform()
        self.print_session_details(stage=1)
        
        # TODO remove this
        # self.session.laser_sync = get_dev3_signals(self.session)
   
        if settings_p.efizz:
            self.session.efizzDataLoaded = LoadEfizz(self.session)
            self.session.ttl = get_TTL(self.session, self.session.efizzDataLoaded.TTL_bin_path)
        
        # Retrieve Dev 3 NIDAQ signals
        self.session.camera_trigger = get_Camera_trigger(self.session, drop_frames = True)[0]
        self.session.audio = get_Audio(self.session)
        self.session.video = get_Video(self.session, video_settings, self.loaded_registration_transform)
        self.session.photo_resistor = get_Photoresistor(self.session)
                
        self.print_session_details(stage=2)
        
        if settings_p.efizz:
            _, slope, intercept, lastPulse = self.quality_check_new_sessions()
        elif settings_p.efizz == False:
            self.quality_check_new_sessions()
            
        if settings_p.efizz:
            self.session.efizzDataProcessed = ProcessedEfizz(efizzDataLoaded = self.session.efizzDataLoaded, 
                                                             slope = slope, 
                                                             intercept = intercept,
                                                             samplingRate = self.session.ttl.sampling_rate,
                                                             filePath = self.session.file_path,
                                                             camera_trigger = self.session.camera_trigger.frame_trigger_onsets_idx,
                                                             lastPulse = lastPulse)
            
        self.save_session()
        
        # TODO remove this
        # plot_laser_sync_test_in_process(laser_signal = self.session.laser_sync.red_Laser_Signal,
        #                                 laser_onsets = self.session.laser_sync.laser_onsets)
        
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
            (r2_value, slope, intercept), lastPulse = Verifications(self).visulize_sync_output()
            Verifications(self).verify_clock_drift(r2_value)
            Verifications(self).plot_residuals(show = False)
            
            logger.success("All verifications steps passed")
            return r2_value, slope, intercept, lastPulse
        
        return None

    def save_session(self, overwrite=True) -> None:
        """
        A function that saves the processes session to a metadata file
        """
        assert not os.path.isfile(self.session.metadata_file) or overwrite, "Permission to save not granted"
        with open(self.session.metadata_file, "wb") as dill_file: pickle.dump(self.session, dill_file)
        return None

    def load_session(self) -> NEW_Session:
        """
        Load a previously exsisting file. If the file does not exsist and the settings process
        is set to skip process. Then an error may occur
        """
        try:
            with open(self.session.metadata_file, "rb") as dill_file: 
                session = pickle.load(dill_file)

        except EOFError:
            print(f"The file location is: {self.session.metadata_file}. Is this correct? Does a metadata file exsist here?")
            print("Delete meta file")
            return
        
        except AttributeError:
            print('poop') # Laurence thought this was good code, have lunch if you don't think that's funny
        return session

    def load_registration_transform(self) -> None:
        """
        A function that loads the registration transform if it exists, otherwise it sets it to None
        """
        if os.path.isfile(self.session.metadata_file) and isinstance(self.load_session().video.registration_transform, np.ndarray):
            self.loaded_registration_transform = self.load_session().video.registration_transform
        else: 
            self.loaded_registration_transform = None
        return None

    def print_session_details(self,stage: int) -> None:
        if stage==1:
            logger.info("Commencing processing of sessions")
            for key in self.session.__dict__.keys():
                if key in ['name','number','mouse','previous_sessions']:
                    logger.info(" {}: {}".format(key, self.session.__dict__[key]))
        if stage==2:
            for key in self.session.__dict__.keys():
                if key in ['camera_trigger', 'laser','audio','video']:
                    logger.info(" {} metadata saved".format(key))
            logger.info(" registration transform: {}".format(isinstance(self.session.video.registration_transform, np.ndarray)))
        return None