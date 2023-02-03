#Import custom libaries
from settings.settings_process import settings_process as settings_p
from behave_analysis.process.session import get_Session
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video
from behave_analysis.process.photoresistor import get_Photoresistor
from behave_analysis.process.ephys import get_Ephys
from behave_analysis.process.ttl_sync import get_TTL
from behave_analysis.process.verify import Verifications

#Import OS libraries
import os
import numpy as np
import dill as pickle
from loguru import logger

class Process():
    def __init__(self, session_ID):
        self.session = get_Session(session_ID)
        
    def create_session(self, video_settings):
        """A function that creates the session, and saves the metadata file. It also runs the quality checks on the session.
        Resamples and aligns signals etc. Need to refactor as a lot is happening.
        
        Refactor potential: Remove downsampling from get functions into seperate function
        """
        
        self.load_registration_transform()
        self.print_session_details(stage=1)
        
        indexs = None # Prevents error if not efizz
        if settings_p.efizz:
            self.session.ttl = get_TTL(self.session)
            indexs = self.session.ttl.choose_index
        
        self.session.camera_trigger = get_Camera_trigger(self.session, 
                                                         indexs_to_remove = indexs, 
                                                         down_sample = settings_p.efizz,
                                                         drop_frames = True)[0]
        
        self.session.audio = get_Audio(self.session, 
                                       indexs_to_remove = indexs, 
                                       down_sample = settings_p.efizz)
        
        self.session.video = get_Video(self.session, 
                                       video_settings, 
                                       self.loaded_registration_transform)
        
        self.session.photo_resistor = get_Photoresistor(self.session, 
                                                        indexs_to_remove = indexs, 
                                                        down_sample = settings_p.efizz)
        
        if settings_p.efizz:
            pass
            # self.session.ephys = get_Ephys(self.session)
            
        self.print_session_details(stage=2)
        self.save_session()
        self.quality_check_new_sessions()
            
        return self.session
    
    def quality_check_new_sessions(self) -> None:
        """A function that runs veritifcation checks on a new session that has been recorded but not processed.
        Checks include:
        + Video frame counts are as expected
        + The bonsai machine matches with spike GLX if doing efizz"""

        Verifications(self).verify_all_frames_saved()
        
        if settings_p.efizz:
            Verifications(self).verify_check_for_abberant_signals_in_bonsai()
            Verifications(self).verify_aligned_data_streams()
            Verifications(self).verify_check_means()
            Verifications(self).verify_onsets_and_offsets()
            Verifications(self).verify_ttl_len_with_frame_duration()
            Verifications(self).visulize_sync_output() # Comment out to prevent sync plot from showing, just a sanity check
            Verifications(self).verify_clock_drift() 
        
        logger.info("All verifications steps passed")
        
        return None 

    def save_session(self, overwrite=True):
        """A function that saves the processes session to a metadata file

        Args:
            overwrite (bool, optional): _description_. Defaults to True.
        """
        assert not os.path.isfile(self.session.metadata_file) or overwrite, "Permission to save not granted"
        with open(self.session.metadata_file, "wb") as dill_file: pickle.dump(self.session, dill_file)

    def load_session(self):
        """Load a previously exsisting file. If the file does not exsist and the settings process
        is set to skip process. Then an error may occur.

        Returns:
            _type_: _description_
        """
        try:
            with open(self.session.metadata_file, "rb") as dill_file: session = pickle.load(dill_file)
        except EOFError:
            print(f"The file location is: {self.session.metadata_file}. Is this correct? Does a metadata file exsist here?")
            print("Delete meta file")
            return

        return session

    def load_registration_transform(self):
        """A function that loads the registration transform if it exists, otherwise it sets it to None"""
        if os.path.isfile(self.session.metadata_file) and isinstance(self.load_session().video.registration_transform, np.ndarray):
            self.loaded_registration_transform = self.load_session().video.registration_transform
        else: 
            self.loaded_registration_transform = None

    def print_session_details(self,stage: int):
        if stage==1:
            print('\n\n---')
            for key in self.session.__dict__.keys():
                if key in ['name','number','mouse','previous_sessions']:
                    print(" {}: {}".format(key, self.session.__dict__[key]))
        if stage==2:
            print('')
            for key in self.session.__dict__.keys():
                if key in ['camera_trigger', 'laser','audio','video']:
                    print(" {} metadata saved".format(key))
            print(" registration transform: {}".format(isinstance(self.session.video.registration_transform, np.ndarray)))

        