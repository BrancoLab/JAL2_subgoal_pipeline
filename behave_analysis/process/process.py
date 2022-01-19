from behave_analysis.process.session import Session, get_Session
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video
import os
import numpy as np
import dill as pickle

class Process():
    def __init__(self, session_ID):
        self.session = get_Session(session_ID)

    def create_session(self, settings) -> Session:        
        self.load_registration_transform()
        self.print_session_details(stage=1)
        self.session.camera_trigger = get_Camera_trigger(self.session)
        self.session.audio          = get_Audio(self.session)
        self.session.video          = get_Video(self.session, settings, self.loaded_registration_transform)
        self.print_session_details(stage=2)
        self.save_session()
        self.verify_all_frames_saved()
        self.verify_aligned_data_streams()
        return self.session

    def save_session(self, overwrite=True):
        assert not os.path.isfile(self.session.metadata_file) or overwrite, "Permission to save not granted"
        with open(self.session.metadata_file, "wb") as dill_file: pickle.dump(self.session, dill_file)

    def load_session(self) -> Session:
        with open(self.session.metadata_file, "rb") as dill_file: session = pickle.load(dill_file)
        return session

    def load_registration_transform(self) -> object:
        if os.path.isfile(self.session.metadata_file) and isinstance(self.load_session().video.registration_transform, np.ndarray):
            self.loaded_registration_transform = self.load_session().video.registration_transform
        else: self.loaded_registration_transform = None

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

    def verify_all_frames_saved(self):
        if self.session.camera_trigger.num_frames != self.session.video.num_frames:
            print("\n - Video contains {} frames, but {} frames were triggered! (for experiment: {}, mouse: {})---".format(self.session.video.num_frames, self.session.camera_trigger.num_frames, self.session.experiment, self.session.mouse))

    def verify_aligned_data_streams(self) -> None:
        if self.session.camera_trigger.num_samples != self.session.audio.num_samples:
            print("\n - Data streams have mismatched numbers of samples---\n  Camera trigger: {}\n  Audio input:    {}\n  Laser output:   {} + {} or {} or {} or {}".format(self.session.camera_trigger.num_samples, self.session.audio.num_samples, self.session.laser.num_samples, known_offset[0], known_offset[1], known_offset[2], known_offset[3]))
