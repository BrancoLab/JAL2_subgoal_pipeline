#Import custom libaries
from behave_analysis.process.session import Session, get_Session
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video
from behave_analysis.process.ttl_sync import get_TTL
from behave_analysis.utils.check_drop_frames import check_drop_frames

#Import OS libraries
import os
import numpy as np
import dill as pickle
import cv2
import matplotlib.pyplot as plt
from loguru import logger

class Process():
    def __init__(self, session_ID):
        self.session = get_Session(session_ID)

    def create_session(self, settings) -> Session:        
        self.load_registration_transform()
        self.print_session_details(stage=1)
        self.session.camera_trigger = get_Camera_trigger(self.session)[0]
        self.session.audio          = get_Audio(self.session)
        self.session.video          = get_Video(self.session, settings, self.loaded_registration_transform)
        self.session.ttl            = get_TTL(self.session)
        self.print_session_details(stage=2)
        self.verify_all_frames_saved()
        self.verify_check_for_abberant_signals()
        self.verify_aligned_data_streams()
        # self.plot_ttl_pulse_index() # Comment out to run check that onsets align with pulse
        self.save_session()
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

    #Functions for data verification -------------------------------------------------------------------------------

    def verify_all_frames_saved(self):
        if self.session.camera_trigger.num_frames != self.session.video.num_frames:
            print("\n - Video contains {} frames, but {} frames were triggered! (for experiment: {}, mouse: {})---".format(self.session.video.num_frames, self.session.camera_trigger.num_frames, self.session.experiment, self.session.mouse))
            # check_drop_frames(self.session)
            self.session.camera_trigger = get_Camera_trigger(self.session, drop_frames=True)[0]
            if self.session.camera_trigger.num_frames == self.session.video.num_frames:
                print(" - Video realigned! Video contains {} frames, and {} frames were triggered (for experiment: {}, mouse: {})---".format(self.session.video.num_frames, self.session.camera_trigger.num_frames, self.session.experiment, self.session.mouse))
            else:
                print(" - Aligning failed")

    def verify_aligned_data_streams(self) -> None:
        if self.session.camera_trigger.num_samples != self.session.audio.num_samples:
            print("\n - Data streams have mismatched numbers of samples---\n  Camera trigger: {}\n  Audio input:    {}\n  Laser output:   {} + {} or {} or {} or {}".format(self.session.camera_trigger.num_samples, self.session.audio.num_samples, self.session.laser.num_samples, known_offset[0], known_offset[1], known_offset[2], known_offset[3]))

    def verify_check_for_abberant_signals(self) -> None:
        """_summary_
        Check for abberant signals via two means:
        1) Check that the signal values aren't lieing outside the logical confines - conduct for both big rig and efizz ttl signal
        2) Check the number of pulses are the same

        To do:
        - Repet for efizz box signal check
        - Write pulse count comparison
        """

        #Bonsai TTL check
        ttl = self.session.ttl.raw_signal #Retrieve raw TTL signal from session object
        above_errors = len(np.where(ttl > 5.2)[0]) #Count number of recordings where TTL signal is above 5.1 V
        below_errors = len(np.where(ttl < -0.2)[0]) #Count number of recordings where TTL signal is below <-0.1V
        num_errors = above_errors + below_errors #Compute a total number of erroneous recordings
        if num_errors:
            logger.info("Found {} samples with too high values in bonsai probe signal".format(num_errors))
            if (num_errors > 1000):
                logger.warning("Fede says this is too many errors. Signal unfit for use, terminating program.")
            return
    
    def plot_ttl_pulse_index(self):
        """_summary_
        Takes in both the ttl pulse onset index and the raw ttl signal.
        Produces a plot to overlay the two to check for any errors.
        Allows the user to verify if onset pulses align with configration. 
        """
        pulse_index = self.session.ttl.pulse_index[:10] #Take first 10 predictions of onset pulses
        ttl = self.session.ttl.raw_signal
        plt.plot(ttl[:110000]) #Plot the first 100k samples of the ttl signal
        for x in pulse_index:
            plt.axvline(x=x, color ='r') #plot a vert line for each onset
        plt.show()
