#Import custom libaries
from behave_analysis.process.session import Session, get_Session
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video
from behave_analysis.process.photoresistor import get_Photoresistor
from behave_analysis.process.ttl_sync import get_TTL, remove_idx_to_align_signals, get_onset_offset
from behave_analysis.utils.check_drop_frames import check_drop_frames
from behave_analysis.utils.load_bin_or_np import load_or_open

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

    def create_session(self, video_settings) -> Session:        
        self.load_registration_transform()
        self.print_session_details(stage=1)
        self.session.ttl            = get_TTL(self.session)
        self.session.camera_trigger = get_Camera_trigger(self.session, self.session.ttl.choose_index, self.session.ttl.temporal_difference)[0]
        self.session.audio          = get_Audio(self.session, self.session.ttl.choose_index, self.session.ttl.temporal_difference)
        self.session.video          = get_Video(self.session, video_settings, self.loaded_registration_transform)
        self.session.photo_resistor = get_Photoresistor(self.session, self.session.ttl.choose_index, self.session.ttl.temporal_difference)
        self.print_session_details(stage=2)
        self.verify_all_frames_saved()
        self.save_session()

        #Verify sync pulses
        self.verify_check_for_abberant_signals_in_bonsai()
        self.verify_aligned_data_streams()
        self.verify_check_TTL_length_and_means()
        self.verify_onsets_and_offsets()
        logger.info("Signals are ok and have past verification steps")

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

    #Functions for data verification / cleaning -------------------------------------------------------------------------------

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
            print("\n - Data streams have mismatched numbers of samples---\n  Camera trigger: {}\n  Audio input: {}\n".format(self.session.camera_trigger.num_samples, self.session.audio.num_samples))
            assert self.session.camera_trigger.num_samples == self.session.audio.num_samples, "Sample lens don't match"
        if self.session.camera_trigger.num_samples != len(self.session.ttl.bonsai_TTL):
            print("Length of camera trigger:", self.session.camera_trigger.num_samples)
            print("Length of bonsai TTL:", len(self.session.ttl.bonsai_TTL))
            assert self.session.camera_trigger.num_samples == len(self.session.ttl.bonsai_TTL), "The length of camera trigger doesn't match the length of the bonsai TTL"

    def verify_check_for_abberant_signals_in_bonsai(self) -> None:
        """_summary_
        Check for abberant signals via two means:
        1) Check that the signal values aren't lieing outside the logical confines - conduct for both big rig and efizz ttl signal
        2) Check the number of pulses are the same

        To do:
        - Repet for efizz box signal check
        - Write pulse count comparison
        """

        #Bonsai TTL check
        ttl = self.session.ttl.bonsai_TTL #Retrieve raw TTL signal from session object
        above_errors = len(np.where(ttl > 5.2)[0]) #Count number of recordings where TTL signal is above 5.1 V
        below_errors = len(np.where(ttl < -0.2)[0]) #Count number of recordings where TTL signal is below <-0.1V
        num_errors = above_errors + below_errors #Compute a total number of erroneous recordings
        if num_errors:
            logger.info("Found {} samples with too high values in bonsai probe signal".format(num_errors))
            if (num_errors > 1000):
                logger.warning("Fede says this is too many errors. Signal unfit for use, terminating program.")
            return

    def verify_check_TTL_length_and_means(self) -> None:
        """Check that the lengths of the bonsai TTL and the imec TTL are of a similar length and are not
        too far away from expected mean.
        """
        if len(self.session.ttl.bonsai_TTL) != len(self.session.ttl.imec_TTL):
            logger.warning("The sync signals have very different lengths, this cant be after resampling!")
            return
        
        if abs(np.mean(self.session.ttl.bonsai_TTL) - 2.5) > 1:
            logger.warning("Bonsai signal mean very far from expected average, cant be!")
            return
        if abs(np.mean(self.session.ttl.imec_TTL) - 38.0) > 6:
            logger.warning("Ephys signal mean very far from exected average, cant be!")
            return

    #Check onset and offsets for errors
    def verify_onsets_and_offsets(self):
        logger.debug("Verifying sync signal pulses")
        is_ok = True  # until proven otherwise
        
        # get pulses onsets
        bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(self.session.ttl.bonsai_TTL, 2.5)
        ephys_sync_onsets, ephys_sync_offsets   = get_onset_offset(self.session.ttl.imec_TTL, 45)

        # check if numbers make sense
        if len(bonsai_sync_onsets) != len(bonsai_sync_offsets):
            is_ok = False
            logger.warning(f"BONSAI - Unequal number of onsets/offsets ({len(bonsai_sync_offsets)}/{len(bonsai_sync_onsets)})")
    
        if len(ephys_sync_onsets) != len(ephys_sync_offsets):
            is_ok = False
            logger.warning(f"EPHYS - Unequal number of onsets/offsets ({len(ephys_sync_offsets)}/{len(ephys_sync_onsets)})")

        # check same results for bonsai and ephys
        if len(bonsai_sync_onsets) != len(ephys_sync_onsets):
            logger.error(f"Incosistent number of triggers! Bonsai {len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}")
            is_ok = False
            logger.warning("When inspecting probe sync signal found different number of pulses for bonsai"
                           f"{len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}")
    
        else:
            logger.debug(f"Both bonsai and spikeGLX have {len(ephys_sync_onsets)} sync pulses")

        if ephys_sync_onsets[0] <= bonsai_sync_onsets[0]:
            is_ok = False
            logger.warning("Bonsai should start first!")

        #Check the interval between sync signals in bonsai
        onsets_delta = np.diff(bonsai_sync_onsets)
        if len(set(onsets_delta)) > 1: #If more values exsist than just 30khz
            counts = {k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)}
            logger.warning(f"Bonsai sync triggers have variable delay. [Delay: Counts attributed to that delay]: {counts}")

        elif list(onsets_delta)[0] != self.session.ttl.sampling_rate:
            # check that it lasts as long as it should
            is_ok = False
            logger.warning(f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {self.session.ttl.sampling_rate})")

        return (is_ok)