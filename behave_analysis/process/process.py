#Import custom libaries
from settings.settings_process import settings_process as settings_p # So to check if pipeline includes efizz
from behave_analysis.process.session import Session, get_Session
from behave_analysis.process.camera_trigger import get_Camera_trigger
from behave_analysis.process.audio import get_Audio
from behave_analysis.process.video import get_Video
from behave_analysis.process.photoresistor import get_Photoresistor
from behave_analysis.utils.check_drop_frames import check_drop_frames

# Additional libraries if running with efizz
if settings_p.efizz:
    from behave_analysis.process.ephys import get_Ephys
    from behave_analysis.process.ttl_sync import get_TTL, remove_bonsai_idx_to_align_signals, get_onset_offset, derivative
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
        
    def create_session(self, video_settings):        
        self.load_registration_transform()
        self.print_session_details(stage=1)
        
        print("hello")
        
        if settings_p.efizz:
            self.session.ttl = get_TTL(self.session)
            
        # Normally would be equal to self.session.ttl.choose_index
        # Downsampling required if there is efiz
        self.session.camera_trigger = get_Camera_trigger(self.session, down_sample = False)[0]
        self.session.audio          = get_Audio(self.session, down_sample = False)
        self.session.video          = get_Video(self.session, video_settings, self.loaded_registration_transform)
        self.session.photo_resistor = get_Photoresistor(self.session, down_sample = False)
        
        if settings_p.efizz:
            self.session.ephys = get_Ephys(self.session)
            
        self.print_session_details(stage=2)
        self.save_session()
        
        if settings_p.efizz:
            self.verify_check_for_abberant_signals_in_bonsai()
            self.verify_aligned_data_streams()
            self.verify_check_means()
            self.verify_onsets_and_offsets()
            self.visulize_sync_output() #Plot the resulting sync pulses, uncomment to see
            self.verify_ttl_len_with_frame_duration()
            
        self.verify_all_frames_saved()
        logger.info("All verifications steps passed")
        return self.session

    def save_session(self, overwrite=True):
        assert not os.path.isfile(self.session.metadata_file) or overwrite, "Permission to save not granted"
        with open(self.session.metadata_file, "wb") as dill_file: pickle.dump(self.session, dill_file)

    def load_session(self):
        try:
            with open(self.session.metadata_file, "rb") as dill_file: session = pickle.load(dill_file)
        except EOFError:
            print(f"The file location is: {self.session.metadata_file}. Is this correct? Does a metadata file exsist here?")
            print("Delete meta file")
            return

        return session

    def load_registration_transform(self):
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
            logger.error("Missing frames check what happened")
            print("\n - Video contains {} frames, but {} frames were triggered! (for experiment: {}, mouse: {})---".format(self.session.video.num_frames, self.session.camera_trigger.num_frames, self.session.experiment, self.session.mouse))
            # check_drop_frames(self.session)
            self.session.camera_trigger = get_Camera_trigger(self.session, drop_frames=True)[0]
            if self.session.camera_trigger.num_frames == self.session.video.num_frames:
                print(" - Video realigned! Video contains {} frames, and {} frames were triggered (for experiment: {}, mouse: {})---".format(self.session.video.num_frames, self.session.camera_trigger.num_frames, self.session.experiment, self.session.mouse))
            else:
                print(" - Aligning failed")
        
        else: 
            logger.info("Frames triggered are the same number as frames captured")

    def verify_aligned_data_streams(self):
        if self.session.camera_trigger.num_samples != self.session.audio.num_samples:
            print("\n - Data streams have mismatched numbers of samples---\n  Camera trigger: {}\n  Audio input: {}\n".format(self.session.camera_trigger.num_samples, self.session.audio.num_samples))
            assert self.session.camera_trigger.num_samples == self.session.audio.num_samples, "Sample lens don't match"
        if self.session.camera_trigger.num_samples != len(self.session.ttl.bonsai_TTL):
            print("Length of camera trigger:", self.session.camera_trigger.num_samples)
            print("Length of bonsai TTL:", len(self.session.ttl.bonsai_TTL))
            logger.error("Fix assertion error")
            # assert self.session.camera_trigger.num_samples == len(self.session.ttl.bonsai_TTL), "The length of camera trigger doesn't match the length of the bonsai TTL"

    def verify_check_for_abberant_signals_in_bonsai(self):
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

    def verify_check_means(self):
        """Check that the means of the bonsai TTL and the imec TTL are not
        too far away from expected mean.
        """
        if abs(np.mean(self.session.ttl.bonsai_TTL) - 2.5) > 1:
            logger.warning("Bonsai signal mean very far from expected average, cant be!")
            return
        if abs(np.mean(self.session.ttl.imec_TTL) - 38.0) > 10:
            logger.warning("Ephys signal mean ({}) very far from exected average, cant be!".format(np.mean(self.session.ttl.imec_TTL)))
            return

    #Check onset and offsets for errors
    def verify_onsets_and_offsets(self):
        logger.info("Verifying sync signal pulses")
        
        # get pulses onsets
        # bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(self.session.ttl.bonsai_TTL, 2.5)
        # ephys_sync_onsets, ephys_sync_offsets   = get_onset_offset(self.session.ttl.imec_TTL, 45)

        # Get onset and offsets
        bonsai_sync_onsets  = self.session.ttl.bonsai_sync_onsets
        bonsai_sync_offsets = self.session.ttl.bonsai_sync_offsets
        ephys_sync_onsets   = self.session.ttl.ephys_sync_onsets
        ephys_sync_offsets  = self.session.ttl.ephys_sync_offset
        
        # check if numbers make sense
        if len(bonsai_sync_onsets) != len(bonsai_sync_offsets):
            logger.error(f"BONSAI - Unequal number of onsets/offsets ({len(bonsai_sync_offsets)}/{len(bonsai_sync_onsets)})")
    
        if len(ephys_sync_onsets) != len(ephys_sync_offsets):
            logger.error(f"EPHYS - Unequal number of offsets/onsets ({len(ephys_sync_offsets)}/{len(ephys_sync_onsets)})")

        # check same results for bonsai and ephys
        if len(bonsai_sync_onsets) != len(ephys_sync_onsets):
            logger.error(f"Incosistent number of triggers! Bonsai {len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}")
            logger.warning("When inspecting probe sync signal found different number of pulses for bonsai: "
                           f"{len(bonsai_sync_onsets)} and SpikeGLX: {len(ephys_sync_onsets)}")
    
        else:
            logger.info(f"Both bonsai and spikeGLX have {len(ephys_sync_onsets)} sync pulses")

        #Check the interval between sync signals in bonsai
        onsets_delta = np.diff(bonsai_sync_onsets)
        if len(set(onsets_delta)) > 1: #If more values exsist than just 30khz
            counts = {k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)}
            logger.warning(f"Bonsai sync triggers have variable delay. [Delay: Counts attributed to that delay]: {counts}")

        elif list(onsets_delta)[0] != self.session.ttl.sampling_rate:
            # check that it lasts as long as it should
            logger.warning(f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {self.session.ttl.sampling_rate})")

        #Test differences
        temporal_difference = np.diff(bonsai_sync_onsets) - np.diff(ephys_sync_onsets) # Comare delta onsets
        off_set_difference  = np.diff(bonsai_sync_offsets) - np.diff(ephys_sync_offsets) # Compare delta offsets
        assert np.all(temporal_difference == 0), "Resample failed, there should be no difference in pulse length at this stage"
        assert np.all(off_set_difference[:-2]) == 0, "Resample failed, there should be no difference in pulse length at this stage apart from last pulse"

    def visulize_sync_output(self):
        """A function to plot the digital signals of the bonsai machine and the imec machine
        to ensure that after resampling and alignment they are identical.
        """

        # Retrieve algined signals
        bonsai_TTL = self.session.ttl.bonsai_TTL
        imec_TTL = self.session.ttl.imec_TTL

        # Print the length of the arrays
        logger.info("Length of the Bonsai TTL signal is {}".format(len(bonsai_TTL)))
        logger.info("Length of the Imec TTL signal is {}".format(len(imec_TTL)))

        # Plotting logic
        fig, axs = plt.subplots(2)
        fig.suptitle("First and last 100k samples, TTL comparison")
        axs[0].plot(bonsai_TTL[:100000], label = "Bonsai TTL")
        axs[0].plot(imec_TTL[:100000], label = "Imec TTL")
        axs[0].set_title("Check the first pulses are aligned")

        axs[1].plot(bonsai_TTL[(len(bonsai_TTL) - 100000):], label = "Bonsai TTL")
        axs[1].plot(imec_TTL[(len(imec_TTL) - 100000):], label = "Imec TTL")
        axs[1].set_title("Check the last pulses are aligned")
        fig.legend()
        plt.show()

        # Assertions
        assert len(bonsai_TTL) == len(imec_TTL), "Imec TLL signal length should be equal to Bonsai TTL"
    
    def verify_ttl_len_with_frame_duration(self):
        """Check that the number of frames multipled by frame duration is the same 
        length of the bonsai signal in seconds
        """
        num_frames = self.session.video.num_frames
        video_length = num_frames * (1/ self.session.video.fps)
        logger.info("The length of the video is: {}s".format(video_length))
        logger.info("The length of bonsai TTL is: {}s". format(len(self.session.ttl.bonsai_TTL) / 30000))

        # Differenece in len
        diff = abs(video_length - len(self.session.ttl.bonsai_TTL) / 30000)
        assert diff < 0.5, "Video length and bonsai signal should not differ by more than half a second"
        