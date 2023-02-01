"""As a result of work with efizz data, many verifications are needed to ensure effective data processing. 
This file contains all the verifications needed to ensure the data is processed correctly. And that we are confident in
our ability to begin analysis."""

# OS Libaries
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

class Verifications():
    def __init__(self, Process):
        """A class that ingests the Process Class inorder to verify it's outputs"""
        self.Process = Process
    
    def verify_all_frames_saved(self):
        """A function that checks if all the triggered frames were saved in the video file. This may 
        not be the case if the machine is running slower than expected. Also due to camera settings, lag etc 
        there may be more video frames than triggered frames. This function checks for both of these cases.
        """
        if self.Process.session.camera_trigger.num_frames != self.Process.session.video.num_frames:
            # assert self.session.camera_trigger.num_frames == self.session.video.num_frames, "The number of triggers does not match the number of video frames, processing has failed, kill script"
            # Assert removed, is it important to have triggers matched to frames?
            logger.warning(f"The number of camera triggers ({self.Process.session.camera_trigger.num_frames}) does not match the number of video frames ({self.Process.session.video.num_frames})")
            
    def verify_check_for_abberant_signals_in_bonsai(self):
        """ Check for abberant signals via two means: 1) Check that the signal values aren't lieing outside the logical confines - 
        conduct for both big rig and efizz ttl signal or 2) Check the number of pulses are the same.

        To do:
        - Repet for efizz box signal check
        - Write pulse count comparison
        """
        ttl = self.Process.session.ttl.bonsai_TTL #Retrieve raw TTL signal from session object
        above_errors = len(np.where(ttl > 5.2)[0]) #Count number of recordings where TTL signal is above 5.1 V
        below_errors = len(np.where(ttl < -0.2)[0]) #Count number of recordings where TTL signal is below <-0.1V
        num_errors = above_errors + below_errors #Compute a total number of erroneous recordings
        
        if num_errors:
            logger.warning("Found {} samples with too high values in bonsai probe signal".format(num_errors))
            if (num_errors > 1000):
                logger.error("Fede says this is too many errors. Signal unfit for use, terminating program.")
                assert False, "Too many errors in TTL signal, signal unfit for use, terminating program."
                
    def verify_aligned_data_streams(self):
        """A function that checks if the data streams are aligned. This is done by checking if the length of signals in the audio and camera trigger streams are the same.
        And also checking if length of the camera trigger and bonsai TTL streams are the same.
        """
        
        if self.Process.session.camera_trigger.num_samples != self.Process.session.audio.num_samples:
            logger.error(f"The number of samples in the audio {self.Process.session.audio.num_samples} and camera trigger {self.Process.session.camera_trigger.num_samples} streams do not match after alignment")
            assert self.Process.session.camera_trigger.num_samples == self.Process.session.audio.num_samples, "The length of camera trigger doesn't match the length of the audio, processing has failed, kill script"
        
        if self.Process.session.camera_trigger.num_samples != len(self.Process.session.ttl.bonsai_TTL):
            print("Length of camera trigger:", self.Process.session.camera_trigger.num_samples)
            print("Length of bonsai TTL:", len(self.Process.session.ttl.bonsai_TTL))
            # assert self.session.camera_trigger.num_samples == len(self.session.ttl.bonsai_TTL), "The length of camera trigger doesn't match the length of the bonsai TTL"
            logger.error("Fix assertion error")

    def verify_onsets_and_offsets(self):
        logger.info("Verifying sync signal pulses")
        
        # Get onset and offsets
        bonsai_sync_onsets  = self.Process.session.ttl.bonsai_sync_onsets
        bonsai_sync_offsets = self.Process.session.ttl.bonsai_sync_offsets
        ephys_sync_onsets   = self.Process.session.ttl.ephys_sync_onsets
        ephys_sync_offsets  = self.Process.session.ttl.ephys_sync_offset
        
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

        elif list(onsets_delta)[0] != self.Process.session.ttl.sampling_rate:
            # check that it lasts as long as it should
            logger.warning(f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {self.Process.session.ttl.sampling_rate})")

        #Test differences
        temporal_difference = np.diff(bonsai_sync_onsets) - np.diff(ephys_sync_onsets) # Comare delta onsets
        off_set_difference  = np.diff(bonsai_sync_offsets) - np.diff(ephys_sync_offsets) # Compare delta offsets
        assert np.all(temporal_difference == 0), "Resample failed, there should be no difference in pulse length at this stage"
        assert np.all(off_set_difference[:-2]) == 0, "Resample failed, there should be no difference in pulse length at this stage apart from last pulse"
    
    def verify_check_means(self):
        """Check that the means of the bonsai TTL and the imec TTL are not
        too far away from expected mean.
        """
        if abs(np.mean(self.Process.session.ttl.bonsai_TTL) - 2.5) > 1:
            logger.warning("Bonsai signal mean very far from expected average, cant be!")
            return
        if abs(np.mean(self.Process.session.ttl.imec_TTL) - 38.0) > 10:
            logger.warning("Ephys signal mean ({}) very far from exected average, cant be!".format(np.mean(self.Process.session.ttl.imec_TTL)))
            return
    
    def verify_ttl_len_with_frame_duration(self):
        """Check that the number of frames multipled by frame duration is the same 
        length of the bonsai signal in seconds
        """
        num_frames = self.Process.session.video.num_frames
        video_length = num_frames * (1 / self.Process.session.video.fps)
        logger.info("The length of the video is: {}s".format(video_length))
        logger.info("The length of bonsai TTL is: {}s". format(len(self.Process.session.ttl.bonsai_TTL) / 30000))

        # Differenece in len
        diff = abs(video_length - len(self.Process.session.ttl.bonsai_TTL) / 30000)
        assert diff < 0.5, "Video length and bonsai signal should not differ by more than half a second"
        
    def visulize_sync_output(self):
        """A function to plot the digital signals of the bonsai machine and the imec machine
        to ensure that after resampling and alignment they are identical.
        """

        # Retrieve algined signals
        bonsai_TTL = self.Process.session.ttl.bonsai_TTL
        imec_TTL = self.Process.session.ttl.imec_TTL

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