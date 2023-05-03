"""As a result of work with efizz data, many verifications are needed to ensure effective data processing. 
This file contains all the verifications needed to ensure the data is processed correctly. And that we are confident in
our ability to begin analysis.

There is one function, verify_all_frames_saved, that it not specific to efizz data. Though this script still seems
a good place to put it."""

# OS Libaries
from loguru import logger
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import scipy

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
        
        # if self.Process.session.camera_trigger.num_samples != len(self.Process.session.ttl.bonsai_TTL):
        #     print("Length of camera trigger:", self.Process.session.camera_trigger.num_samples)
        #     print("Length of bonsai TTL:", len(self.Process.session.ttl.bonsai_TTL))
        #     assert self.Process.session.camera_trigger.num_samples == len(self.Process.session.ttl.bonsai_TTL), "The length of camera trigger data doesn't match the length of the bonsai TTL"

    def verify_onsets_and_offsets(self):
        
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
            logger.success(f"Both bonsai and spikeGLX have equal number of {len(ephys_sync_onsets)} sync pulses")

        #Check the interval between sync signals in bonsai
        onsets_delta = np.diff(bonsai_sync_onsets)
        if len(set(onsets_delta)) > 1: # If more values exsist than just 30khz
            counts = {k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)}
            logger.warning(f"Bonsai sync triggers have variable delay. [Delay: Counts attributed to that delay]: {counts}")

        elif list(onsets_delta)[0] != self.Process.session.ttl.sampling_rate:
            # check that it lasts as long as it should
            logger.warning(f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {self.Process.session.ttl.sampling_rate})")
 
    def verify_check_means(self):
        """Check that the means of the bonsai TTL and the imec TTL are not
        too far away from expected mean.
        """
        if abs(np.mean(self.Process.session.ttl.bonsai_TTL) - 2.5) > 1:
            logger.error("Bonsai signal mean very far from expected average, cant be!")
            return
        if abs(np.mean(self.Process.session.ttl.imec_TTL) - 38.0) > 10:
            logger.error("Ephys signal mean ({}) very far from exected average, cant be!".format(np.mean(self.Process.session.ttl.imec_TTL)))
            return
    
    def verify_ttl_len_with_frame_duration(self):
        """Log the length of the video in seconds for visual inspection"""
        logger.info("The length of the video is: {}s".format(self.Process.session.video.num_frames * (1 / self.Process.session.video.fps)))

    def visulize_sync_output(self) -> tuple:
        """A function to plot the digital signals of the bonsai machine and the imec machine
        to ensure that after alignment they are identical. The alignment is done by
        regressing the imec signal on the bonsai signal and then shifting the imec signal.
        The intercept is removed because the true origin is close to zero. Adding the intercept
        breaks the regression. ALthough confusing the intercept was not learnt to be zero.
        
        Returns: Tuple:
        (r_value, slope)
        """
        
        # Trucate Signals to the first pulse onset 
        bonsaiSignal = self.Process.session.ttl.bonsai_TTL
        imecSignal   = self.Process.session.ttl.imec_TTL
        
        #TODO remove this
        # bonsaiSignal = self.Process.session.laser_sync.probe_Copy_TTL # this makes TTL focus on dev3
        
        # Create time vectors
        bonsai_samples = np.arange(0, len(bonsaiSignal))
        imec_samples = np.arange(0, len(imecSignal))
        
        # Align signals
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self.Process.session.ttl.ephys_sync_onsets, 
                                                                             self.Process.session.ttl.bonsai_sync_onsets)
        
        regression = lambda x: (slope * x) + intercept

        plt.scatter(self.Process.session.ttl.bonsai_sync_onsets / self.Process.session.ttl.sampling_rate, \
                    self.Process.session.ttl.ephys_sync_onsets / self.Process.session.ttl.sampling_rate)
        plt.title('slope = ' + str(slope) + '\n' + ' and intercept = ' + str(intercept))
        plt.xlabel('bonsai onset (s)')
        plt.ylabel('efizz onset (s)')
        plt.savefig(str(self.Process.session.processed_path) + "/" + "pulse_sync_regression.png")

        # Plot starting, middle and end samples to check alignment
        fig, axs = plt.subplots(3)
        fig.suptitle('Efizz syncing checks')
        axs[0].set_title("Start of sync")
        axs[0].plot(regression(imec_samples)[:500000] / self.Process.session.ttl.sampling_rate, imecSignal[:500000], color='blue', label = 'Imec')
        axs[0].plot(bonsai_samples[:500000]/ self.Process.session.ttl.sampling_rate, bonsaiSignal[:500000], color='red', label = 'Bonsai')
        
        middlePulseEfizz = int(np.median(self.Process.session.ttl.ephys_sync_onsets))
        middleBon = int(np.median(self.Process.session.ttl.bonsai_sync_onsets))
        
        axs[1].set_title("Middle of sync")
        axs[1].plot(regression(imec_samples)[middlePulseEfizz : middlePulseEfizz + 500000] / self.Process.session.ttl.sampling_rate, \
                    imecSignal[middlePulseEfizz : middlePulseEfizz + 500000], color='blue', label = 'Imec')
        axs[1].plot(bonsai_samples[middleBon : middleBon + 500000] / self.Process.session.ttl.sampling_rate, \
                    bonsaiSignal[middleBon : middleBon + 500000], color='red', label = 'Bonsai')
        
        LastPulsesBon = self.Process.session.ttl.bonsai_sync_onsets[-10]
        LastPulsesEfizz = self.Process.session.ttl.ephys_sync_onsets[-10]

        axs[2].set_title("End of sync")
        axs[2].plot(regression(imec_samples)[LastPulsesEfizz : LastPulsesEfizz + 500000] / self.Process.session.ttl.sampling_rate, \
                    imecSignal[LastPulsesEfizz : LastPulsesEfizz + 500000], color='blue', label = 'Imec')
        axs[2].plot(bonsai_samples[LastPulsesBon : LastPulsesBon + 500000] / self.Process.session.ttl.sampling_rate, \
                    bonsaiSignal[LastPulsesBon : LastPulsesBon + 500000], color='red', label = 'Bonsai')
                
        plt.savefig(str(self.Process.session.processed_path) + "/" + "pulse_sync_visualize.png")
        plt.legend()
        plt.show()
        
        # Last pulse to check that efizz spikes are not longer than this in another module
        LastPulse = self.Process.session.ttl.bonsai_sync_offsets[-1]
        
        return (r_value**2, slope, intercept), LastPulse
        
    def verify_clock_drift(self, r2_value):
        """Check that the clock drift is linear and not too large given that it is deterministic that
        the clocks between the two machines are not perfectly synced (imec and bonsai). Assuming that the bonsai
        clock is faster and thus we project from the spike machine to the bonsai machine. An R squared value of
        0.9999 was recommended by the NeuroGears team.
        """
        
        assert r2_value > 0.9999, "The R squared value of the linear regression is too low"
        logger.success(f"The R squared value of the linear regression for clock drift check has passed the tests and is: {r2_value**2}")
        
    def plot_residuals(self, show):
        """
        Caswell says to plot residuals across time points to check for clock drift
        """
        # Align signals
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self.Process.session.ttl.ephys_sync_onsets, 
                                                                             self.Process.session.ttl.bonsai_sync_onsets)
        
        regression = lambda x: (slope * x) + intercept
        
        imec_regressed = regression(self.Process.session.ttl.ephys_sync_onsets)
        
        residuals = self.Process.session.ttl.bonsai_sync_onsets - imec_regressed
        
        xs = np.arange(0, len(residuals))
        
        if show:
            plt.plot(xs, residuals)
            plt.show()