""" The point of this script is to run the necessary preprocessing spike interface functions 
and then run Kilosort2.5 on the preprocessed recording object. There is also scope to do more 
such as drift correction, post processing and validation of the sorting output. """

import spikeinterface.full as si
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

# Parameters for Kilosort2.5 change these to suit your data
PARAMS_KS_25 = {
    "detect_threshold": 6,
    "projection_threshold": [10, 4],
    "preclust_threshold": 8,
    "car": True,
    "minFR": 0.1,
    "minfr_goodchannels": 0.1,
    "nblocks": 5,
    "sig": 20,
    "freq_min": 150,
    "sigmaMask": 30,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "NT": None,
    "do_correction": True,
    "wave_length": 61,
    "keep_good_only": False,
    "n_jobs": 40,
    "chunk_duration": "1s",
    "progress_bar": True,
}


class SpikeInterface:
    """A class that calls the spike interface API to preprocess and spike sort our data

    Ouput:
    -- The output of the Kilosort2.5 spike sorting algorithm is saved"""

    def __init__(self):
        self.spikeglx_folder = r"E:\efizz\JAL007\JAL007_empty_shelter_2024_03_05T13_45_47\empty_shelter_5mar2024_g0"
        self.raw_rec = self.load_imec_bin(self.spikeglx_folder)
        self.filt_rec = self.preprocess_spike_interface()
        # self.run_kilosort25()

        # Plot the channel map and noise estimation
        self.plot_channel_map()
        self.plot_noise_estimation()

    def load_imec_bin(self, spikeglx_folder):
        """Return the raw recording object from the spikeglx folder loaded from the .imec0.ap.bin file"""
        return si.read_spikeglx(spikeglx_folder, stream_name="imec0.ap", load_sync_channel=False)

    def preprocess_spike_interface(self):
        """Preprocess the raw recording using spikeinterface.

        This includes:
        + bandpass filtering,
        + bad channel detection and removal,
        + correcting for phase shift this is due a small delay in sample periods between channels
        + common reference

        Returns:
        -- The preprocessed recording object.
        """
        rec1 = si.bandpass_filter(recording=self.raw_rec, freq_min=300, freq_max=6000)
        bad_channel_ids, _ = si.detect_bad_channels(rec1)
        rec2 = rec1.remove_channels(bad_channel_ids)
        logger.info(f"Bad channel IDs that have been removed: {bad_channel_ids}")
        if len(bad_channel_ids) > 10:
            logger.warning("More than 10 channels have been removed, seems like a lot?")
        rec3 = si.phase_shift(rec2)  # Correct sampling delay between channels in NPX devices
        rec4 = si.common_reference(rec3, operator="median", reference="global")
        rec = rec4
        return rec

    def run_kilosort25(self):
        """Run Kilosort2.5 on the preprocessed recording object and save the output to the spikeglx folder."""
        logger.info("Running Kilosort 2.5 from spikeinterface")
        ks_path = r"C:\Users\laurence\Documents\Kilosort-2.5"
        si.Kilosort2_5Sorter.set_kilosort2_5_path(ks_path)
        _ = si.run_sorter("kilosort2_5", self.filt_rec, output_folder=self.spikeglx_folder + "/SIkilosort2.5_output", verbose=True, **PARAMS_KS_25)
        sorting_obj = si.read_sorter_folder(self.spikeglx_folder + "/SIkilosort2.5_output")
        logger.info(f"Kilosort 2.5 has been run and the output has been saved to the spikeglx folder: {sorting_obj}")

    # --------------------------------- PLOTTING ---------------------------------

    def plot_channel_map(self):
        """Plots each channel ID to it's pad location on the probe and saves the plot to the spikeglx folder."""
        _, ax = plt.subplots(figsize=(15, 10))
        si.plot_probe_map(self.raw_rec, ax=ax, with_channel_ids=True)
        ax.set_ylim(-300, 2500)
        plt.savefig(self.spikeglx_folder + r"\channel_map.png")
        plt.close()

    def plot_noise_estimation(self):
        """we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16)."""
        noise_levels_microv = si.get_noise_levels(self.filt_rec, return_scaled=True)
        noise_levels_int16 = si.get_noise_levels(self.filt_rec, return_scaled=False)
        _, ax = plt.subplots()
        _ = ax.hist(noise_levels_microv, bins=np.arange(5, 30, 2.5))
        ax.set_xlabel("noise  [microV]")
        plt.savefig(self.spikeglx_folder + r"\noise_estimation.png")
        plt.close()
        return noise_levels_int16

    # --------------------------------- Not working ---------------------------------

    # def check_for_drifts(rec, noise_levels_int16):
    #     logger.info("Checking for drifts in the recording")
    #     job_kwargs = dict(n_jobs=40, chunk_duration="1s", progress_bar=True)
    #     peaks = detect_peaks(rec, method="locally_exclusive", noise_levels=noise_levels_int16, detect_threshold=5, radius_um=50.0, **job_kwargs)
    #     peak_locations = localize_peaks(rec, peaks, method="center_of_mass", radius_um=50.0, **job_kwargs)
    #     fs = rec.sampling_frequency
    #     _, ax = plt.subplots(figsize=(10, 8))
    #     ax.scatter(peaks["sample_ind"] / fs, peak_locations["y"], color="k", marker=".", alpha=0.002)
    #     plt.show()
