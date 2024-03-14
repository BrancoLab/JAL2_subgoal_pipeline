""" A local test variantion of our spike interface pipeline. The point of this script is to run the necessary 
preprocessing, sorting, postprocessing and quality metric computation spike interface functions. It is designed
to run on HPC and thus this is a indepedent module. To run on HPC, the paths need to be changed to suit the HPC.

#NOTE:
-- There are many more functions in the interface to be used and looked at, this is a basic pipeline
-- Currently no post processing functions are being used
"""

import os
from pathlib import Path

import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np

# Set global arguments for parrallel processing
global_job_kwargs = dict(n_jobs=10, chunk_duration="1s", progress_bar=True)
si.set_global_job_kwargs(**global_job_kwargs)

# Parameters for Kilosort2.5 change these to suit your data
PARAMS_KS_25 = {
    "detect_threshold": 6,
    "projection_threshold": [8, 4],  # Spike detection threshold
    "AUCsplit": 0.8,  # Splitting cluster threshold, 0.9 is stricter, 0.7 is looser
    "preclust_threshold": 8,
    "car": False,  # Common average reference setting to False as we do this with spikeinterface
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
    "do_correction": True,  # Drift correction
    "wave_length": 61,
    "keep_good_only": False,
    "n_jobs": 40,
    "chunk_duration": "1s",
    "progress_bar": True,
}


class SpikeInterface:
    """A class that calls the spike interface API and conducts a full pipeline of processing efizz data"""

    def __init__(self, path_to_ks, binary_file_path):

        # Set the paths
        self.save_path = binary_file_path.parent[0]
        self.spikeglx_folder = binary_file_path.parent[1]
        self.path_to_ks = path_to_ks
        self.path_to_binary = binary_file_path

        # Run the pipeline
        self.raw_rec = self.load_imec_bin(self.spikeglx_folder)
        self.extract_and_save_sync_channel()
        self.filt_rec = self.preprocess_spike_interface()
        self.plot_channel_map()
        self.plot_noise_estimation()
        self.sorting = self.run_kilosort25()
        self.we = self.extract_waveforms()
        self.quality_metrics_spike_interface()

    def load_imec_bin(self, spikeglx_folder: Path) -> object:
        """Return the raw recording object from the spikeglx folder loaded from the .imec0.ap.bin file
        In the documentation it says that if the load_sync_channel is set to True then the probe is not loaded.
        I.e plotting channel maps is not possible. So I wrote a separate function to extract the sync channel."""

        print("Loading the raw .bin file from spikeglx")
        raw_recording = si.read_spikeglx(spikeglx_folder, stream_name="imec0.ap", load_sync_channel=False)
        print(f"Raw recording object loaded from spikeglx folder: {raw_recording}")
        return raw_recording

    def extract_and_save_sync_channel(self) -> None:
        """Extract and save the sync channel from the raw recording object"""
        sync_channel = si.read_binary(self.path_to_binary, num_channels=385, dtype="int16", sampling_frequency=30000).channel_slice(channel_ids=[384])
        sync_trace = sync_channel.get_traces()
        np.save(self.save_path / "sync_channel.npy", sync_trace)

    def preprocess_spike_interface(self):
        """Preprocess the raw recording using spikeinterface.

        This includes:
        + bandpass filtering,
        + bad channel detection and removal,
        + correcting for phase shift this is due a small delay in sample periods between channels in npx devices
        + common reference

        Returns:
        -- The preprocessed recording object.
        """
        print("Preprocessing the raw recording using SpikeInterface")
        rec1 = si.bandpass_filter(recording=self.raw_rec, freq_min=300, freq_max=6000)
        bad_channel_ids, _ = si.detect_bad_channels(rec1)
        rec2 = rec1.remove_channels(bad_channel_ids)
        print(f"Bad channel IDs that have been removed: {bad_channel_ids}")
        if len(bad_channel_ids) > 10:
            print("ERROR - More than 10 channels have been removed, seems like a lot?")
        rec3 = si.phase_shift(rec2)  # Correct sampling delay between channels in NPX devices
        rec4 = si.common_reference(rec3, operator="median", reference="global")
        rec = rec4
        print("Preprocessing complete")
        return rec

    def run_kilosort25(self, rerun_sorter=False):
        """Run Kilosort2.5 on the preprocessed recording object and save the output to the spike interface folder."""
        print("Running Kilosort 2.5 from spikeinterface")
        si.Kilosort2_5Sorter.set_kilosort2_5_path(self.path_to_ks)

        if rerun_sorter:
            sorting_obj = si.run_sorter(
                "kilosort2_5",
                self.filt_rec,
                output_folder= self.save_path / "SI_KS_output",
                verbose=True,
                remove_existing_folder=True,
                **PARAMS_KS_25,
            )
            print(f"Kilosort 2.5 has been re-run and the output has been saved to the spikeglx folder: {sorting_obj}")

        # Check if Kilosort2.5 has already been run if not then run it
        else:
            if (self.save_path / "SI_KS_output").is_dir():
                sorting_obj = si.read_sorter_folder(self.save_path / "SI_KS_output")
                print("Kilosort 2.5 has already been run and the output has been read from the spike interface folder")
            else:
                sorting_obj = si.run_sorter(
                    "kilosort2_5",
                    self.filt_rec,
                    output_folder= self.save_path / "SI_KS_output",
                    verbose=True,
                    remove_existing_folder=True,
                    **PARAMS_KS_25,
                )
                print(f"Kilosort 2.5 has been run and the output has been saved to the spikeglx folder: {sorting_obj}")

        print(f"Loaded Kilosort 2.5 output: {sorting_obj}")

        return sorting_obj

    def extract_waveforms(self):
        """The core of postprocessing and quality metrice computations
        revolves around extracting waveforms from paired recording-sorting objects."""
        print("Extracting waveforms")
        we = si.extract_waveforms(self.filt_rec, self.sorting, folder=self.save_path / "waveforms", load_if_exists=True, **global_job_kwargs)
        return we

    def quality_metrics_spike_interface(self) -> None:
        """Create a pandas dataframe of quality metrics for each unit in the sorting output."""
        print("Compuyting quality metrics for the sorting output")
        qm = si.compute_quality_metrics(self.we, verbose=True, n_jobs=global_job_kwargs["n_jobs"])
        qm.to_csv(self.save_path / "quality_metrics.csv")

    # --------------------------------- PLOTTING ---------------------------------

    def plot_channel_map(self):
        """Plots each channel ID to it's pad location on the probe and saves the plot to the spikeglx folder."""
        print("Plotting channel map")
        _, ax = plt.subplots(figsize=(15, 10))
        si.plot_probe_map(self.raw_rec, ax=ax, with_channel_ids=True)
        ax.set_ylim(-300, 2500)
        plt.savefig(self.save_path / Path("channel_map.png"))
        plt.close()

    def plot_noise_estimation(self):
        """we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
        Estimate noise for each channel using MAD (Mean absolute deviation) methods. A method of estimating the
        variance of the noise on a wavelet component. As a reference, the spikeinterface documentation has a histogram
        centred around 15 mV, with max at 25mv. And min at 12.5mv
        """
        print("Plotting noise estimation")
        noise_levels_microv = si.get_noise_levels(self.filt_rec, return_scaled=True)
        noise_levels_microv_unprocessed = si.get_noise_levels(self.raw_rec, return_scaled=True)
        # noise_levels_int16 = si.get_noise_levels(self.filt_rec, return_scaled=False)
        _, ax = plt.subplots()
        _ = ax.hist(noise_levels_microv, bins=np.arange(5, 30, 2.5), color="b", alpha=0.5, label="filtered")
        _ = ax.hist(noise_levels_microv_unprocessed, bins=np.arange(5, 30, 2.5), color="r", alpha=0.5, label="unfiltered")
        ax.legend()
        ax.set_xlabel("noise  [microV]")
        ax.set_ylabel("Channel count")
        ax.set_title("Variation in noise levels across channels for filtered and unfiltered data")
        plt.savefig(self.save_path / Path("noise_estimation.png"))
        plt.close()
        return


if __name__ == "__main__":
    print("Running SpikeInterface piepline. This will preprocess and spike sort the data")

    # Change these paths
    binary_file_path = Path(
        r"E:\efizz\JAL007\JAL007_empty_shelter_2024_03_05T13_45_47\empty_shelter_5mar2024_g0\empty_shelter_5mar2024_g0_imec0\empty_shelter_5mar2024_g0_t0.imec0.ap.bin"
    )
    path_to_ks = r"C:\Users\laurence\Documents\Kilosort-2.5"

    SpikeInterface(path_to_ks, binary_file_path)
    print("SpikeInterface has run successfully")
