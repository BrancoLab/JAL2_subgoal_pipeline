# Custom lib
from behave_analysis.utils.AI_dataClass_objects import Electrophysiology

# OS libs
import os
import numpy as np
from loguru import logger
import pandas as pd

class LoadEfizz:
    def __init__(self, session_ID, settings):
        self.file_path = os.path.join(session_ID.base_path, session_ID.file_path)
        self.settings = settings
        self.files = self.collect_efizz_files()

    def collect_efizz_files(self) -> list:
        """
        A function that collects all the efizz files from the session folder. Although dirnames
        not used, it is needed to walk the directory tree
        """
        logger.info("Searching through efizz files")
        efizz_folder = os.path.join(self.file_path,[x for x in os.listdir(self.file_path) if '_g0' in x][0])
        KS_folder = os.path.join(efizz_folder, [x for x in os.listdir(efizz_folder) if 'imec' in x][0])
        KS_folder = os.path.join(KS_folder, 'SI_KS_output', 'sorter_output')
        files = []
        for dirpath, dirnames, filenames in os.walk(KS_folder):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        logger.success("Efizz file names collected")
        return files

    def select_and_load_efizz_files(self) -> None:
        """
        A function that selects the efizz files that are needed for the pipeline.
        Should return a list of strings
        """
        assert len(self.files) != 0, "Session list should not be empty"
        assert len(self.filter_by_ending(self.files, "spike_times.npy")) > 0, "No spike_times.npy file was found! Make sure KS was run and output is in imec folder"
        assert len(self.filter_by_ending(self.files, "spike_times.npy")) == 1, "There should only be one spike_times.npy file"
        assert len(self.filter_by_ending(self.files, "spike_clusters.npy")) == 1, "There should only be one spike_clusters.npy file"
        assert len(self.filter_by_ending(self.files, "cluster_group.tsv")) == 1, "There should only be one cluster_group.tsv file"

        spike_times = np.load(self.filter_by_ending(self.files, "spike_times.npy")[0])
        spike_clusters = np.load(self.filter_by_ending(self.files, "spike_clusters.npy")[0])
        spike_clusters = np.hstack(spike_clusters)
        if self.settings.remove_duplicate_spike_times:
            self.spike_times, self.spike_clusters = self.remove_duplicate_spike_times(spike_times, spike_clusters, censored_period_ms=self.settings.duplicate_spikes_censored_period_ms)
            logger.info(f"The number of spikes after removing duplicates is: {len(self.spike_times)}")
        else:
            self.spike_times = spike_times
            self.spike_clusters = spike_clusters
            logger.info(f"The number of spikes is: {len(self.spike_times)}")
        
        efizz_folder = os.path.join(self.file_path,[x for x in os.listdir(self.file_path) if '_g0' in x][0])
        efizz_folder = os.path.join(efizz_folder, [x for x in os.listdir(efizz_folder) if 'imec' in x][0])

        # sync channel
        sync = self.filter_by_ending(os.listdir(efizz_folder), "exported.imec0.ap.bin")
        if len(sync) > 0: 
            self.imec_sync_path = os.path.join(efizz_folder,sync[0])
        else:
            logger.warning("No exported .bin sync channel was found!")
            self.imec_sync_path = os.path.join(efizz_folder, self.filter_by_ending(os.listdir(efizz_folder), "_t0.imec0.ap.bin")[0])
        self.imec_bin_path = self.filter_by_ending(os.listdir(efizz_folder), "_t0.imec0.ap.bin")[0]
        
        # cluster classification
        if self.settings.cluster_labels == "manual":
            logger.info("Using manual (phy) cluster classification, if no curation this is the same as kilosort")
            cluster_label = self.filter_by_ending(self.files, "cluster_group.tsv")[0]
        if self.settings.cluster_labels == "bombcell":
            cluster_label = self.filter_by_ending(self.files, "cluster_bc_unitType.tsv")
            if len(cluster_label) == 0:
                logger.warning("No bombcell cluster classification file was found! Defaulting to kilosort classification")
                cluster_label = self.filter_by_ending(self.files, "cluster_KSLabel.tsv")[0]
            else:
                logger.info("Using bombcell cluster classification")
                cluster_label = cluster_label[0]
        if self.settings.cluster_labels == "kilosort":
            cluster_label = self.filter_by_ending(self.files, "cluster_KSLabel.tsv")[0]
            logger.info("Using kilosort cluster classification")
        self.cluster_group = np.loadtxt(cluster_label, dtype=str, delimiter="\t", skiprows=1)
        self.num_of_good_units = self.count_number_of_label_units("good")
        logger.info(f"The number of good units is: {self.num_of_good_units} out of {len(self.cluster_group)} units")
        num_mua = self.count_number_of_label_units("mua")
        logger.info(f"The number of mua is: {num_mua} out of {len(self.cluster_group)} units")

        # assert self.cluster_group[0][0] == "0", "The first cluster should be indexed by 0"  # sort check

        return Electrophysiology(
            spike_times=self.spike_times,
            spike_clusters=self.spike_clusters,
            cluster_group=self.cluster_group,
            cluster_labels=self.settings.cluster_labels,
            TTL_bin_path=self.imec_bin_path,
            number_of_good_units=self.num_of_good_units,
            imec_sync_path=self.imec_sync_path, 
        )


    def filter_by_ending(self, lst, ending):
        """
        Returns a list of strings from `lst` that end with the specified `ending`.
        """
        return [s for s in lst if s.endswith(ending)]

    def count_number_of_label_units(self, label):
        """
        A function that counts the number of units matching `label` in the cluster group file.
        Case-insensitive and whitespace-tolerant.
        """
        labels = np.char.lower(np.char.strip(self.cluster_group[:, 1].astype(str)))
        target = str(label).strip().lower()
        return np.count_nonzero(labels == target)

    def remove_duplicate_spike_times(self, spike_times, spike_clusters, censored_period_ms=0.1):
        """
        A function that removes duplicate spike times from the spike_times array.
        This is important because duplicate spike times can cause issues in downstream analyses.
        """
        # sort globally by time (recommended if not guaranteed sorted)
        order = np.argsort(spike_times, kind="mergesort")
        t = spike_times[order]
        c = spike_clusters[order]

        # parameters
        fs = 30000.0               # sampling rate Hz (set correctly!)
        thr = int(round(censored_period_ms * 1e-3 * fs))  # samples

        keep = np.ones(t.shape[0], dtype=bool)

        # remove duplicates within each cluster: keep first spike in refractory window
        dup_clusters = []
        dup_counts = []

        for clu in np.unique(c):
            idx = np.flatnonzero(c == clu)
            tt = t[idx]
            dt = np.diff(tt)
            dup = dt <= thr
            n_dup = int(np.count_nonzero(dup))
            if n_dup > 0:
                dup_clusters.append(int(clu))
                dup_counts.append(n_dup)
            keep[idx[1:][dup]] = False

        total_removed = int(np.sum(dup_counts))
        logger.info(
            f"Removed {total_removed} duplicate spikes across {len(dup_clusters)} clusters; "
            f"cluster_ids={dup_clusters}; duplicates_per_cluster={dup_counts}"
        )

        spike_times_clean = t[keep]
        spike_clusters_clean = c[keep]

        return spike_times_clean, spike_clusters_clean