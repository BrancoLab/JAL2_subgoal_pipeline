import polars as pl
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import os
import inspect


class ProcessedEfizz:
    def __init__(self, efizzDataLoaded, slope, intercept, samplingRate, filePath, camera_trigger, lastPulse, firstPulse):
        logger.info("Processing Efizz Data via alignment to bonsai machine and creating a polars dataframe")
        self.samplingRate = samplingRate
        self.spike_times = efizzDataLoaded.spike_times
        self.spike_clusters = efizzDataLoaded.spike_clusters
        self.cluster_group = efizzDataLoaded.cluster_group
        self.cluster_labels = efizzDataLoaded.cluster_labels
        self.filePath = filePath
        self.camera_trigger = camera_trigger
        self.lastPulse = lastPulse
        self.firstPulse = firstPulse

        initDF = self.generate_polar_dataframe()
        alignedDataFrame = self.align_spike_times(initDF, slope, intercept)

        print("MANUAL CHECK: The resulting CSV to be saved - Does it look correct?")
        print(alignedDataFrame)  # Print the dataframe to check it is correct

        self.save_processed_efizz()
        logger.success("Efizz data loaded and processed")

    def generate_polar_dataframe(self) -> object:
        """
        Reshapes numpy arrays into a polars dataframe

        Returns:
            object: Polars dataframe
        """
        assert self.spike_times.shape[0] == self.spike_clusters.shape[0], "Spike times and clusters are not the same shape this can't be"

        # delete all spike times that come before the first pulse or after the last one
        spike_times_to_delete = np.hstack([np.where(self.spike_times < self.firstPulse)[0], np.where(self.spike_times > self.lastPulse)[0]])
        self.spike_times = np.delete(self.spike_times, spike_times_to_delete)
        self.spike_clusters = np.delete(self.spike_clusters, spike_times_to_delete)

        # CREATE DATAFRAME of SPIKE TIMES and CLUSTERS ids
        dataFrame = pl.DataFrame({"spike_times": self.spike_times.ravel(), "spike_clusters": self.spike_clusters.astype(np.int32)})

        # ADD CLUSTER GROUPS labels
        clusterLabelDataFrame = pl.DataFrame({"spike_clusters": self.cluster_group[:, 0].astype(np.int32), "cluster_group": self.cluster_group[:, 1]})

        # MERGE
        dataFrame = dataFrame.join(clusterLabelDataFrame, on="spike_clusters")

        # UNIT TESTs
        assert len(dataFrame) == len(self.spike_times), "Dataframe not created correctly incorrect length"

        return dataFrame

    def align_spike_times(self, preAlignedDataFrame: object, slope: float, intercept) -> object:
        """
        Align the spike times to the bonsai machine, create a new column in the dataframe
        for the aligned spike times
        """
        self.alignedDataFrame = preAlignedDataFrame.select(
            [
                pl.col("*"),  # select all
                (((pl.col("spike_times") * slope) + intercept) / self.samplingRate).alias("aligned_spike_times"),
                (((pl.col("spike_times") * slope) + intercept)).alias("aligned_spike_times_in_samples"),
            ]
        )

        frame_labels = [str(x) for x in np.arange(0, len(self.camera_trigger) + 1)]

        if "bin" in inspect.getfullargspec(pl.Series.cut).args:  # older Polars version
            logger.warning("You are using an old version of Polars that is deprecated but this code should still work for now")
            self.alignedDataFrame = self.alignedDataFrame.with_columns(
                pl.col("aligned_spike_times_in_samples")
                .cut(bins=self.camera_trigger, labels=frame_labels)["category"]
                .cast(pl.Utf8)
                .cast(pl.Float64, strict=False)
                .alias("spike_aligned_to_frame")
            )
        elif "breaks" in inspect.getfullargspec(pl.Series.cut).args:  # jasmine's more recent polars version
            logger.warning("Congrats! You have an updated version of polars but we're not super confident this data alignment thing works")
            self.alignedDataFrame = self.alignedDataFrame.with_columns(
                pl.col("aligned_spike_times_in_samples")
                .cut(breaks=self.camera_trigger, labels=frame_labels)
                .cast(pl.Utf8)
                .cast(pl.Float64, strict=False)
                .alias("spike_aligned_to_frame")
            )

        # UNIT TESTs
        print("MANUAL CHECK: The number of null values in each column - This should be 0")
        print(self.alignedDataFrame.null_count())
        lastPulseTime = self.lastPulse / self.samplingRate
        assert (
            lastPulseTime + 60 > self.alignedDataFrame["aligned_spike_times"].max()
        ), "The last spike was recorded more than 60 seconds after the last TTL pulse, unlikely. Check the data."

        return self.alignedDataFrame

    def save_processed_efizz(self) -> None:
        """
        Save the processed efizz data
        """
        qualifier = "_bc" if self.cluster_labels == "bombcell" else ""
        if "sep" in inspect.getfullargspec(pl.DataFrame.write_csv).kwonlyargs:
            self.alignedDataFrame.write_csv(str(self.filePath) + "/" + "Processed_efizz_data" + qualifier, sep=",")
            logger.success("Processed Efizz data saved")
        elif "separator" in inspect.getfullargspec(pl.DataFrame.write_csv).kwonlyargs:
            self.alignedDataFrame.write_csv(str(self.filePath) + "/" + "Processed_efizz_data" + qualifier, separator=",")
            logger.success("Processed Efizz data saved")
        else:
            logger.warning("DF saving was not successful because of polars version issues")

        # UNIT TESTS
        assert os.path.exists(str(self.filePath) + "/" + "Processed_efizz_data" + qualifier), "Processed Efizz data not saved"
