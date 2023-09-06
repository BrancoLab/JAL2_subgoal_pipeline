import polars as pl
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import os

class ProcessedEfizz:
    def __init__(self, efizzDataLoaded, slope, intercept, samplingRate, filePath, camera_trigger, lastPulse, firstPulse):
        logger.info("Processing Efizz Data via alignment to bonsai machine and creating a polars dataframe")
        self.samplingRate = samplingRate
        self.spike_times = efizzDataLoaded.spike_times
        self.spike_clusters = efizzDataLoaded.spike_clusters
        self.cluster_group = efizzDataLoaded.cluster_group
        self.filePath = filePath
        self.camera_trigger = camera_trigger
        self.lastPulse = lastPulse
        self.firstPulse = firstPulse
        
        initDF = self.generate_polar_dataframe()
        alignedDataFrame = self.align_spike_times(initDF, slope, intercept)
        
        print("MANUAL CHECK: The resulting CSV to be saved - Does it look correct?")
        print(alignedDataFrame) # Print the dataframe to check it is correct
        
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
        spike_times_to_delete = np.hstack([np.where(self.spike_times < self.firstPulse)[0],np.where(self.spike_times > self.lastPulse)[0]])
        self.spike_times = np.delete(self.spike_times,spike_times_to_delete)
        self.spike_clusters = np.delete(self.spike_clusters,spike_times_to_delete)

        # CREATE DATAFRAME of SPIKE TIMES and CLUSTERS ids
        dataFrame = pl.DataFrame({"spike_times": self.spike_times.ravel(), "spike_clusters": self.spike_clusters.astype(np.int32)})
        
        # ADD CLUSTER GROUPS labels
        clusterLabelDataFrame = pl.DataFrame({"spike_clusters": self.cluster_group[:,0].astype(np.int32), "cluster_group": self.cluster_group[:,1]})
        
        # MERGE
        dataFrame = dataFrame.join(clusterLabelDataFrame, on = "spike_clusters")

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
        
        self.alignedDataFrame = self.alignedDataFrame.with_columns(self.alignedDataFrame['aligned_spike_times_in_samples'].cut(bins=self.camera_trigger, labels = [str(x) for x in np.arange(0,len(self.camera_trigger)+1)])['category'].alias('spike_aligned_to_frame'))
        self.alignedDataFrame = self.alignedDataFrame.select([pl.col('spike_aligned_to_frame').apply(float),pl.exclude('spike_aligned_to_frame')])

        # UNIT TESTs
        print("MANUAL CHECK: The number of null values in each column - This should be 0")
        print(self.alignedDataFrame.null_count())
        lastPulseTime = self.lastPulse / self.samplingRate
        assert lastPulseTime + 60 > self.alignedDataFrame["aligned_spike_times"].max(), "The last spike was recorded more than 60 seconds after the last TTL pulse, unlikely. Check the data."
        
        return self.alignedDataFrame
    
    def save_processed_efizz(self) -> None:
        """
        Save the processed efizz data
        """
        self.alignedDataFrame.write_csv(str(self.filePath) + "/" + "Processed_efizz_data", sep=",")
        logger.success("Processed Efizz data saved")
        
        # UNIT TESTS
        assert os.path.exists(str(self.filePath) + "/" + "Processed_efizz_data"), "Processed Efizz data not saved"