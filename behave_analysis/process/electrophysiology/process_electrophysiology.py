import polars as pl
import numpy as np
from loguru import logger
import os

class ProcessedEfizz:
    def __init__(self, efizzDataLoaded, slope, samplingRate, filePath, lastPulse):
        logger.info("Processing Efizz Data via alignment to bonsai machine and creating a polars dataframe")
        self.samplingRate = samplingRate
        self.spike_times = efizzDataLoaded.spike_times
        self.spike_clusters = efizzDataLoaded.spike_clusters
        self.cluster_group = efizzDataLoaded.cluster_group
        self.filePath = filePath
        self.lastPulse = lastPulse
        
        initDF = self.generate_polar_dataframe()
        alignedDataFrame = self.align_spike_times(initDF, slope)
        
        print("MANNUAL CHECK: The resulting CSV to be saved - Does it look correct?")
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
        
        dataFrame = pl.DataFrame({"spike_times": self.spike_times.ravel(),
                                  "spike_clusters": self.spike_clusters,
                                 })
        
        # ADD CLUSTER GROUPS
        dataFrame = dataFrame.select(
            [
                pl.col("*"),  # select all
                (pl.col("spike_clusters").apply(lambda x: self.cluster_group[x][1])).alias("cluster_group"),
            ]
        )
        
        # UNIT TESTs
        assert len(dataFrame) == len(self.spike_times), "Dataframe not created correctly incorrect length"
        
        return dataFrame
    
    def align_spike_times(self, preAlignedDataFrame: object, slope: float) -> object:
        """
        Align the spike times to the bonsai machine, create a new column in the dataframe
        for the aligned spike times
        """
        self.alignedDataFrame = preAlignedDataFrame.select(
                                                [
                                                    pl.col("*"),  # select all
                                                    (pl.col("spike_times") * slope / self.samplingRate).alias("aligned_spike_times"),
                                                ] 
                                            )
        
        # UNIT TESTs
        print("MANNUAL CHECK: The number of null values in each column - This should be 0")
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