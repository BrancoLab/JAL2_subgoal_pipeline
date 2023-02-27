import polars as pl
import numpy as np
from loguru import logger

class ProcessedEfizz:
    def __init__(self, efizzDataLoaded, slope, samplingRate):
        logger.info("Processing Efizz Data via alignment to bonsai machine and creating a polars dataframe")
        self.samplingRate = samplingRate
        self.spike_times = efizzDataLoaded.spike_times
        self.spike_clusters = efizzDataLoaded.spike_clusters
        
        initDF = self.generate_polar_dataframe()
        alignedDataFrame = self.align_spike_times(initDF, slope)
        print(alignedDataFrame)
        logger.success("Efizz data loaded and processed")
    
    def generate_polar_dataframe(self) -> object:
        """
        Reshapes numpy arrays into a polars dataframe

        Returns:
            object: Polars dataframe
        """
        data = np.hstack((self.spike_times, self.spike_clusters)).T # Reshape Required for polars ingestion
        dataFrame = pl.from_numpy(data, schema=["spike_samples", "spike_clusters"], orient="col")
        
        return dataFrame
    
    def align_spike_times(self, preAlignedDataFrame: object, slope: float) -> object:
        """
        Align the spike times to the bonsai machine, create a new column in the dataframe
        for the aligned spike times
        """
        alignedDataFrame = preAlignedDataFrame.select(
                                                [
                                                    pl.col("*"),  # select all
                                                    (pl.col("spike_samples") * slope / self.samplingRate).alias("aligned_spike_times"),
                                                ] 
                                            )
        
        return alignedDataFrame