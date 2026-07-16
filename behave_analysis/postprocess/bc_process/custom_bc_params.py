"""A script where we can set up our custom bombcell params and save them to a set location on ceph to be used for running bombcell"""
import numpy as np

param = {}

param["reextractRaw"] = False

# noise params
param["spDecayLinFit"] = False # don't use linear fit, use exponential fit instead (recommended)
param["computeSpatialDecay"] = True

param['presenceRatioBinSize'] = 300 # default 60s, our recordings are so long...

# 1. classification thresholds like: 
# for noise
param["maxWvBaselineFraction"] = .2
param["maxWvDuration"] = 750

# for mua
param["minAmplitude"] = 10
param["minSNR"] = 10
param["maxRPVviolations"] = .08
param["maxPercSpikesMissing"] = 10

#  2. or which quality metrics are computed (by default these are not): 


#  3. how quality metrics are calculated:

# a. Refractory period violation (RPV) method - choose one of:
#    'hill' (default): Hill et al. method
#    'llobet': Llobet et al. method, llobet more stringent, relevant for cells that have FR>30, EXTREMELY SLOW
param["rpv_method"] = "hill" # can we do llobet on server?

# b. Refractory period values to test (in seconds)
#    For a single value: np.array([0.002])

# c. Censored period (in seconds) - ISIs below this are excluded as duplicates
# param["tauC"] = 0.0001  # 0.1ms

# e. Whether the recording is split into time chunks to determine "good" time chunks: 
# param["computeTimeChunks"] = 0

# full list in the wiki or in the bc.get_default_parameters function

np.save(r"Z:\Jasmine_Laurence\bombcell\bombcell_params.npy", param, allow_pickle=True)
