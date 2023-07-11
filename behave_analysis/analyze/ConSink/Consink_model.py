# OS libaries
from loguru import logger
import numpy as np
import os
import polars as pl
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# import functions
from behave_analysis.visualize.visualize_efizz import filter_video_dataframe, generate_bin_angles 

def Consink(self):
    """
    Generate consink maps as in Ormond & O'Keefe
    """
    bin_num = 10

    # tile the arena

    # calculate head-to-sink angle at each position
    # m x n matrix (m = each spatial tile (where the mouse is), n = head angle to each sink)

    # bin the head-to-sink angle

    # only include tiles where each head-to-sink angle has at least 1s of time

    # for each tile, calculate rayleigh vector