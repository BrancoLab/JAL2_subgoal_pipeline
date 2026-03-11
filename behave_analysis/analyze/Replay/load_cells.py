import numpy as np
from loguru import logger
import dill as pickle
import os

from behave_analysis.analyze.EscapePattern.ComputeEscapeTuning import load_or_compute_escape_tuning

def load_hdir_cells(session):
    """Loads in the pickle with all the hdir cells and the good cluster Ids from the sessions in the experiments_objects list.
    INPUTS:
        session: session for which to load hdir cells
    RETURNS:
        hdir: list of which good clusters are hdir cells 
    """
    file_name = os.path.join(session.base_path, session.processed_path, "cells", "hdir_cells.pkl")
    
    # assert os.path.exists(file_name), "No hdir_cells.pkl file found. Please run the head direction classification first."
    try:
        with open(file_name, "rb") as dill_file:
            hdir = pickle.load(dill_file)

    except FileNotFoundError:
        logger.warning("Hdir file not found, returning empty list.")
        hdir = []

    return hdir


def load_escape_tuned_cells(aefizz):
    """Load in the significant cells for each experiment and return a matrix of significant cells
    INPUTS:
        aefizz: AnalyzeEfizz object
    RETURNS:
        xval: boolean array of shape (num_cells, 3) indicating significant escape tuned cells for each condition"""

    var = "escape"
    time_period = "homing&escape"

    # 1. load in escape homing/escape tuning curve
    CT = load_or_compute_escape_tuning(aefizz, var + ' in ' + time_period)
    # identify cells that are sig tuned to %escape in homing/escape
    shift0 = int(np.shape(CT.y_fitted_shift)[0] / 2)
    sig_escape = CT.params_shifts[shift0, :, :, 0] > np.nanpercentile(CT.params_shifts[:, :, :, 0], 95, axis=0)

    # 2. load in residuals data
    var = "residual: escape in homing&escape - bird_dist_shelter in exploration"
    CT = load_or_compute_escape_tuning(aefizz, var)
    # find cells whose residual tuning to %escape - distance to shelter in exploration is significant
    shift0 = int(np.shape(CT.y_fitted_shift)[0] / 2)
    sig_res = CT.params_shifts[shift0, :, :, 0] > np.nanpercentile(CT.params_shifts[:, :, :, 0], 95, axis=0)

    """Select the cells I want to analyse"""
    # cells that are tuned to %escape in homing/escape (subselect ones that are not tuned to distance to shelter in exploration and passed the residual test)
    xval = np.full_like(sig_escape, np.nan)
    for c in range(3):

        # NB: the original definition of significant used for CoSyNe
        # A = (sig_escape[:, c] == True) & (exp_sig_dist[:, c] == False) & (sig_res[:, c] == False)  # V1 only
        # AC = (sig_escape[:, c] == True) & (sig_res[:, c] == True)  # Both V1 and V1 regressed
        # xval[:,c] = (A == True) | (AC == True)

        xval[:, c] = (sig_escape[:, c] == True) & (sig_res[:, c] == True)  # Both V1 and V1 regressed

    return xval
