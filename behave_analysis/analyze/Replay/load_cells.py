import numpy as np
from pathlib import Path
import dill as pickle
import os

from behave_analysis.process.process import Process
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import load_or_compute_escape_tuning


def load_hdir_cells(experiments_objects, session_names):
    """Loads in the pickle with all the hdir cells and the good cluster Ids from the sessions in the experiments_objects list.
    INPUTS:
        experiments_objects: list of session objects
        session_names: list of session names (must match the experiments_objects list)
    RETURNS:
        hdir_sesh: list of which good clusters are hdir cells for each session
    """
    dir = Path("Z:\Jasmine_Laurence\single_trial_overview\decoding_spatial_efficiency\head_direction_cells.pkl")
    with open(dir, "rb") as dill_file:
        hdir = pickle.load(dill_file)

    hdir_sesh = []

    for idx, exp in enumerate(experiments_objects):
        session = Process(exp).load_session()
        clu_Ids = np.load(os.path.join(session.base_path, session.processed_path) + "\\" + "good_cluster_Ids.npy")

        hdir_n = hdir[session_names[idx]]
        hdir_sesh.append([int(np.where(clu_Ids == int(h))[0][0]) for h in hdir_n])
        
    return hdir_sesh


def load_escape_tuned_cells(aefizz):
    """Load in the significant cells for each experiment and return a matrix of significant cells
    INPUTS:
        aefizz: AnalyzeEfizz object
    RETURNS:
        xval: boolean array of shape (num_cells, 3) indicating significant escape tuned cells for each condition"""

    var = "escape"
    time_period = "homing&escape"

    # 1. load in escape homing/escape tuning curve
    CT = load_or_compute_escape_tuning(aefizz, aefizz.settings.escape_tuning_bins, var, time_period)
    # identify cells that are sig tuned to %escape in homing/escape
    shift0 = int(np.shape(CT.y_fitted_shift)[0] / 2)
    sig_escape = CT.params_shifts[shift0, :, :, 0] > np.nanpercentile(CT.params_shifts[:, :, :, 0], 95, axis=0)

    # 2. load in residuals data
    var = "residual_escape"
    CT = load_or_compute_escape_tuning(aefizz, aefizz.settings.escape_tuning_bins, var, time_period)
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
