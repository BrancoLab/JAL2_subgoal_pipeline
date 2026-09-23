from behave_analysis.analyze.EscapePattern.ComputeEscapeTuning import ComputeEscapeTuning
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import compute_tuning_stat

import numpy as np
import os
from pathlib import Path


def find_sig_tuned_cells_after_residual(session, settings, variable, vars_to_subtract, remove_hdir, stat="zscore_peak"):
    """INPUTS:
    session: dict, session info
    variable: str, the variable to analyze, formatted as <variables> in <time_period> (e.g. "frac_route in escape")
    vars_to_subtract: list of str, the variables to subtract from the main variable, formatted as <variables> in <time_period> (e.g. "speed in explore")
    stat: str, the statistic to use for the analysis"""

    complete_var = [variable] + [f"residual: {variable} - {var_to_subtract}" for var_to_subtract in vars_to_subtract]

    sig = {v: [] for v in complete_var}
    sig_cells_dict = {}

    for v in complete_var:
        computeET = ComputeEscapeTuning(tuning=v, session=session, settings=settings, aefizz=None)

        if computeET.do_analysis:
            print("analysis not done yet for this session {0} for variable {1}! Sad".format(session["name"], v))
            if v == variable:
                print("skipping session {0} for variable {1} because analysis not done yet!".format(session["name"], v))
                return sig_cells_dict
            continue

        if computeET.database[computeET.database["name"] == computeET.hexaname]["insufficient_data"].values[0]:
            print("Skipping session {0} due to insufficient data with hexaname {1}".format(session["name"], computeET.hexaname))
            return sig_cells_dict

        print("loading data for session {} with hexaname {}".format(session["name"], computeET.hexaname))
        data = computeET.load_results()
        shift0 = int(np.shape(data["y_fitted_shift"])[0] / 2)
        real_stat, shift_stat = compute_tuning_stat(
            stat=stat, shifted_tuning_curves=data["fr_shift"], shift0=int(np.shape(data["fr_shift"])[0] / 2), neural_matrix=data["neural_matrix"], condition=data["condition_vector"]
        )
        sig_escape = real_stat > np.nanpercentile(shift_stat, 95, axis=0)
        sig[v] = sig_escape

    sig_remaining_sig = sig[complete_var[0]] & sig[complete_var[1]] & sig[complete_var[2]] & sig[complete_var[3]]

    # load hdir
    cluster_Ids = np.load(str(os.path.join(session["base_path"], session["processed_path"]) + "/" + "good_cluster_Ids__bc.npy"))
    hdir = np.full(len(cluster_Ids), False)
    if remove_hdir:
        path = os.path.join(session["base_path"], session["processed_path"], "cells")
        hdir_path = list(Path(path).glob("*_hdir_cells.npy"))[0]
        hdir_cells = np.load(hdir_path)
        hdir = np.full(len(cluster_Ids), False)
        for hc in hdir_cells:
            hdir[cluster_Ids == hc] = True

    for c_idx, c in enumerate(computeET.ET.all_conditions):
        sig_cells_dict[c] = sig_remaining_sig[c_idx, :] & ~hdir

    return sig_cells_dict
