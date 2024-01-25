"""A script to programmatically filter and classify head shelter direction cells
based on the following criteria:

1) The angle of the rayleigh vector between compartments must be similar in the shelter condition
2) The rayleigh score must be above a threshold in the shelter condition in both compartments
3) The cluster id must not be in the head direction cell list
"""

import numpy as np
import os
import dill as pickle
from loguru import logger

from behave_analysis.utils.rayleigh.load_rayleigh import extract_rayleigh_path, load_rayleigh_data
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values

# User defined constants
RAYLEIGH_THRESHOLD = 0.2
SIMILAR_ANGLE_THRESHOLD = 0.8


def classify_hsa(session: object, cluster_type: str, hdir_cells: list) -> list:
    """Label cells as head shelter based on a set of criteria

    Inputs:
    -- hdir_cells: list of cell ids that are head direction cells in
    order to filter them out of the head shelter cells

    Returns:
    -- cell ids (list) that are head shelter direction cells"""

    path = extract_rayleigh_path(session, cluster_type, condition="shelter_only", file_name="hsa_Rayleigh.arrow")
    data = load_rayleigh_data(path)

    angles = extract_compartment_values(data, "Rayleigh_theta")
    magnitude = extract_compartment_values(data, "Rayleigh")
    sig = extract_compartment_values(data, "Rayleigh_sig")

    head_shelter_cells = []
    for row, cell_id in enumerate(data["clusterID"]):
        simlar_angles = angle_similarity(angles[row][0], angles[row][1])
        above_threshold = rayleigh_threshold(magnitude[row][0], magnitude[row][1])
        sig_bool = check_both_compartments_significant(sig[row])
        if simlar_angles > SIMILAR_ANGLE_THRESHOLD and above_threshold and sig_bool:
            head_shelter_cells.append(cell_id)

    hsa_cells = [cell for cell in head_shelter_cells if cell not in hdir_cells]

    save_cell_ids(session, hsa_cells)

    return hsa_cells


def angle_similarity(theta1: float, theta2: float) -> float:
    """
    Compute the similarity score between two angles in radians.

    Parameters:
    theta1 (float): First angle in radians.
    theta2 (float): Second angle in radians.

    Returns:
    float: Similarity score between 0 and 1.
    Smaller values indicate similar angles
    """
    # Calculate the absolute difference
    diff = np.abs(theta1 - theta2)

    # Adjust for the fact that 0 and 2pi are the same
    adjusted_diff = min(diff, 2 * np.pi - diff)
    assert adjusted_diff <= np.pi, "Adjusted difference is greater than pi"

    # Normalize the difference to get the similarity score
    similarity_score = (np.pi - adjusted_diff) / np.pi

    assert similarity_score >= 0 and similarity_score <= 1, "Similarity score is not between 0 and 1"

    return similarity_score


def rayleigh_threshold(mag1: float, mag2: float) -> bool:
    """Return True if both magnitudes are above the threshold"""
    return mag1 > RAYLEIGH_THRESHOLD and mag2 > RAYLEIGH_THRESHOLD


def check_both_compartments_significant(sig: tuple) -> bool:
    """Return True if both compartments are significant"""
    return sig[0] and sig[1]


def save_cell_ids(session, cell_ids) -> None:
    """Saves the cell ids to a pickle file to a within a folder called cells"""
    path = make_directory(os.path.join(session.base_path, session.processed_path, "cells"))
    file_name = os.path.join(path, "hsa_cells.pkl")
    with open(file_name, "wb") as dill_file:
        pickle.dump(cell_ids, dill_file)
    logger.success("HSA cell ids saved")
