"""A script to programmatically filter and classify head direction cells
based on the following criteria:

1) The angle of the rayleigh vector between compartments must be similar
thus ensuring the firing fields are stable across compartments for all time

2) Must be significant for all time in both compartments

NOTE: This script can be used to find the bi directional split cells by inversing the similarity threshold
i.e angles must be as different as possible with a high rayleigh score in both compartments

TODO: Include the output of the Tuned Model

"""

import os

import polars as pl
import numpy as np
from loguru import logger

from settings.settings_analyze_efizz import Settings_ae as Settings

# User defined constants
RAYLEIGH_THRESHOLD = 0.5
SIMILAR_ANGLE_THRESHOLD = 0.8


def classify_hdir(session: object, cluster_type: str) -> list:
    """Label cells as head direction based on a set of criteria

    Returns:
    -- cell ids (list) that are head direction cells"""
    path = extract_rayleigh_path(session, cluster_type)
    data = load_rayleigh_data(path)

    angles = extract_compartment_values(data, "Rayleigh_theta")
    magnitude = extract_compartment_values(data, "Rayleigh")
    sig = extract_compartment_values(data, "Rayleigh_sig")

    head_direction_cells = []
    for row, cell_id in enumerate(data["clusterID"]):
        simlar_angles = angle_similarity(angles[row][0], angles[row][1])
        above_threshold = rayleigh_threshold(magnitude[row][0], magnitude[row][1])
        sig_bool = check_both_compartments_significant(sig[row])
        if simlar_angles > SIMILAR_ANGLE_THRESHOLD and above_threshold and sig_bool:
            head_direction_cells.append(cell_id)

    logger.info(f"Found {len(head_direction_cells)} head direction cells")

    return data


def check_both_compartments_significant(sig: tuple) -> bool:
    """Return True if both compartments are significant"""
    return sig[0] and sig[1]


def extract_compartment_values(data, column_name: str) -> tuple:
    """Extract compartment values from a polars DataFrame

    Returns:
    -- compartment values (tuple) for each cell e.g ((x1, y1), (x2, y2), ...
    first value is shelter zone, second value is threat zone"""
    first = [x[0] for x in data[column_name]]
    second = [x[1] for x in data[column_name]]
    output = tuple(zip(first, second))
    assert len(output) == len(
        data
    ), "Length of extracted compartment values does not match length of data"
    return output


def extract_rayleigh_path(session: object, cluster_type: str) -> str:
    """Extract paths to Rayleigh data"""
    path = os.path.join(
        session.base_path,
        session.processed_path,
        "models",
        "Rayleigh",
        cluster_type,
        "all_time",
        "hdir_Rayleigh.arrow",
    )
    return path


def load_rayleigh_data(path_to_rayleigh: str) -> pl.DataFrame:
    """Load in Rayleigh data as polars DataFrame"""
    return pl.read_ipc(path_to_rayleigh)


def rayleigh_threshold(mag1: float, mag2: float) -> bool:
    """Return True if both magnitudes are above the threshold"""
    return mag1 > RAYLEIGH_THRESHOLD and mag2 > RAYLEIGH_THRESHOLD


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

    assert (
        similarity_score >= 0 and similarity_score <= 1
    ), "Similarity score is not between 0 and 1"

    return similarity_score
