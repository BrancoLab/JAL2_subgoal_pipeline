import os

import polars as pl

from settings.settings_analyze_efizz import Settings_ae as settings

# -------------------------- Extract paths to feed into the loading functions --------------------------


def extract_arrow_files_from_a_condition(path_to_condition: str) -> list:
    """Extract paths to all arrow files from a condition"""
    arrow_files = []
    for root, dirs, files in os.walk(path_to_condition):
        for file in files:
            if file.endswith(".arrow"):
                arrow_files.append(os.path.join(root, file))
    return arrow_files


def extract_rayleigh_path(session: object, cluster_type: str, condition: str, file_name: str) -> str:
    """Extract paths to one arrow file for a condition"""

    con_dir = settings.condition_types

    path = os.path.join(
        session.base_path,
        session.processed_path,
        "models",
        "Rayleigh",
        cluster_type,
        con_dir,
        condition,
        file_name,
    )
    return path


# -------------------------- Provide paths to Rayleigh data and then load it --------------------------


def load_rayleigh_data(path_to_rayleigh: str) -> pl.DataFrame:
    """Load in a single Rayleigh related polars DataFrame"""
    return pl.read_ipc(path_to_rayleigh)


def load_all_rayleigh_data(paths_to_arrows: dict) -> dict:
    """Load all polars Rayleigh related polars DataFrame for a single condiiton

    I.e across all angles.

    Input:
    -- paths (dict) of all rayleigh data for each condition and each angle where paths are in a list
    each key is a condition and each value is a list of paths to arrow files

    Returns:
    -- data (dict) nested dictionary of all rayleigh data for each condition and each angle
    e.g {"All time": {"hdir.arrow": pl.dataframe, "hsa.arrow": pl.dataframe, ...}, ...}
    """
    condition_data = {}
    for condition in paths_to_arrows.keys():
        condition_data[condition] = {}
        for file in paths_to_arrows[condition]:
            basename = os.path.basename(file)
            condition_data[condition][basename] = load_rayleigh_data(file)
    return condition_data


# -------------------------- Bundle all of the data into a dictionary --------------------------


def collect_all_rayleigh_paths(session, cluster_type, conditions) -> dict:
    """Extract all rayleigh data from the database.

    Returns:
    -- paths (dict) of all rayleigh data for each condition and each angle where paths are in a list
    e.g {'all_time': ['E:\\efizz\\JAL004\\004_...eigh.arrow', 'E:\\efizz\\JAL004\\004_...eigh.arrow', ...],}

    """

    con_dir = "experimental_conditions"
    paths = {}
    for condition in conditions:
        # Get path to each condition
        path = os.path.join(
            session.base_path,
            session.processed_path,
            "models",
            "Rayleigh",
            cluster_type,
            con_dir,
            condition,
        )

        # Extract all angles from a condition and append to a list
        arrow_files = extract_arrow_files_from_a_condition(path_to_condition=path)
        arrow_data_list = []
        for file_name in arrow_files:
            arrow_data_list.append(extract_rayleigh_path(session, cluster_type, condition, file_name))

        # Add to dictionary
        paths[str(condition)] = arrow_data_list
    assert len(paths) == len(conditions), "Length of paths does not match length of conditions"
    return paths
