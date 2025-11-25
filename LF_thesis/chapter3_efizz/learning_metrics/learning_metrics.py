# Learning metrics to extract per session, linked to the conditions of escapes and homings
# Number of homings - how many homings occured per session in each condition
# Spatial efficiency of escapes - how direct were the escapes in each condition - e.g we could get the average of all escapes in each condition
# Spatial efficiency of homings - how direct were the homings in each condition - e.g we could get the average of all homings in each condition
# Average speed of escapes and homings in each condition
# Rate of homings across some time window - e.g number of homings in each condition per 10 minutes

import os
import pickle

from behave_analysis.process.session import get_experiment
from loguru import logger

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr, JAL7_30apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip4_10may, JAL8_14may, JAL8_21may

experiments_objects = [
    JAL6_flip7_1apr,
    JAL6_flip3_18mar,
    JAL6_flip4_21mar,
    JAL6_flip5_25mar,
    JAL6_28mar,
    JAL3_25aug,
    JAL3_1sept,
    JAL3_4sept,
    JAL3_7sept,
    JAL005_8thSept,
    JAL005_21stSept,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
    JAL7_flip5_22mar,
    JAL7_flip2_12mar,
    JAL7_23apr,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_flip4_10may,
    JAL8_14may,
    JAL4_3rdSept,
    JAL4_19thSept,
    JAL4_28aug,
    JAL4_11thSept,
]

session_NAMES = [
    "JAL6_flip7_1apr",
    "JAL6_flip3_18mar",
    "JAL6_flip4_21mar",
    "JAL6_flip5_25mar",
    "JAL6_28mar",
    "JAL3_25aug",
    "JAL3_1sept",
    "JAL3_4sept",
    "JAL3_7sept",
    "JAL005_8thSept",
    "JAL005_21stSept",
    "JAL7_sesh8_9apr",
    "JAL7_sesh9_16apr",
    "JAL7_flip5_22mar",
    "JAL7_flip2_12mar",
    "JAL7_23apr",
    "JAL8_flip1_25apr",
    "JAL8_flip2_29apr",
    "JAL8_flip4_10may",
    "JAL8_14may",
    "JAL4_3rdSept",
    "JAL4_19thSept",
    "JAL4_28aug",
    "JAL4_11thSept",
]


# Function to load homings object
def load_homings_object(session_path):
    """Load homings object from a session"""
    homings_path = os.path.join(session_path, "homings", "homings_obj.pkl")

    if os.path.exists(homings_path):
        try:
            with open(homings_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading homings object from {session_path}: {e}")

    return None


def load_escapes_object(session_path):
    """Load escapes object from a session, checking multiple possible paths"""
    possible_paths = [
        os.path.join(session_path, "escapes", "escapes_obj.pkl"),
        os.path.join(session_path, "escape", "escapes_obj.pkl"),
        os.path.join(session_path, "escapes_obj.pkl"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading escapes object from {path}: {e}")

    return None


# Function to count homings and escapes

from collections import defaultdict
dictionary_results = defaultdict(dict)

conditions = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]

for i, session in enumerate(experiments_objects):
    session_name = session_NAMES[i]
    logger.info(f"Processing session: {session_name}")
    loaded_session = get_experiment(session)
    session_path = os.path.join(loaded_session.base_path, loaded_session.processed_path)

    # Get homings object and metrics
    homings_obj = load_homings_object(session_path)
    if homings_obj is None:
        print(f"No homings object found for session {session_NAMES[i]}. Skipping.")
        continue
    number_of_homings = len(homings_obj.onset_frames)  # scalar
    avg_speed_homings = homings_obj.avg_speed  # (, len) - float
    spatial_efficiency_homings = homings_obj.spatial_efficiency  # (, len) - float
    conditions_homings = homings_obj.homing_condition  # (, len) - string
    timings_homings = [x / 40 for x in homings_obj.onset_frames]  # (, len) - float

    # Get escapes object and metrics
    escapes_obj = load_escapes_object(session_path)
    if escapes_obj is None:
        print(f"No escapes object found for session {session_NAMES[i]}. Skipping.")
        continue
    number_of_escapes = len(escapes_obj.stim_onset_frames)  # scalar
    escape_speeds = escapes_obj.avg_speed  # (, len) - float
    spatial_efficiency_escapes = escapes_obj.spatial_efficiency  # (, len) - float
    conditions_escapes = escapes_obj.escape_condition  # (, len) - string
    timings_escapes = [x / 40 for x in escapes_obj.stim_onset_frames]  # (, len) - float

    # Store results in dictionary
    dictionary_results[session_name]["number_of_homings"] = number_of_homings # a scalar number of homings
    dictionary_results[session_name]["avg_speed_homings"] = avg_speed_homings # a list of average speeds for each homing
    dictionary_results[session_name]["spatial_efficiency_homings"] = spatial_efficiency_homings # a list of spatial efficiencies for each homing
    dictionary_results[session_name]["conditions_homings"] = conditions_homings # a list of conditions for each homing (e.g "shelter_only", "barrier_pre_flip", "barrier_post_flip")
    dictionary_results[session_name]["timings_homings"] = timings_homings # a list o ftimings in seconds when each homing started
    
    dictionary_results[session_name]["number_of_escapes"] = number_of_escapes # a scalar number of escapes
    dictionary_results[session_name]["escape_speeds"] = escape_speeds # a list of average speeds for each escape
    dictionary_results[session_name]["spatial_efficiency_escapes"] = spatial_efficiency_escapes # a list of spatial efficiencies for each escape
    dictionary_results[session_name]["conditions_escapes"] = conditions_escapes # a list of conditions for each escape (e.g "shelter_only", "barrier_pre_flip", "barrier_post_flip")
    dictionary_results[session_name]["timings_escapes"] = timings_escapes # a list of timings in seconds when each escape started