"""
This script provides functionality for analyzing and visualizing rayleigh data.
It includes methods to retrieve specific or all Rayleigh data based on experimental conditions and angles, compute the
delta of Rayleigh magnitude between pairs of conditions, and visualize these differences. The script primarily handles
data related to two distinct compartments in an experiment, referred to as 'shelter zone' and 'threat zone'.

# TODO;
- Remove comparison of all time and shelter present conditions in barrier settings, this results in zero delta
and is not useful and then is fed into pca so could be a problem
-- remove rubish cells that might be biasing the distribution, mabye cells that have a low magnitude in both conditions
maybe try a low rayleigh across both compartments and all conditions
"""

import os
import itertools

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scipy.stats as stats

from behave_analysis.utils.rayleigh.load_rayleigh import extract_rayleigh_path, load_rayleigh_data
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, identify_angles
from settings.settings_analyze_efizz import Settings_ae as Settings

def retrieve_specific_rayleigh_data(condition, angle, session, cluster_type, data_type="Rayleigh") -> dict:
    """
    Retrieve Rayleigh data for a specified condition and angle, distinguishing between shelter and threat zones.

    Parameters:
    - condition: The experimental condition.
    - angle: The angle of interest
    - session: Session identifier for the experiment.
    - cluster_type: Type of cluster being analyzed e.g good, mua, synthetic, syntheticHdir.
    - data_type (optional): Type of data to be retrieved, default is "Rayleigh".

    Returns:
    - A dictionary with keys one and two which are respectively'shelter_zone' and 'threat_zone', representing the respective compartment values for each cell.
    """
    path = extract_rayleigh_path(session, cluster_type, condition=condition, file_name=angle + "_Rayleigh.arrow")
    data = load_rayleigh_data(path)
    magnitude = extract_compartment_values(data, data_type)
    one_compartments = [x[0] for x in magnitude]
    two_compartments = [x[1] for x in magnitude]
    rayleigh_data = {"one": one_compartments, "two": two_compartments}
    return rayleigh_data


def retrieve_all_rayleigh_data(session, cluster_type) -> dict:
    """
    Aggregate Rayleigh data for all conditions and angles, organized by compartment zones.

    Parameters:
    - session: Session identifier for the experiment.
    - cluster_type: Type of cluster being analyzed.

    Returns:
    - A nested dictionary where the first key is the condition, the second key is the angle, and the third key represents
      either 'shelter_zone' or 'threat_zone', containing the respective compartment values for each cell.
      
    Should run with user_defined_conditions = True else plot will be too big
    """
    if Settings.user_defined_conditions:
        conditions = Settings.conditions
    else:
        conditions = identify_conditions(session)
    
    angles = identify_angles(session)
    dict_of_conditions = {}
    for condition in conditions:
        dict_of_angles = {}
        for angle in angles:
            dict_of_angles[angle] = retrieve_specific_rayleigh_data(condition, angle, session, cluster_type)
        dict_of_conditions[condition] = dict_of_angles

    # -------------- UNIT TESTS ----------------

    # Check that none of the values are the same
    for condition in conditions:
        for angle in angles:
            assert dict_of_conditions[condition][angle]["one"] != dict_of_conditions[condition][angle]["two"], "The compartment values are the same"

    return dict_of_conditions


def compute_rayleigh_delta_between_conditions(session, cluster_type) -> dict:
    """A function that computes the delta in rayleigh magnitude between pairs of conditions.

    Note that some conditions are the same and the delta will be zero this is expected behaviour,
    e.g. all time and shelter present are the same condition in some experiments

    Return a nested dictionary with:
    -- First key is the pair of conditions
    -- Second key is the angle
    -- Third key is the compartment ("one_delta" or "two_delta")
    """

    dict_of_conditions = retrieve_all_rayleigh_data(session, cluster_type)
    conditions = dict_of_conditions.keys()
    angles = identify_angles(session)

    # Create pairs of conditions
    unique_pairs = list(itertools.combinations(conditions, 2))

    # check unique pairs are unique
    assert len(unique_pairs) == len(set(unique_pairs)), "The unique pairs are not unique"

    # Loop through each pair of conditions and calculate the deltas
    deltas = {}
    for pair in unique_pairs:
        angles_dic = {}
        for angle in angles:
            angles_dic[angle] = {
                "one_delta": np.asarray(dict_of_conditions[pair[0]][angle]["one"]) - np.asarray(dict_of_conditions[pair[1]][angle]["one"]),
                "two_delta": np.asarray(dict_of_conditions[pair[0]][angle]["two"]) - np.asarray(dict_of_conditions[pair[1]][angle]["two"]),
            }
        deltas[pair] = angles_dic
    return deltas


def plot_rayleigh_deltas(session, cluster_type):
    """
    Create plots illustrating the delta in Rayleigh magnitude between pairs of conditions for shelter and threat zones.

    Parameters:
    - session: Session identifier for the experiment.
    - cluster_type: Type of cluster being analyzed.

    This function generates and saves plots that help in visual comparison of Rayleigh magnitude changes across
    different experimental conditions and angles.
    """

    deltas = compute_rayleigh_delta_between_conditions(session, cluster_type)
    angles = identify_angles(session)

    # Plotting logic
    nrows = len(angles) + 1
    ncols = len(deltas.keys()) + 1
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1] + [4] * (ncols - 1), height_ratios=[1] + [4] * (nrows - 1), wspace=0.3, hspace=0.6)
    fig = plt.figure(figsize=(30, 30))  # width, height
    axs_fontsize = 10
    labels = ["Shelter compartment", "Threat zone compartment"]
    col = ["cornflowerblue", "darkorchid"]
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(col, labels)]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=axs_fontsize, handlelength=4)

    # Add angle subtitles
    for a_counter, a in enumerate(angles):
        ax = plt.subplot(gs[a_counter + 1, 0])
        ax.text(0, 0.5, a, rotation="horizontal", va="center", ha="center", fontsize=axs_fontsize)
        ax.set_axis_off()

    # Add condition subtitles
    for c_counter, c in enumerate(deltas.keys()):
        ax = plt.subplot(gs[0, c_counter + 1])
        text = f"{c[0]} - {c[1]}"
        ax.text(0.5, 0.5, text, rotation=5.0, va="center", ha="center", fontsize=axs_fontsize)
        ax.set_axis_off()

    # plot hists
    for ai, angle in enumerate(angles):
        for i, pair in enumerate(deltas.keys()):
            ax = plt.subplot(gs[ai + 1, i + 1])
            ax.hist(deltas[pair][angle]["one_delta"], bins=100, color="cornflowerblue", alpha=0.7, label="Delta in shelter zone")
            ax.hist(deltas[pair][angle]["two_delta"], bins=100, color="darkorchid", alpha=0.7, label="Delta in threat zone")
            
            sample_distribution = deltas[pair][angle]["one_delta"]
            skewness = stats.skew(sample_distribution, nan_policy="omit")
            kurtosis = stats.kurtosis(sample_distribution, fisher=True, nan_policy="omit")
            
            secomd_sample_distribution = deltas[pair][angle]["two_delta"]
            skewness_2 = stats.skew(secomd_sample_distribution, nan_policy="omit")
            kurtosis_2 = stats.kurtosis(secomd_sample_distribution, fisher=True, nan_policy="omit")
            
            ax.set_title(f"Skewness: {skewness:.2f}, {skewness_2:.2f} Kurtosis: {kurtosis:.2f}, {kurtosis_2:.2f}", fontsize=10)
            
    fig.supxlabel("Delta in Rayleigh Magnitude")
    fig.supylabel("# of Cells")
    fig.suptitle(
        f"Delta in Rayleigh Magnitude for all angles for {cluster_type} cells. \nPositive values indicate a reduction in tuning. Negative values indicate an increase in tuning."
    )

    save_rayleigh_deltas_plots(session, cluster_type)
    plt.close()
    
    return deltas


def save_rayleigh_deltas_plots(session, cluster_type) -> None:
    """Save the rayleigh delta plots to the processed data folder"""
    file_path = os.path.join(session.base_path, session.processed_path, "models", "Rayleigh", cluster_type)
    file_name = f"{cluster_type}_rayleigh_delta_magnitudes.png"
    join = os.path.join(file_path, file_name)
    plt.savefig(join)
    logger.success("Change in rayleigh delta figures saved")
