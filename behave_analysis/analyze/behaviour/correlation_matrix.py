"""A function that takes in video_df and a list of variables
and builds a correlation matrix between them"""

from loguru import logger
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import polars as pl
import os
from astropy.stats import circcorrcoef

from behave_analysis.analyze.EscapePattern.escape_pattern_utils import homing_escape_onsets, create_discretized_behave_var

def compute_correlation_matrix(abehave, variables):
    # 1. filter video_df to restrict to homing&escape periods
    onset_dict = homing_escape_onsets(abehave, "homing&escape")
    ons, offs = onset_dict["ons"], onset_dict["offs"]
    
    # find shelter entries after escape onset
    shelter_entries = np.where(np.diff(abehave.video_df["OutofshelterIdx"].to_numpy().astype(int)) == -1)[0] + 1  # +1 to get the entry frame
    # if mouse enters shelter before offset of homing/escape, set offset to shelter entry
    time_mask = np.full(abehave.video_df.shape[0], False)
    for on, off in zip(ons, offs):
        entry_after_escape = shelter_entries[shelter_entries > int(on)]
        if len(entry_after_escape) == 0:
            time_mask[int(on) : int(off)] = True
            continue
        if entry_after_escape[0] < off:  # only consider shelter entries within 20s of escape onset or within stimulus duration
            off = entry_after_escape[0]
        time_mask[int(on) : int(off)] = True
    
    # 2. extract or compute the variables of interest and store in a new dataframe
    var_matrix = np.zeros((len(variables), time_mask.sum()))
    var_properties = []
    
    if isinstance(variables, str):
        variables = [variables]

    for i, var in enumerate(variables):
        if var == 'speed':
            var_matrix[i,:] = abehave.video_df.filter((time_mask))["speed"].to_numpy()
            var_properties.append("linear")
        elif var in ['bird_dist_shelter', 'position', 'escape']:
            if var == 'position': logger.warning("Using bird_dist_shelter as 1D proxy for position.")
            condition_vector = np.zeros(np.sum(time_mask), dtype=int)
            condition_vector[abehave.video_df.filter((time_mask))["barrier_present"].to_numpy().astype(bool)] += 1
            condition_vector[abehave.video_df.filter((time_mask))["barrier_flipped"].to_numpy().astype(bool)] += 1
            var_matrix[i,:] = create_discretized_behave_var(abehave, 
                                                            x = abehave.video_df.filter((time_mask))["mouse_x_position"].to_numpy(), 
                                                            y = abehave.video_df.filter((time_mask))["mouse_y_position"].to_numpy(),
                                                            condition = condition_vector,
                                                            tuning_var = var,
                                                            time_mask_vector=time_mask,
                                                            discretize=False)
            var_properties.append("linear")
        elif var in ['hdir', 'hsa', 'h_preflipbar_a', 'h_postflipbar_a', 'h_bar_centre_a']:
            var_matrix[i,:] = abehave.video_df.filter((time_mask))[var].to_numpy()
            var_properties.append("circular")
        else:
            logger.warning("Variable {} not implemented yet".format(var))

    # 3. compute the correlation matrix between the variables
    # if two linear variables, use spearman correlation
    # if one of the variables is circular, use circular-linear correlation
    # if both variables are circular, use circorrcoeff
    if len(variables) == 1:
        logger.warning("Not implemented: Compute autocorrelation for variable: {}".format(variables[0]))
    corr_matrix = np.zeros((len(variables), len(variables)))
    for i in range(len(variables)):
        for j in range(len(variables)):
            if var_properties[i] == "linear" and var_properties[j] == "linear":
                corr_matrix[i,j], _ = spearmanr(var_matrix[i,:], var_matrix[j,:])
            elif var_properties[i] == "circular" and var_properties[j] == "circular":
                corr_matrix[i,j] = circcorrcoef(var_matrix[i,:], var_matrix[j,:])
            else:
                # determine which variable is circular and which is linear
                if var_properties[i] == "circular":
                    theta, x = var_matrix[i,:], var_matrix[j,:]
                else:
                    theta, x = var_matrix[j,:], var_matrix[i,:]
                corr_matrix[i,j] = circular_linear_corr(theta, x)

    return corr_matrix, var_matrix


def plot_correlation_matrix(corr_matrix, variables):
    plt.figure(figsize=(4, 3))
    plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(ticks=np.arange(len(variables)), labels=variables, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(variables)), labels=variables)
    plt.tight_layout()
    plt.show()

def plot_vars_over_time(var_matrix, variables, t_range=[0,1200]):
    fig, axs = plt.subplots(len(variables), 1, figsize=(8, len(variables)), sharex=True)
    for i in range(var_matrix.shape[0]):
        axs[i].plot(var_matrix[i,t_range[0]:t_range[1]], 'k')
        axs[i].set_ylabel(variables[i])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].set_xlim(0, t_range[1]-t_range[0])
        axs[i].set_xticks(np.arange(0, t_range[1]-t_range[0], 200))
        axs[i].set_xticklabels(np.arange(0, (t_range[1]-t_range[0])/40, 5).astype(int))
        if i == var_matrix.shape[0]-1:
            axs[i].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

def circular_linear_corr(theta, x):
    """Circular-linear correlation (Mardia & Jupp, 2000)

    Parameters:
        theta: circular variable in radians
        x: linear variable
    Returns:
        r_cl: circular-linear correlation coefficient
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    r_cx, _ = pearsonr(cos_theta, x)  # corr(cos(θ), x)
    r_sx, _ = pearsonr(sin_theta, x)  # corr(sin(θ), x)
    r_cs, _ = pearsonr(cos_theta, sin_theta)  # corr(cos(θ), sin(θ))

    r_cl = np.sqrt((r_cx**2 + r_sx**2 - 2 * r_cx * r_sx * r_cs) / (1 - r_cs**2))
    return r_cl
