"""
This script is used to test and implement different circular correlation 
coefficient functions to see which one is the most accurate. After several tests, 
the astropy one seems to be the most accurate. Though the differences in accuracy could be 
down to the examples being created from that specific implementation method. The differences in 
methods and results highlight the need for a clear explanation of the algorithm used.
And also the robustness of the algorithms. However, this algorithm will not form a central part of 
the analysis so for now the astropy one will be used.

AstroPy implemnetation:
- Jammalamadaka and SenGupta (Citation2001)

Jess and I's implementations:
- Fisher and Lee (Citation1983)
- Implementation can be found from:
    - p151 6.36 of Statistical Analysis of Circular Data by N. I. Fisher, or
    - Mahmood paper
"""

# Import standard libaries
import os
from itertools import combinations

# Import 3rd party libaries
import numpy as np
import polars as pl
from astropy.stats import circcorrcoef
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

# Import custom libaries
from settings.settings_visualize import defined_settings_visualize as settings_v
from behave_analysis.analyze.filtering_data.filtering_functions import (
    identify_angles,
    identify_conditions,
    filter_video_dataframe,
)

# Main functions --------------------------------------------------------------------------------------------------------------------------------------------


def plot_condition_titles(conditions, nrows, columns) -> None:
    """Plot titles and remove the axes from the first
    column of subplots that act as sub titles"""
    for c_counter, c in enumerate(conditions):
        ax = plt.subplot(nrows, columns, c_counter * columns + 1)
        ax.text(1, 0.5, c, rotation="horizontal", va="center", ha="center", fontsize=20)
        ax.set_axis_off()


def plot_the_circular_rho(session, video_df, save_path) -> None:
    """Plot coeff into bar chart"""
    dirty_angles = identify_angles(session)
    angles = [angle for angle in dirty_angles if angle != "h_bar_centre_a"]
    perms = create_all_the_permutations_of_angles(angles)
    conditions = identify_conditions(session, overide=settings_v.over_ride_conditions_bool)

    # Create optimal rho dict
    optimals = load_optimals(save_path)
    optimal_rho_dic = loop_permutations_apply_circcoeff(perms, optimals)
    optimal_rhos = list(optimal_rho_dic.values())
    x_labels = [f"{perms[i][0]} VS {perms[i][1]}" for i in range(len(perms))]

    # plotting logic
    fig, axs = plt.subplots(
        nrows=len(conditions),
        ncols=2,
        figsize=(24, 6),
        sharey=False,
        sharex=True,
        gridspec_kw={"width_ratios": [1, 8]},
    )
    plot_condition_titles(conditions, len(conditions), 2)
    x = np.arange(len(x_labels))  # the label locations
    labels = ["Sampled Distribution", "Optimal Distribution"]
    colors = ["dimgrey", "lightgreen"]
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=18)

    for con_i, con in enumerate(conditions):
        # compute data
        data = filter_video_dataframe(video_df, con)
        rho_dic = loop_permutations_apply_circcoeff(perms, data)
        real_rhos = list(rho_dic.values())

        # Plot the bar chart
        axs[con_i, 1].bar(x - 0.1, real_rhos, color="darkgrey", label="Real data", width=0.2)
        axs[con_i, 1].bar(x + 0.1, optimal_rhos, color="lightgreen", label="Optimal data", width=0.2)
        axs[con_i, 1].set_xticks(x, x_labels, fontsize = 16)
        axs[con_i, 1].spines["top"].set_visible(False)
        axs[con_i, 1].spines["right"].set_visible(False)
        axs[con_i, 1].spines["left"].set_visible(False)
        axs[con_i, 1].set_ylabel("Circular rho (ρ)", fontsize=16)
        axs[con_i, 1].axhline(linewidth=1, color="black", linestyle="--")
        axs[con_i, 1].set_yticks([-1, -0.5, 0, 0.5, 1], fontsize=16)
        axs[con_i, 1].set_xticklabels(x_labels, rotation=10)


    if settings_v.show_plots:
        plt.show()
    matplotlib.rc('xtick', labelsize=20)
    plt.savefig(os.path.join(save_path, "Circular_coefficient_barplot.png"))
    plt.close()


# Helper functions -----------------------------------------------------------------------------------------------------------------------------------------


def load_optimals(save_path):
    """Load the optimal angles for synth mouse"""
    return pl.read_csv(os.path.join(save_path, "optimal_distributions.csv"))


def create_all_the_permutations_of_angles(columns) -> list:
    """Permutate a list of angle strings"""
    return list(combinations(columns, 2))


def loop_permutations_apply_circcoeff(combinations, video_df) -> dict:
    """Apply coeff to all permutations of angles"""
    rhoDict = {}
    for angle_set in combinations:
        alpha = np.array(video_df[angle_set[0]].to_numpy())
        beta = np.array(video_df[angle_set[1]].to_numpy())
        rho = circcorrcoef(alpha, beta)
        rhoDict[angle_set] = rho
    return rhoDict


# if __name__ == "__main__":
#     """
#     The following code is used to test three different implemntations of a circular correlation coefficient function:

#     The three implementations are:
#     (1) Jess Hamrick's implementation: https://github.com/jhamrick/python-snippets/blob/master/snippets/circstats.py
#     (2) My implementation taken from the Mahmood paper: Mahmood, E.A., 2022. Robust circular-circular correlation coefficient. Communications in Statistics-Theory and Methods, pp.1-9.
#     based on the formula by Pewsey, Neuhäuser, and Ruxton (2013) derived from the Fisher and Lee (1983) formula.
#     (3) Astropy's implementation: https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html

#     All functions expect equal np.array lengths and angles in radians.

#     Results:
#     - It seems mine and Jess Hamrick's implementation are identical
#     - The Astropy's implementation produces slightly different results but passes all the example tests so will use that one for now. ALthough note that the differences could be in the
#     underlying implementation of the functions used to compute the circular correlation coefficient. It's not clear what implementation Astropy uses, but the fisher one is defined in
#     this script. So choosing the Astropy one for now for the sake of passing tests over clarity on the algorithm used.
#     """

#     # Implementations -----------------------------------------------------------------------------------------------------------------------------------------
#     # (1) Jess Hamrick's implementation:

#     def jessHamCorr(alpha1, alpha2, nanrobust=False, axis=None):
#         if axis is not None and alpha1.shape[axis] != alpha2.shape[axis]:
#             raise (ValueError, "shape mismatch")

#         # compute mean directions
#         if axis is None:
#             n = alpha1.size
#         else:
#             n = alpha1.shape[axis]

#         c1 = np.cos(alpha1)
#         c1_2 = np.cos(2 * alpha1)
#         c2 = np.cos(alpha2)
#         c2_2 = np.cos(2 * alpha2)
#         s1 = np.sin(alpha1)
#         s1_2 = np.sin(2 * alpha1)
#         s2 = np.sin(alpha2)
#         s2_2 = np.sin(2 * alpha2)

#         if nanrobust:
#             sumfunc = lambda x: np.nansum(x, axis=axis)
#         else:
#             sumfunc = lambda x: np.sum(x, axis=axis)

#         num = 4 * (sumfunc(c1 * c2) * sumfunc(s1 * s2) - sumfunc(c1 * s2) * sumfunc(s1 * c2))
#         den = np.sqrt(
#             (n**2 - sumfunc(c1_2) ** 2 - sumfunc(s1_2) ** 2) * (n**2 - sumfunc(c2_2) ** 2 - sumfunc(s2_2) ** 2)
#         )
#         rho = num / den

#         return rho

#     # (2) My implementation for the fisher equation from - Mahmood, E.A., 2022.

#     def numerator(theta, phi):
#         """
#         Computes the numerator of the circular correlation coefficient:
#             4(AB - CD)

#         Where:
#             A = ∑cosϑicosφi
#             B = ∑sinϑisinφi
#             C = ∑cosϑisinφi
#             D = ∑sinϑicosφi
#         """

#         A = np.sum(np.cos(theta) * np.cos(phi))
#         B = np.sum(np.sin(theta) * np.sin(phi))
#         C = np.sum(np.cos(theta) * np.sin(phi))
#         D = np.sum(np.sin(theta) * np.cos(phi))

#         return 4 * (A * B - C * D)

#     def denominator(theta, phi):
#         """
#         Computes the denominator of the circular correlation coefficient:
#             SQRT([n**2 - E**2 - F**2][n**2 - G**2 - H**2])
#         """
#         assert len(theta) == len(phi), "The theta and phi arrays must be of the same length"

#         n = len(theta)  # Could also use phi as they are the same length
#         E = np.sum(np.cos(2 * theta))
#         F = np.sum(np.sin(2 * theta))
#         G = np.sum(np.cos(2 * phi))
#         H = np.sum(np.sin(2 * phi))

#         return np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))

#     def fisher_lee_coefficient(theta, phi):
#         """
#         Computes the circular correlation coefficient
#         """

#         return numerator(theta, phi) / denominator(theta, phi)

#     # (3) Astropy's implementation:
#     # circcorrcoef(theta, phi) - https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html

#     # ---------------------------------------------------- Known correlation value tests ----------------------------------------------------------------------

#     # Example taken from https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html))
#     x1 = np.array([0.785, 1.570, 3.141, 3.839, 5.934])
#     x2 = np.array([0.593, 1.291, 2.879, 3.892, 6.108])
#     rhoX1X2 = 0.94
#     assert round(circcorrcoef(x1, x2), 2) == rhoX1X2
#     # assert round(fisher_lee_coefficient(x1, x2), 2) == rhoX1X2, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(x1, x2), 2)} != {rhoX1X2}"

#     # Example taken from https://gist.github.com/kn1cht/89dc4f877a90ab3de4ddef84ad91124e
#     a1 = np.deg2rad(np.array([-30, 45, 0, 10, -15]))
#     a2 = np.deg2rad(np.array([200, 180, 170, 150, 210]))
#     rhoA1A2 = -0.53
#     assert round(circcorrcoef(a1, a2), 2) == rhoA1A2
#     # assert round(fisher_lee_coefficient(a1, a2), 2) == rhoA1A2, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(a1, a2), 2)} != {rhoA1A2}"

#     # # Example taken from https://docs.astropy.org/en/stable/api/astropy.stats.circcorrcoef.html
#     b1 = np.deg2rad(
#         np.array(
#             [356, 97, 211, 232, 343, 292, 157, 302, 335, 302, 324, 85, 324, 340, 157, 238, 254, 146, 232, 122, 329]
#         )
#     )
#     b2 = np.deg2rad(
#         np.array([119, 162, 221, 259, 270, 29, 97, 292, 40, 313, 94, 45, 47, 108, 221, 270, 119, 248, 270, 45, 23])
#     )
#     rhoB1B2 = 0.270
#     assert round(circcorrcoef(b1, b2), 3) == rhoB1B2
#     # assert round(fisher_lee_coefficient(b1, b2), 3) == rhoB1B2, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(b1, b2), 3)} != {rhoB1B2}"

#     # Example taken from CircStats package in Matlab written by Philip Berens - https://www.jstatsoft.org/article/view/v031i10
#     alpha_rad = np.deg2rad(
#         np.array([13, 15, 21, 26, 28, 30, 35, 36, 41, 60, 92, 103, 165, 199, 210, 250, 301, 320, 343, 359])
#     )
#     beta_deg = np.deg2rad(
#         np.array([1, 13, 41, 56, 67, 71, 81, 85, 99, 110, 119, 131, 145, 177, 199, 220, 291, 320, 340, 355])
#     )
#     rhoCircStats = 0.67
#     assert round(circcorrcoef(alpha_rad, beta_deg), 2) == rhoCircStats
#     # assert round(fisher_lee_coefficient(alpha_rad, beta_deg), 2) == rhoCircStats, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(alpha_rad, beta_deg), 2)} != {rhoCircStats}"
