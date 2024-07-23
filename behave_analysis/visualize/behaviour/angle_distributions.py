"""
The point of this script to observe how well we have sampled all of the angles of interest.
It does this by plotting the sampled angles as probability distribution functions (PDFs)
and comparing them to the optimal PDFs. The optimal distributions are what would be observed if 
the mouse was sampling all of the angles of interest uniformly. This is done by binning the 
arena up into a grid and then spinning an artifical mouse around in each bin and calculating 
the angle between the head and the point of interest.

NOTE:
- Discovered negative x and y coordinates for the min head direction points which is not possible. 
Leaving this here as a reminder that we need more tests and quality checks in the pipeline as the 
pipeline is untested and unreliable in it's current state 

TODO:
- The logic is not robust in the sense that it does not prevent the synthetic mouse from walking through the barrier
This should not be a too big of an issue and should only have minimal impact on the marginals
- Make util functions for removing points away from the center global as been used 3x now in individual scripts
"""

import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import polars as pl

from behave_analysis.analyze.filtering_data.filtering_functions import (
    identify_angles,
    filter_video_dataframe,
)

# ---------------------------------------Main Functions-----------------------------------------------

def plot_condition_titles(conditions, nrows, columns) -> None:
    """Plot titles and remove the axes from the first 
    column of subplots that act as sub titles"""
    for c_counter, c in enumerate(conditions):
        ax = plt.subplot(nrows, columns, c_counter * columns + 1)
        ax.text(0.5, 0.4, c, rotation="horizontal", va="center", ha="center", fontsize=20)
        ax.set_axis_off()

def plot_angle_distributions(session, settings, trackingData, video_data, conditions, sessionHeight, save_path) -> None:
    """
    Plot sample and optimal angle distributions of behaviour in the arena.

    This function is the main function that calls all of the other
    functions in this script. It plots all of the sampled angled
    distributions vs the optimal distributions for each point of
    interest in the arena.

    """
        
    angles = identify_angles(session)
    optimal_dic, hdir_df = create_optimal_distributions(trackingData, sessionHeight, session.barrier_time)
    save_optimal_as_csv(optimal_dic, hdir_df, save_path)
    
    # Plotting
    fig, axs = plt.subplots(nrows=len(conditions), ncols=len(angles) + 1, figsize=(24, 6), sharey=False, sharex=True)
    plot_condition_titles(conditions, len(conditions), len(angles) + 1)
    labels = ["Sampled Distribution", "Optimal Distribution"]
    colors = ["dimgrey", "lightgreen"]
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=16)
    
    for con_i, con in enumerate(conditions):
        videoDf = filter_video_dataframe(video_data, con)

        # Plot each angle
        for i, angle in enumerate(angles):
            axs[con_i, i + 1].hist(videoDf[str(angle)], bins=30, density=True, color="dimgrey", alpha=0.8)
            axs[con_i, i + 1].set_xlabel(f"{str(angle)} (radians)", fontsize=16)

            # Plot optimal distribution if it exists
            if angle in optimal_dic:
                dataframe = pd.DataFrame(optimal_dic[angle], columns=["angle"])
                axs[con_i, i + 1].hist(dataframe, bins=30, density=True, color="lightgreen", alpha=0.3)

        # Settings for all axes
        for ax in axs.flat:
            ax.set_xlim([-np.pi, np.pi])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.set_yticks(np.arange(0, 0.5, 0.1))

        axs[con_i, 1].set_ylabel("Probability Density", fontsize=16)
    

    if settings.show_plots:
        plt.show()
    
    for ax in axs[:, 1].flat:
        plt.setp(ax.get_xticklabels(), visible=True)

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(os.path.join(save_path, "behavioural_angle_distributions_vs_optimals.png"))
    plt.close()


def create_optimal_distributions(trackingData, session_height, mushroom) -> dict:
    """Generates the optimal sampling distributions for objs of interest

    Returns:
    -- dict: of pdfs for each point of interest
    """
    pointsOfInterest = extract_xys_of_arena_objects(trackingData, mushroom)
    xEdges, yEdges = create_grid()
    positionPoints = generate_position_points(xEdges, yEdges, session_height)
    hdir_df = generate_hdir_data_for_each_position_point(positionPoints)

    # Gen angles between synthetic hdir and points of interest
    dict = {}
    for _, point in enumerate(pointsOfInterest):
        dict[point] = compute_the_angle_between_the_head_and_a_point(hdir_df, pointsOfInterest[point])
    return dict, hdir_df


# ----------------------------------------------------------------------------------------------------------------------------------------------


def extract_xys_of_arena_objects(trackingData, barrier_time: bool) -> dict:
    """Get X,Y coordinates of the points of interest in the arena.

    These are not angles, but XY points. They are named as angles so that
    the variable names can be used as a lookup table for the angles generated"""
    if not barrier_time:
        shelter_location = (
            np.mean([trackingData["shelter_loc"][0][0], trackingData["shelter_loc"][1][0]]),
            np.mean([trackingData["shelter_loc"][0][1], trackingData["shelter_loc"][1][1]]),
        )
        return {"hsa": shelter_location}

    if barrier_time:
        shelter_location = (
            np.mean([trackingData["shelter_loc"][0][0], trackingData["shelter_loc"][1][0]]),
            np.mean([trackingData["shelter_loc"][0][1], trackingData["shelter_loc"][1][1]]),
        )
        preflipLocation = trackingData["barrier_loc"][0]
        centerLocation = trackingData["barrier_loc"][2]
        postflipLocation = trackingData["barrier_loc"][1]
        return {
            "hsa": shelter_location,
            "h_bar_centre_a": centerLocation,
            "h_postflipbar_a": postflipLocation,
            "h_preflipbar_a": preflipLocation,
        }


def create_grid() -> tuple:
    """Create a grid of bins in the arena."""
    xs = np.arange(0, 1024, 1)  # 1024 is the size of the arena
    ys = np.arange(0, 1024, 1)
    _, xEdges, yEdges = np.histogram2d(xs, ys, bins=300)
    return xEdges, yEdges


def generate_position_points(xEdges, yEdges, session_height, plotPointsGenerated=False) -> list:
    """Generate X,Y coordinates of the center of each bin in the arena."""
    xCoords = (xEdges[1:] + xEdges[:-1]) / 2  # Find center of bins
    yCoords = (yEdges[1:] + yEdges[:-1]) / 2  # Find center of bins
    xCoords, yCoords = remove_points_away_from_center_of_circle(xCoords, yCoords, session_height)
    positionPoints = [[x, y] for x in xCoords for y in yCoords]  # pack coords

    # Plot the points generated for visualisation purposes and debugging
    if plotPointsGenerated:
        plt.scatter(*zip(*positionPoints))
        plt.grid(True)
        plt.xlabel("X Coordinates")
        plt.ylabel("Y Coordinates")
        plt.title("Plot of Coordinate Pairs")
        plt.show()

    return positionPoints


def generate_hdir_data_for_each_position_point(positionPoints) -> pd.DataFrame:
    """
    Create synthetic head direction data for each position point in the arena.

    Generate a dataframe of all possible combinations of position points and spins.
    This is the function that generates the synthetic head direction data for each
    "grid" or bin in the arena generated from the bin_the_data_create_edges function.
    """
    spins = np.arange(-np.pi, np.pi, 0.1)

    # create an array of all possible combinations of position points and spins
    data = np.array(list(itertools.product(positionPoints, spins)), dtype=object)
    dataframe = pd.DataFrame(data, columns=["xY", "hDir"])
    return dataframe


# --------------------------------------------------------------------------------------------
def remove_points_away_from_center_of_circle(x, y, session_height) -> tuple:
    """
    Ensures there are no positions outside of the areana by removing them from the x and y coordinates based
    on the fact that the radius of the arena is 460 pixels.

    TODO:
    + Make this function global
    + Make the radius of the arena a variable not hard coded
    """

    dist = np.sqrt(
        ((x - session_height / 2) ** 2) + ((y - session_height / 2) ** 2)
    )  # Use the euclidean distance formula to find the distance from the center of the arena
    filtX = x[dist < 460]  # 460 is size of arena circle radius, see register
    filtY = y[dist < 460]
    return filtX, filtY


# This was tacken from track and changed so could be refactored to share a single component of code
def compute_the_angle_between_the_head_and_a_point(headPositionDf, pointOfInterest):
    """
    Input:
    + headPosition, (X, Y) coordinates of the head as well as synthetic head direction
    + pointname: the column name of the point you want to compute the angle to in the tracking data e.g barrier location, etc#
    + idx: Some of the columns have multiple index points e.g barrier location has two points so you need to index which one you want

    Expects:
    - angles to be in radians

    Logic:
    - xLen and yLen are the x and y components of the triangle formed by the head and the point of interest
    - Inverse of Tan is used to obtain the angle in radians between the x and y components, arcTan2 is used to ensure 
    the correct quadrant is returned (Unsure why negative is needed)
    - The pos and negative logic is to covert a 270 degree turn into a -90 degree turn
    - Rotate the coordinate plane by 90 degrees to ensure it is in the same coordinate system as the head direction
    - Ensure angle generated is (from pi to -pi) by wrapping values over 180 degrees back to negative
    """

    # Use the inverse of tan to compute the angle between the head and the point of interest
    xLen = -headPositionDf["xY"].apply(lambda x: x[0]) + pointOfInterest[0]
    yLen = -headPositionDf["xY"].apply(lambda x: x[1]) + pointOfInterest[1]
    angleOFInterest = -np.arctan2(yLen, xLen)

    # Ensure a 270 degree turn is converted to a -90 degree turn
    isAngleOFInterestPositive = angleOFInterest > 0
    isAngleOFInterestNegative = angleOFInterest < 0
    angleOFInterest[isAngleOFInterestNegative] += np.pi
    angleOFInterest[isAngleOFInterestPositive] -= np.pi

    # Ensure angle generated is (from pi to -pi)
    adjustedAngleOfInterest = np.pi + (
        angleOFInterest - headPositionDf["hDir"]
    )  # brackets for order of operations its not atuple
    adjustedAngleOfInterest[adjustedAngleOfInterest > np.pi] = adjustedAngleOfInterest[
        adjustedAngleOfInterest > np.pi
    ] - (2 * np.pi)

    return adjustedAngleOfInterest


# -------------------------------------------- Utilities ----------------------------------------------------------------------------------------


def save_optimal_as_csv(dict, hdir_df, save_path) -> None:
    """Converts dictionary to csv"""
    df = pl.DataFrame(dict)
    arr = hdir_df["hDir"].to_numpy().astype(float)

    # Ensure the length of 'arr' matches the number of rows in 'df'
    if len(arr) != df.height:
        print(f"The length of 'arr' does not match the number of rows in 'df': {len(arr)} vs {df.height}.")
        return

    # large = df.with_column(pl.Series("hDir", arr).alias("hdir")).to_pandas() # with_column deprecated in newer polars versions
    large = df.hstack([pl.Series("hdir",arr)]).to_pandas()
    path = os.path.join(save_path, "optimal_distributions.csv")
    large.to_csv(path)
    return None
