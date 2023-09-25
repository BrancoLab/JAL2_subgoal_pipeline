""" 
This script is used to test and implement different circular correlation coefficient functions to see which one is the most accurate. After several tests, the astropy one seems to be the most accurate.
Though the differences in accuracy could be down to the examples being created from that specific implementation method. The differences in methods and results highlight the need for a clear explanation of the algorithm used.
And also the robustness of the algorithms. However, this algorithm will not form a central part of the analysis so for now the astropy one will be used.

AstroPy implemnetation:
- Jammalamadaka and SenGupta (Citation2001)

Jess and Mine implementations:
- Fisher and Lee (Citation1983)
- Implementation can be found from:
    - p151 6.36 of Statistical Analysis of Circular Data by N. I. Fisher, or
    - Mahmood paper
"""

import numpy as np
from itertools import combinations
import polars as pl
from astropy.stats import circcorrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import settings

from settings.settings_visualize import defined_settings_visualize as settings_v

# Main function --------------------------------------------------------------------------------------------------------------------------------------------

def compute_the_circular_rho(postProcessingObject) -> dict:
    """ 
    Ingests the frame by frame behavioural data, filters on the 
    available angles when the mouse is out of the shelter and then 
    computes the circular correlation coefficient between the angles
    """
    
    data = postProcessingObject.video_df
    outOfShelterFrames = data.filter(pl.col("OutofshelterIdx") == True)
    angles = select_angle_columns(outOfShelterFrames)
    combinations = create_all_the_permutations_of_angles(angles.columns)
    rhoDict = loop_through_permutations_of_angles_and_apply_circcoeff(combinations, angles)
    
    return rhoDict

def plot_the_circular_rho(postProcessingObject, save_path) -> None:
    """ 
    Plots the circular correlation coefficient into a bar chart and saves it to the processed data folder
    """
    
    # Extract the rho values and the pair wise combinations produced by the compute_the_circular_rho function   
    rhoDict = compute_the_circular_rho(postProcessingObject)
    rhos = list(rhoDict.values())
    pairWiseCombinations = list(rhoDict.keys())
    xlabels = [f"{pairWiseCombinations[i][0]} VS {pairWiseCombinations[i][1]}" for i in range(len(pairWiseCombinations))]
    
    # Plot the bar chart
    fig, ax = plt.subplots()
    sns.barplot(x=xlabels, y=rhos, ax=ax, color = 'cornflowerblue')
    plt.axhline(y=0, color='black', linestyle='--')
    sns.despine(top=True, right=True, left=True, bottom=False, offset=None, trim=True)
    ax.set(ylim=(-1.1, 1.1))
    ax.bar_label(ax.containers[0], label_type='center', fmt='%.2f', color='black', fontsize=12)
    ax.set_yticks(np.arange(-1, 1.1, 0.25))
    ax.set_ylabel('Rho (ρ)', fontsize=14)
    plt.title('Circular correlation coefficients', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    fig.set_size_inches(20, 8)
    
    # Save and show the plot if the user wants to
    if settings_v.show_plots: plt.show()
    plt.savefig(os.path.join(save_path, "Circular_coefficient_barplot.png"))
    plt.close()
     
# Helper functions -----------------------------------------------------------------------------------------------------------------------------------------

def select_angle_columns(videoDf) -> pl.DataFrame:
    """ 
    Depending on whether it's a mushroom or barrier experiment, filter on the available angles.
    """
    
    if "h_bar_north_a" and "h_bar_south_a" in videoDf.columns:
        angles = videoDf.select(["hdir", "hsa", "h_bar_north_a", "h_bar_south_a"])
                
    else:
        angles = videoDf.select(["hdir", "hsa"])
        
    return angles

def create_all_the_permutations_of_angles(columns) -> list:
    """ 
    Given a list of angles, create all the permutations of angles
    """
    
    return list(combinations(columns, 2))

def loop_through_permutations_of_angles_and_apply_circcoeff(combinations, videoDf) -> dict:
    """ 
    Loop through the permutations of angles and compute the circular correlation coefficient
    """
    
    rhoDict = {}
    for angleSet in combinations:
        alpha = np.array(videoDf[angleSet[0]].to_numpy())
        beta = np.array(videoDf[angleSet[1]].to_numpy())
        rho = circcorrcoef(alpha, beta)
        rhoDict[angleSet] = rho
    return rhoDict

if __name__ == "__main__":
    """ 
    The following code is used to test three different implemntations of a circular correlation coefficient function:
    
    The three implementations are:
    (1) Jess Hamrick's implementation: https://github.com/jhamrick/python-snippets/blob/master/snippets/circstats.py
    (2) My implementation taken from the Mahmood paper: Mahmood, E.A., 2022. Robust circular-circular correlation coefficient. Communications in Statistics-Theory and Methods, pp.1-9.
    based on the formula by Pewsey, Neuhäuser, and Ruxton (2013) derived from the Fisher and Lee (1983) formula.
    (3) Astropy's implementation: https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html
    
    All functions expect equal np.array lengths and angles in radians.
    
    Results:
    - It seems mine and Jess Hamrick's implementation are identical
    - The Astropy's implementation produces slightly different results but passes all the example tests so will use that one for now. ALthough note that the differences could be in the
    underlying implementation of the functions used to compute the circular correlation coefficient. It's not clear what implementation Astropy uses, but the fisher one is defined in 
    this script. So choosing the Astropy one for now for the sake of passing tests over clarity on the algorithm used.
    """
        
    # Implementations -----------------------------------------------------------------------------------------------------------------------------------------
    # (1) Jess Hamrick's implementation:
    
    def jessHamCorr(alpha1, alpha2, nanrobust = False, axis= None):
        if axis is not None and alpha1.shape[axis] != alpha2.shape[axis]:
            raise(ValueError, "shape mismatch")

        # compute mean directions
        if axis is None:
            n = alpha1.size
        else:
            n = alpha1.shape[axis]

        c1 = np.cos(alpha1)
        c1_2 = np.cos(2*alpha1)
        c2 = np.cos(alpha2)
        c2_2 = np.cos(2*alpha2)
        s1 = np.sin(alpha1)
        s1_2 = np.sin(2*alpha1)
        s2 = np.sin(alpha2)
        s2_2 = np.sin(2*alpha2)

        if nanrobust:
            sumfunc = lambda x: np.nansum(x, axis=axis)
        else:
            sumfunc = lambda x: np.sum(x, axis=axis)

        num = 4 * (sumfunc(c1*c2) * sumfunc(s1*s2) - sumfunc(c1*s2) * sumfunc(s1*c2))
        den = np.sqrt((n**2 - sumfunc(c1_2)**2 - sumfunc(s1_2)**2) * (n**2 - sumfunc(c2_2)**2 - sumfunc(s2_2)**2))
        rho = num / den

        return rho
    
    # (2) My implementation for the fisher equation from - Mahmood, E.A., 2022. 
    
    def numerator(theta, phi):
        """ 
        Computes the numerator of the circular correlation coefficient:
            4(AB - CD)
         
        Where:
            A = ∑cosϑicosφi
            B = ∑sinϑisinφi
            C = ∑cosϑisinφi
            D = ∑sinϑicosφi
        """
        
        A = np.sum(np.cos(theta) * np.cos(phi))
        B = np.sum(np.sin(theta) * np.sin(phi))
        C = np.sum(np.cos(theta) * np.sin(phi))
        D = np.sum(np.sin(theta) * np.cos(phi))
        
        return 4 * (A*B - C*D)
    
    def denominator(theta, phi):
        """ 
        Computes the denominator of the circular correlation coefficient:
            SQRT([n**2 - E**2 - F**2][n**2 - G**2 - H**2])
        """
        assert len(theta) == len(phi), "The theta and phi arrays must be of the same length"
        
        n = len(theta) # Could also use phi as they are the same length
        E = np.sum(np.cos(2 * theta))
        F = np.sum(np.sin(2 * theta))
        G = np.sum(np.cos(2 * phi))
        H = np.sum(np.sin(2 * phi))
        
        return np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    
    def fisher_lee_coefficient(theta, phi):
        """ 
        Computes the circular correlation coefficient
        """
        
        return numerator(theta, phi) / denominator(theta, phi)
    
    # (3) Astropy's implementation:
    # circcorrcoef(theta, phi) - https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html
    
    # ---------------------------------------------------- Known correlation value tests ----------------------------------------------------------------------
    
    
    # Example taken from https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html))
    x1 = np.array([0.785, 1.570, 3.141, 3.839, 5.934])
    x2 = np.array([0.593, 1.291, 2.879, 3.892, 6.108])
    rhoX1X2 = 0.94
    assert round(circcorrcoef(x1, x2), 2) == rhoX1X2
    # assert round(fisher_lee_coefficient(x1, x2), 2) == rhoX1X2, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(x1, x2), 2)} != {rhoX1X2}"
    
    # Example taken from https://gist.github.com/kn1cht/89dc4f877a90ab3de4ddef84ad91124e
    a1 = np.deg2rad(np.array([-30,  45,   0,  10, -15]))
    a2 = np.deg2rad(np.array([200, 180, 170, 150, 210]))
    rhoA1A2 = -0.53
    assert round(circcorrcoef(a1, a2), 2) == rhoA1A2
    # assert round(fisher_lee_coefficient(a1, a2), 2) == rhoA1A2, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(a1, a2), 2)} != {rhoA1A2}"

    # # Example taken from https://docs.astropy.org/en/stable/api/astropy.stats.circcorrcoef.html
    b1 = np.deg2rad(np.array([356, 97, 211, 232, 343, 292, 157, 302, 335, 302, 324, 85, 324, 340, 157, 238, 254, 146, 232, 122, 329]))
    b2 = np.deg2rad(np.array([119, 162, 221, 259, 270, 29, 97, 292, 40, 313, 94, 45, 47, 108, 221, 270, 119, 248, 270, 45, 23]))
    rhoB1B2 = 0.270
    assert round(circcorrcoef(b1, b2), 3) == rhoB1B2
    # assert round(fisher_lee_coefficient(b1, b2), 3) == rhoB1B2, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(b1, b2), 3)} != {rhoB1B2}"

    # Example taken from CircStats package in Matlab written by Philip Berens - https://www.jstatsoft.org/article/view/v031i10
    alpha_rad = np.deg2rad(np.array([13, 15 ,21 ,26 ,28 ,30 ,35 ,36 ,41 ,60 ,92 ,103 ,165 ,199, 210, 250, 301, 320, 343, 359]))
    beta_deg = np.deg2rad(np.array([1, 13, 41, 56, 67, 71, 81, 85, 99, 110, 119, 131, 145, 177, 199, 220, 291, 320, 340, 355]))
    rhoCircStats = 0.67
    assert round(circcorrcoef(alpha_rad, beta_deg), 2) == rhoCircStats
    # assert round(fisher_lee_coefficient(alpha_rad, beta_deg), 2) == rhoCircStats, f"fisher_lee_coefficient: {round(fisher_lee_coefficient(alpha_rad, beta_deg), 2)} != {rhoCircStats}"

