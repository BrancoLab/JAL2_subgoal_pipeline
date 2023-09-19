""" 
This script is used to test and implement a circular correlation coefficient function. The one selected was the one from the Mahmood paper: Mahmood, E.A., 2022. 
Robust circular-circular correlation coefficient. Communications in Statistics-Theory and Methods, pp.1-9. based on the formula by Pewsey, Neuhäuser, and Ruxton (2013) derived 
from the Fisher and Lee (1983) formula.

rho = 4[AB - CD] / SQRT([n**2 - E**2 - F**2][n**2 - G**2 - H**2])

where: 
    A = ∑cosϑicosφi,
    B = ∑sinϑisinφi,
    C = ∑cosϑisinφi,
    D = ∑sinϑicosφi,
    E = ∑cos**2ϑi,
    F = ∑sin**2ϑi,
    G = ∑cos**2φi
    H = ∑sin**2φi.

"""

import numpy as np
from itertools import combinations
import polars as pl

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
    rhoDict = loop_through_permutations_of_angles_and_apply_fishers_coeff(combinations, angles)
    
    return rhoDict

# Helper functions -----------------------------------------------------------------------------------------------------------------------------------------

def select_angle_columns(video_df) -> pl.DataFrame:
    """ 
    Depending on whether it's a mushroom or barrier experiment, filter on the available angles.
    """
    
    if "h_bar_north_a" and "h_bar_south_a" in video_df.columns:
        angles = video_df.select(["hdir", "hsa", "h_bar_north_a", "h_bar_south_a"])
                
    else:
        angles = video_df.select(["hdir", "hsa"])
        
    return angles

def create_all_the_permutations_of_angles(columns) -> list:
    """ 
    Given a list of angles, create all the permutations of angles
    """
    
    return list(combinations(columns, 2))

def loop_through_permutations_of_angles(combinations, videoDf) -> dict:
    """ 
    Loop through the permutations of angles and compute the circular correlation coefficient
    """
    
    rhoDict = {}
    for angleSet in combinations:
        alpha = videoDf[angleSet[0]].to_numpy()
        beta = videoDf[angleSet[1]].to_numpy()
        rho = circular_rho(alpha, beta)
        rhoDict[angleSet] = rho
    return rhoDict

def create_all_the_permutations_of_angles(columns) -> list:
    """ 
    Given a list of angles, create all the permutations of angles
    """
    
    return list(combinations(columns, 2))

def circular_rho(alpha, beta) -> np.float:
    """ 
    Computes the circular correlation coefficient from the Fisher and Lee (1983) formula.
    
    Inputs: np.array of angles in radians
    Output: np.float of the circular correlation coefficient between two angles
    """
    
    assert alpha.shape == beta.shape, "The alpha and beta arrays must be of the same shape"
    assert np.all(np.abs(alpha) <= np.pi), "The values in the alpha array must be between -pi and pi"
    assert np.all(np.abs(beta) <= np.pi), "The values in the beta array must be between -pi and pi"
    assert isinstance(alpha, np.ndarray), "The alpha array must be a numpy array"
    assert isinstance(beta, np.ndarray), "The beta array must be a numpy array"
    
    numerator = 4 * (((np.sum(np.cos(alpha) * np.cos(beta))) * (np.sum(np.sin(alpha) * np.sin(beta)))) - 
                     ((np.sum(np.cos(alpha) * np.sin(beta))) * (np.sum(np.sin(alpha) * np.cos(beta)))))
    
    denumerator = np.sqrt((len(alpha)**2 - np.sum(np.cos(alpha)**2) - np.sum(np.sin(alpha)**2)) *
                          (len(alpha)**2 - np.sum(np.cos(beta)**2) - np.sum(np.sin(beta)**2)))
    
    return numerator / denumerator

def loop_through_permutations_of_angles_and_apply_fishers_coeff(combinations, videoDf) -> dict:
    """ 
    Loop through the permutations of angles and compute the circular correlation coefficient
    """
    
    rhoDict = {}
    for angleSet in combinations:
        alpha = videoDf[angleSet[0]].to_numpy()
        beta = videoDf[angleSet[1]].to_numpy()
        rho = circular_rho(alpha, beta)
        rhoDict[angleSet] = rho
    return rhoDict

if __name__ == "__main__":
    """ 
    The following code is used to test four different implemntations of a circular correlation coefficient function as the astropy one seemed to be giving strange results.
    
    The four implementations are:
    (1) Jess Hamrick's implementation: https://github.com/jhamrick/python-snippets/blob/master/snippets/circstats.py
    (2) My implementation taken from the Mahmood paper: Mahmood, E.A., 2022. Robust circular-circular correlation coefficient. Communications in Statistics-Theory and Methods, pp.1-9.
    based on the formula by Pewsey, Neuhäuser, and Ruxton (2013) derived from the Fisher and Lee (1983) formula.
    (3) Astropy's implementation: https://docs.astropy.org/en/stable/api/astropy.stats.circstats.circcorrcoef.html
    (4) Chat GPT's implementation after of the formula from the Mahmood paper
    
    All functions expect equal np.array lengths and angles in radians.
    
    Results:
    - It seems the astropy implementation provides inaccurate results based on this test.
    - GPT's and my implementation seem to be the same.
    - Jess's also seems identical to mine and GPTs but fails when the correlation factor is 0. Not sure why. 
    """
    
    import matplotlib.pyplot as plt
    from astropy.stats import circcorrcoef
    
    # Build toy data -----------------------------------------------------------------------------------------------------------------------------------------
    arrayLength = 100
    alpha = np.random.uniform(-np.pi, np.pi, arrayLength)
    correlationFactors = np.arange(-1, 1, 0.1) # Generates 20 correlation factors between -1 and 1
    betas = (alpha.reshape(arrayLength, 1) * correlationFactors).T # Generates an array of shape (corelationFactors, arrayLength)
    
    # Helper function
    def wrap_to_pi(angle):
        """ 
        Helper function to ensure angles are between -pi and pi
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
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
    
    """
    nom = 4[AB - CD]
    denom = SQRT([n**2 - E**2 - F**2][n**2 - G**2 - H**2])
    rho = nom / denom
    
    where: 
    A = ∑cosϑicosφi,
    B = ∑sinϑisinφi,
    C = ∑cosϑisinφi,
    D = ∑sinϑicosφi,
    E = ∑cos**2ϑi,
    F = ∑sin**2ϑi,
    G = ∑cos**2φi
    H = ∑sin**2φi.
    """
    
    def nom(alpha, beta):
        """ 
        Computes the numerator of the circular correlation coefficient
        """
        
        nomNom = 4 * (
            ((np.sum(np.cos(alpha) * np.cos(beta))) * (np.sum(np.sin(alpha) * np.sin(beta)))) - 
            ((np.sum(np.cos(alpha) * np.sin(beta))) * (np.sum(np.sin(alpha) * np.cos(beta))))
        )
        
        return nomNom
    
    def denom(alpha, beta):
        """ 
        Computes the denominator of the circular correlation coefficient
        """
        
        denomNom = np.sqrt(
            (len(alpha)**2 - np.sum(np.cos(alpha)**2) - np.sum(np.sin(alpha)**2)) *
            (len(alpha)**2 - np.sum(np.cos(beta)**2) - np.sum(np.sin(beta)**2))
        )
    
        return denomNom
    
    def fisher_lee_coefficient(alpha, beta):
        """ 
        Computes the circular correlation coefficient
        """
        
        return nom(alpha, beta) / denom(alpha, beta)
    
    # (3) GPT implementation's of the fisher_lee_coefficient
    def fisher_lee_coefficient_GPT(angles1, angles2):
        """
        Compute the Fisher and Lee coefficient between two arrays of angles in radians.
        Based on the formula by Pewsey, Neuhäuser, and Ruxton (Citation2013).
        
        Parameters:
        - angles1: array-like, angles in radians
        - angles2: array-like, angles in radians
        
        Returns:
        - r_FL: Fisher and Lee coefficient
        """
        
        n = len(angles1)  # Assuming angles1 and angles2 are of the same length
        
        A = np.sum(np.cos(angles1) * np.cos(angles2))
        B = np.sum(np.sin(angles1) * np.sin(angles2))
        C = np.sum(np.cos(angles1) * np.sin(angles2))
        D = np.sum(np.sin(angles1) * np.cos(angles2))
        
        E = np.sum(np.cos(angles1)**2)
        F = np.sum(np.sin(angles1)**2)
        G = np.sum(np.cos(angles2)**2)
        H = np.sum(np.sin(angles2)**2)
        
        nom = 4 * ((A * B) - (C * D))
        denom = np.sqrt((n**2 - E - F) * (n**2 - G - H))
        
        r_FL = nom / denom
        
        return r_FL
    
    # Test -----------------------------------------------------------------------------------------------------------------------------------------
    
    jess, my, astro, gpt = {}, {}, {}, {}
    
    for i, correlation in enumerate(betas):
        
        betas[i] = wrap_to_pi(betas[i])
            
        # Checks
        assert len(alpha) == len(betas[i]), "The two arrays must be of equal length"
        assert np.all(np.abs(alpha) <= np.pi), "The values in the alpha array must be between -pi and pi"
        assert np.all(np.abs(betas[i]) <= np.pi), "The values in the beta array must be between -pi and pi"
            
        jess[i] = jessHamCorr(alpha, betas[i])
        my[i] = circcorrcoef(alpha, betas[i])
        astro[i] = fisher_lee_coefficient(alpha, betas[i])
        gpt[i] = fisher_lee_coefficient_GPT(alpha, betas[i])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    ax1.plot(jess.values(), label="JessHamCorr")
    ax1.plot(correlationFactors, label="Real correlation value", c = "black")
    
    ax2.plot(my.values(), label="Astropy")
    ax2.plot(correlationFactors, label="Real correlation value", c = "black")
    
    ax3.plot(astro.values(), label="Me")
    ax3.plot(correlationFactors, label="Real correlation value", c = "black")
    
    ax4.plot(gpt.values(), label="GPT")
    ax4.plot(correlationFactors, label="Real correlation value", c = "black")
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    plt.xlabel("Iterated correlation factor")
        
    plt.show()
            
    
    
    

