import numpy as np
from astropy.stats import circcorrcoef
from astropy import units as u
from itertools import combinations
import polars as pl

def compute_the_circular_rho(postProcessingObject) -> dict:
    """ 
    Ingests the frame by frame behavioural data, filters on the 
    available angles and then computes the circular correlation coefficient between the angles
    """
    
    data = postProcessingObject.video_df
    angles = select_angle_columns(data)
    combinations = create_all_the_permutations_of_angles(angles.columns)
    rhoDict = loop_through_permutations_of_angles(combinations, angles)
    
    return rhoDict

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
        alpha = videoDf[angleSet[0]].to_numpy() * u.deg
        beta = videoDf[angleSet[1]].to_numpy() * u.deg
        rho = circcorrcoef(alpha, beta)
        rhoDict[angleSet] = rho
    return rhoDict

if __name__ == "__main__":
    """ 
    The following code is used to test the circular correlation coefficient function. 
    It expects two arrays of equal length in radians, converts to degrees and then uses 
    the circcorrcoef function to compute the circular correlation coefficient.
    """
    
    alpha = np.array([356, 97, 211, 232, 343, 292, 157, 302, 335, 302, 324, 85, 324, 340, 157, 238, 254, 146, 232, 122, 329])*u.deg
    beta = np.array([119, 162, 221, 259, 270, 29, 97, 292, 40, 313, 94, 45, 47, 108, 221, 270, 119, 248, 270, 45, 23])*u.deg
    x = circcorrcoef(alpha, beta) 
    print(x)