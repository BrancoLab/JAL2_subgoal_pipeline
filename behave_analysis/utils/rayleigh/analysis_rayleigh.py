import math

import numpy as np  

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
    assert math.isnan(theta1) == False, "First angle is NaN"
    assert math.isnan(theta2) == False, "Second angle is NaN"
    
    # Calculate the absolute difference
    diff = np.abs(theta1 - theta2)

    # Adjust for the fact that 0 and 2pi are the same
    adjusted_diff = min(diff, 2 * np.pi - diff)

    # Normalize the difference to get the similarity score
    similarity_score = (np.pi - adjusted_diff) / np.pi

    assert adjusted_diff <= np.pi, "Adjusted difference is greater than pi"
    assert (
        similarity_score >= 0 and similarity_score <= 1
    ), "Similarity score is not between 0 and 1"

    return similarity_score