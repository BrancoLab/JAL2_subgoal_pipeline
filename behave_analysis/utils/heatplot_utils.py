import numpy as np

def remove_points_away_from_center_of_circle(x, y, session_height) -> tuple:
    """Remove all points that are outside of the arena circle and return the filtered x and y coordinates.
 
    TODO:
        + Make the radius of the arena a variable not hard coded
    """

    dist = np.sqrt(
        ((x - session_height / 2) ** 2) + ((y - session_height / 2) ** 2)
    )  # Use the euclidean distance formula to find the distance from the center of the arena
    filt_x = x[dist < 460]  # 460 is size of arena circle radius, see register
    filt_y = y[dist < 460]
    return filt_x, filt_y