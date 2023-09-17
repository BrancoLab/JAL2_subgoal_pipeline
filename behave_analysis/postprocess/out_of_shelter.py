import numpy as np

def out_of_shelter_filter(tracking_data: dict) -> np.ndarray:
    """
    Purpose: Compute per frame whether the mouse is out of the shelter or not. This is done by first checking for all the frames
    where the mouse is in the shelter and then inverting the boolean array using the logical_not function.
    
    Input: A dictionary of tracking data containing x, y coordinates of the mouse, shelter, barrier, etc.
    + shelter_loc: list of lists, each list contains the x, y coordinates of the opposite corners of the shelter
    
    Output: np.ndarray of booleans, True if the mouse is out of the shelter and False if the mouse is in the shelter.
    
    # BUG - I think this function is not working properly. I think it is not filtering out the frames where the mouse is in the shelter
    when the shelter is at the top of the arena and the points clicked where top left and bottom right. 
    """
    
    avg_loc_x, avg_loc_y   = tracking_data['avg_loc'][:, 0], tracking_data['avg_loc'][:, 1]
    shelter_x1, shelter_y1 = tracking_data['shelter_loc'][0]
    shelter_x2, shelter_y2 = tracking_data['shelter_loc'][1]
    in_shelter_x = np.logical_and(avg_loc_x > shelter_x1, avg_loc_x < shelter_x2)
    in_shelter_y = np.logical_and(avg_loc_y > shelter_y1, avg_loc_y < shelter_y2)
    in_shelter = np.logical_and(in_shelter_x, in_shelter_y)
    
    return np.logical_not(in_shelter)
