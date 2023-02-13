# OS Libaries
import numpy as np

def calculate_vector_between_coordinates(headCoords, goalCoords) -> np.ndarray:
    """A function that takes in two vectors of (x,y) coordinates for 
    the head and goal positions and computes a vector between them.
    
    Args:
        headCoords np.array(tuples): (x,y) coordinates of the head
        goalCoords np.array(tuples): (x,y) coordinates of the goal
    """
    return goalCoords - headCoords
