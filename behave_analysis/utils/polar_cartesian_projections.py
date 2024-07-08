import numpy as np

def polar_to_cartesian(theta) -> tuple:
    """Converts polar coordinates to cartesian coordinates"""
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y

def cartesian_to_polar(x, y) -> float:
    """Converts cartesian coordinates to polar coordinates"""
    theta = np.arctan2(y, x)
    return theta

def negative_radians_to_positive(angle_array: np.ndarray) -> np.ndarray:
    """Converts negative radians to positive radians. Leaves postive radians unchanged."""
    return np.where(angle_array < 0, angle_array + 2 * np.pi, angle_array)