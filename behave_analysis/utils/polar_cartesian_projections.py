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
    """Converts negative radians to positive radians. Leaves postive radians unchanged.
    Now values should be between 0 and 2 * pi (6.28)."""
    angle_array = np.where(angle_array < 0, angle_array + 2 * np.pi, angle_array)
    assert np.all(angle_array >= 0), "There are still negative radians in the array"
    assert np.all(angle_array < 2 * np.pi), "There are radians greater than 2 * pi in the array"
    return angle_array
