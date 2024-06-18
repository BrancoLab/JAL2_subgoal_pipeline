import numpy as np
import matplotlib.pyplot as plt


def get_color_based_on_neural_activity(neural_data: float) -> np.ndarray:
    """Produces a colour at a given point in space when provided with neural data
    

    
    
        Returns:
            np.ndarray: A 3D vector representing the colour at a given point in space
                normalized to the range 0-1 as required by matplotlib
    """

    colormap = [[90, 90, 90], [90, 90, 180], [80, 150, 240], [90, 230, 240], [90, 230, 240]]
    # colormap = [[90, 90, 90], [90, 90, 180], [143, 91, 75], [199, 91, 74], [247, 43, 15]]
    colormap = [np.array(x) for x in colormap]
    thresholds = np.array([0, 5, 15, 25, 999])  # spikes / frame
    assert len(colormap) == len(thresholds)

    # if neural_data is 0, set the index to 0
    if neural_data == 0.0:
        idx = 0 # idx of threshold
        
    # if neural_data is zero, find the index of the last threshold that is less than the neural data
    # i.e if the neural spike count is 10, which index threshold does it fall into
    if neural_data > 0.0:
        idx = np.where((neural_data - thresholds) > 0)[0][-1] # idx of threshold

    color = ((thresholds[idx + 1] - neural_data) * colormap[idx] + (neural_data - thresholds[idx]) * colormap[idx + 1]) / (
        thresholds[idx + 1] - thresholds[idx]
    )
    
    assert len(color) == 3, "Must return a 3D color vector"
    color = color / 256 # 0-1 range

    return color


def get_color_based_on_speed(speed: float, object_to_color: str, stim_status: float = 0, stim_type: str = "audio") -> tuple:

    # No idea whay return this numbers if speed is none
    if speed == None:
        return (1, 0.8, 0.4, 0.3)

    colormap, speed_thresholds = get_color_parameters(stim_type, stim_status, object_to_color)

    if speed == 0.0:
        idx = 0
    if speed > 0.0:
        idx = np.where((speed - speed_thresholds) > 0)[0][-1]
    color = ((speed_thresholds[idx + 1] - speed) * colormap[idx] + (speed - speed_thresholds[idx]) * colormap[idx + 1]) / (
        speed_thresholds[idx + 1] - speed_thresholds[idx]
    )
    if object_to_color in ["plot", "trial"]:
        color = np.append(color[::-1] / 256, 0.7)  # BGR to RGB and 0-256 to 0-1 range
    return color


def get_color_based_on_target_score(target_score: float, edge_vector_threshold: float) -> tuple:
    homing_vector_color = np.array([0.4, 0.4, 0.4, 0.6])
    edge_vector_color = np.array([0, 0.4, 1, 0.5])  # np.array([0, 0, 0, 1])
    if target_score > edge_vector_threshold:
        color = edge_vector_color
    if target_score < edge_vector_threshold:
        color = homing_vector_color
    return color


def get_color_parameters(stim_type: str = "audio", stim_status: float = 0, object_to_color: str = "trail"):
    if stim_type == "audio":
        speed_thresholds = np.array([0, 20, 40, 70, 999])  # cm/s
    else:
        speed_thresholds = np.array([0, 15, 20, 30, 999])  # cm/s
    if object_to_color == "trail" or object_to_color == "plot":
        if stim_type in ["audio", "homing", "threshold_crossing", "explore"]:
            # colormap = [[50, 50,  50], [50, 50, 100], [50, 100,200], [250,250,255], [250,250,255]]
            colormap = [[90, 90, 90], [90, 90, 180], [80, 150, 240], [90, 230, 240], [90, 230, 240]]
        elif stim_type == "laser" and stim_status != 0:
            colormap = [[25, 25, 25], [100, 50, 50], [200, 100, 50], [255, 230, 230], [255, 230, 230]]
        elif stim_type == "laser" and stim_status == 0:
            colormap = [[255, 200, 0], [255, 200, 0], [255, 200, 0], [255, 200, 0], [255, 200, 0]]
    elif object_to_color == "text":
        colormap = [[100, 100, 100], [100, 100, 175], [100, 175, 220], [250, 250, 255], [250, 250, 255]]
    elif object_to_color == "trial":
        colormap = [[90, 90, 90], [90, 90, 180], [90, 150, 240], [90, 230, 240], [90, 230, 240]]

    return [np.array(x) for x in colormap], speed_thresholds


def get_colormap(object_to_color="tracking video", plot_type="trajectory"):
    colormap = [
        (0, 0, 255),
        (255, 0, 255),
        (120, 120, 255),
        (0, 255, 255),
        (0, 255, 150),
        (0, 150, 0),
        (255, 255, 0),
        (120, 120, 120),
        (255, 50, 0),
        (255, 50, 80),
        (255, 50, 150),
        (150, 0, 150),
        (30, 0, 180),
    ]
    if object_to_color == "plot":
        colormap = plt.get_cmap("viridis")(np.linspace(0, 0.95, 16))
        if plot_type == "trajectory":
            colormap[:, 3] = np.linspace(0.4, 0.8, 16)
        if plot_type == "scatter":
            colormap[:, 3] = 0.6
        colormap = colormap[np.array([6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1]), :]
    return colormap


def generate_list_of_colors(
    color_by: str = "speed", stim_type: str = "audio", speeds: np.ndarray = None, RT: int = 0, object_to_color: str = "plot"
) -> list:
    if "speed" in color_by:
        colors = []
        for i, speed in enumerate(speeds):
            if "RT" in color_by and i < RT:
                speed = None
            color = get_color_based_on_speed(speed=speed, object_to_color=object_to_color, stim_status=-1, stim_type=stim_type)
            colors.append(color)
    if color_by == "time":
        colormap = plt.get_cmap("viridis_r")
        colors = colormap(np.linspace(0.03, 0.75, len(speeds)))
        colors[:, 3] = np.linspace(0.4, 1, len(speeds))[::-1]

    return colors
