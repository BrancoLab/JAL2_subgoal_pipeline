# custome libaries
from behave_analysis.utils.color_funcs import get_colormap, generate_list_of_colors, get_color_based_on_target_score

# Os Libraries
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.collections as plt_coll
import numpy as np

def solid_line(self, trial: dict):
    color = get_plot_color(self, trial, plot_type='trajectory')
    self.ax.plot(trial['trajectory x'], trial['trajectory y'], color = color, linewidth=2)

def get_plot_color(self, trial: dict, plot_type:str='trajectory') -> tuple:
    if not self.color_by: 
        color=(.4,.4,.4, .7)
    if self.color_by=='target':
        color = get_color_based_on_target_score(trial['escape target score'], self.settings.edge_vector_threshold)
    if self.color_by in ['trial', 'session']:      
        if self.color_by=='trial':   self.color_counter = trial['trial count']
        if self.color_by=='session': self.color_counter = trial['session count']
        color = get_colormap(object_to_color='plot', plot_type=plot_type)[self.color_counter%16]
    if self.stim_type in ['homing', 'threshold_crossing'] and 'block' in self.session.experiment:
        if trial['frames before laser'] < 6*self.session.video.fps and trial['which side']=='left':
            # color[:3] = color[:3]/20
            color = (0,0,0,1)
    return color

def gradient_line(self, trial: dict):
    points = np.array([trial['trajectory x'], trial['trajectory y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = generate_list_of_colors(self.color_by, self.stim_type, trial['speed'], RT=trial['escape initiation idx'])
    set_of_lines = plt_coll.LineCollection(segments, colors=colors, linewidth=2)
    self.ax.add_collection(set_of_lines)

def apply_x_jitter(self, offset_x=.35, min_distance_y=0.01, jitter_distance_x=0.04):
    data_x = self.data_x.copy()
    if self.settings.x_jitter:
        all_points_adequately_separated = False
        sign_inverter = -1
        multiplier = 1
        while not all_points_adequately_separated:
            all_points_adequately_separated=True
            for round in ['do jitter', 'test separation']:
                for i, data_point_y in enumerate(self.data_y):
                    all_but_this_point_idx      = np.arange(len(self.data_y))!=i
                    y_distances                 = abs(data_point_y - self.data_y[all_but_this_point_idx])
                    y_close_points_idx          = np.where(y_distances < min_distance_y)[0]
                    x_distances_of_close_points = abs(data_x[i] - data_x[all_but_this_point_idx][y_close_points_idx])
                    x_close_points_idx          = np.where(np.round(x_distances_of_close_points,2) < np.round(jitter_distance_x*2,2))[0]
                    if y_close_points_idx.size and x_close_points_idx.size:
                        if round=='do jitter':
                            sign = np.sign(np.mean(data_x[all_but_this_point_idx][y_close_points_idx][x_close_points_idx])+.001) * sign_inverter
                            data_x[i] += sign*jitter_distance_x*multiplier
                        if round=='test separation':
                            all_points_adequately_separated=False
            sign_inverter = 1
            multiplier = 2
    self.jittered_data_x = data_x+offset_x

def format_axis(self):
    self.fig.canvas.set_window_title(self.settings.analysis.title) 
    self.fig.tight_layout()
    plt.axis('off')
    self.ax.margins(0, 0)
    self.ax.xaxis.set_major_locator(plt.NullLocator())
    self.ax.yaxis.set_major_locator(plt.NullLocator())

def rate_map(self,
    var: np.ndarray,
    samples: np.ndarray,
    *,
    bins=20,
    fs: int = 30000,
    min_occ: float = 0,
    smooth_half_win: int = 0,
    smooth_axis: int = 0,
    smooth_pad_type: str = "reflect",
    method: str = "continuous",
    preserve_nan_opt: bool = True,):
    """
    rate_map [summary]
    Parameters
    ----------
    var : np.ndarray
        for multidimensional data this function expect the input to be var[n_samples, n_dim]
    samples : np.ndarray
        can be either a continuous variable of size nsamples or indexes
    bins : int, optional
        [description], by default 10
    fs : int
        sampling rate
    smooth_half_win : np.ndarray
        half window size (in samples) of the smoothing kernel
    smooth_axis : int
        axis along which to apply the smoothing
    smooth_pad_type : str
        type of padding before the smoothing. ex: reflect, circular,...
    method : str, optional
        "point_process" or "continuous", by default "point_process"
    preserve_nan_opt : bool
        specify if nan value should be preserved in the output
    Returns
    -------
    rate : np.ndarray
        [description]
    act : np.ndarray
        [description]
    occ : np.ndarray
        [description]
    Raises
    ------
    ValueError
        in case of error in provided method
    """
    # check inputs
    var = np.asarray(var)
    samples = np.asarray(samples)

    if not method in ["point_process", "continuous"]:
        raise ValueError("Method should be either continuous or point_process")
    if method == "point_process" and samples.dtype.kind != "i":
        samples = float_to_int(samples)

    # first take care of the bins
    if isinstance(bins, (list, np.ndarray)):
        bins = [bins]
    elif isinstance(bins, tuple):
        bins = [*bins]

    # calculate occupancy
    occ, _ = np.histogramdd(var, bins)

    # affect no occupancy to nan. This avoid zero division later
    # and is necessary to take into account non explored areas
    no_occ = occ <= min_occ
    occ[no_occ] = np.nan

    if method == "continuous":
        act, _ = np.histogramdd(var, bins, weights=samples)
        act /= occ
    elif method == "point_process":
        occ = occ / fs  # convert in Hz
        act, _ = np.histogramdd(var[samples], bins)

    # sigma = 1
    # act_s = gaussian_filter(act, sigma=sigma)
    # occ_s = gaussian_filter(occ, sigma=sigma)
   
    act_s, occ_s = act, occ

    if method == "point_process":
        rate_s = act_s / occ_s
    elif method == "continuous":
        rate_s = act_s

    return rate_s, act_s, occ_s

def float_to_int(array) -> np.array:
    """
    float_to_int
    Cast float to int but check value are then equal
    Parameters
    ----------
    array : [type]
        input array
    Returns
    -------
    array:
        output array converted to int
    Raises
    ------
    TypeError
        [description]
    Reference
    ---------
    https://stackoverflow.com/questions/36505879/convert-float64-to-int64-using-safe
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if array.dtype.kind == "i":
        return array

    int_array = array.astype(int, casting="unsafe", copy=True)
    if not np.equal(array, int_array).all():
        raise TypeError(
            "Cannot safely convert float array to int dtype. "
            "Array must only contain whole numbers."
        )
    return int_array
    