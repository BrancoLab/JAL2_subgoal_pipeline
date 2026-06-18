
import numpy as np
import os
import dill as pickle
import polars as pl
from matplotlib import pyplot as plt
from scipy.stats import norm

# Select a bunch of variables
x = behaviour['mouse_x_position'].to_numpy()
y = behaviour['mouse_y_position'].to_numpy()
s = behaviour['speed'].to_numpy()
d = np.unwrap(behaviour['hdir'].to_numpy())
h = behaviour['homingPeriod'].to_numpy()

t = np.arange(len(x))

# Create a measure of homing-run-ness: fast turn, followed by high acceleration, to high speed
# Append zeros to keep consistent shape despite taking difference (bit hacky)
d_diff_abs = np.append(np.abs(np.diff(d)), [0])
s_diff_abs = np.append(np.abs(np.diff(s)), [0])
s_abs = np.abs(s)
# Z-score these variables of interest
d_diff_abs_z = (d_diff_abs - np.nanmean(d_diff_abs)) / np.nanstd(d_diff_abs)
s_diff_abs_z = (s_diff_abs - np.nanmean(s_diff_abs)) / np.nanstd(s_diff_abs)
s_abs_z = (s_abs - np.nanmean(s_abs)) / np.nanstd(s_abs)
# Then do smoothing by convolution with an gaussian kernel
# But you want a turn *then* acceleration *then* high speed
# So I'll shift these gaussian kernels by different delays
kernel_width = 50
kernel_shift = 40
kernel = np.exp(-np.linspace(-kernel_shift*3,kernel_shift*3,kernel_shift*6)**2/kernel_width)
kernel = kernel/np.sum(kernel)
# Smooth while shifting backwards in time
d_diff_abs_z_smooth = np.convolve(d_diff_abs_z, kernel, mode='same')
s_diff_abs_z_smooth = np.convolve(s_diff_abs_z, np.roll(kernel,-kernel_shift), mode='same')
s_abs_z_smooth = np.convolve(s_abs_z, np.roll(kernel,-2*kernel_shift), mode='same')
# Then convert these z-values to pseudo-probabilities
def z_to_p_norm(z):
    return 1 - norm.cdf(z)
def z_to_p_log(z):
    return 1 / (1 + np.exp(-z))   
def z_to_p_relu(z):
    return np.clip(z,0, None)
# And define homing score as product of the three
# With loose probabilitistic interpretation: "you need all three"
scores = [d_diff_abs_z_smooth, s_diff_abs_z_smooth, s_abs_z_smooth]
homing_score = np.prod(np.stack([z_to_p_relu(z) for z in scores]), axis=0)
# Now a plot with just the scores for readability
threshold = 7
homing = homing_score>threshold
homing[np.isnan(homing)] = False

# Current homing detection is just the start, so convolve with hat to make it longer
# This is extremely hacky - better to look for deceleration to define homing end
homing_ext = np.convolve(homing, np.concatenate([np.zeros(200), np.ones(200)]), mode='same')>0