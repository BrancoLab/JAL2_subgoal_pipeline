import numpy as np
from skimage.transform import radon


def bayesian_decoder(firing_rate_maps, spike_counts, occupancy_map, time_bin_width, n_time_bins, n_position_bins):
    """
    Implements Bayesian decoding to estimate position from neural activity.

    Args:
        firing_rate_maps (np.ndarray): Shape (n_neurons, n_position_bins).
                                       Average firing rate templates.
        spike_counts (np.ndarray): Shape (n_time_bins, n_neurons).
                                   Spike counts in each time bin.
        occupancy_map (np.ndarray): Shape (n_position_bins,).
                                    Normalized occupancy probability P(x).
        time_bin_width (float): Duration of each time bin (τ) in seconds.

    Returns:
        np.ndarray: Shape (n_time_bins, n_position_bins). Posterior probability P(x|n).
    """

    # Add a small epsilon to firing rates to avoid log(0) or 0^0 issues
    epsilon = 1e-10
    firing_rate_maps = np.maximum(firing_rate_maps, epsilon)  # Ensure rates > 0

    # Pre-calculate sum of firing rates over neurons for the exponential term
    sum_fr = np.sum(firing_rate_maps, axis=0)  # Shape: (n_position_bins,)

    # Initialize posterior matrix
    posterior = np.zeros((n_time_bins, n_position_bins))

    # Calculate posterior for each time bin
    for t in range(n_time_bins):
        n_t = spike_counts[t, :]  # Spike counts for this time bin, shape: (n_neurons,)

        # Calculate log-likelihood for numerical stability
        # log(prod(f_i(x)^n_i)) = sum(n_i * log(f_i(x)))
        log_likelihood_term = np.sum(n_t[:, np.newaxis] * np.log(firing_rate_maps), axis=0)

        # Exponential term: exp(-τ * sum(f_i(x)))
        exp_term = -time_bin_width * sum_fr

        # Combine terms in log space: log(P(x)) + log(Likelihood)
        # log(P(x|n)) ~ log(P(x)) + sum(n_i*log(f_i(x))) - τ*sum(f_i(x))
        log_posterior_unnormalized = np.log(occupancy_map + epsilon) + log_likelihood_term + exp_term

        # Convert back from log space and normalize
        # Subtract max for numerical stability before exponentiating
        posterior_unnormalized = np.exp(log_posterior_unnormalized - np.max(log_posterior_unnormalized))
        norm_factor = np.sum(posterior_unnormalized)

        if norm_factor > 0:
            posterior[t, :] = posterior_unnormalized / norm_factor
        else:
            # If sum is zero (e.g., no spikes and zero rates), assign uniform probability
            posterior[t, :] = 1.0 / n_position_bins

    return posterior


# --- Trajectory Evaluation Functions ---


def calculate_radon_score(posterior_matrix):
    """Calculates the Radon score for trajectory linearity."""
    if np.sum(posterior_matrix) == 0:  # Handle empty matrices
        return 0, 0

    theta = np.linspace(0.0, 180.0, max(posterior_matrix.shape), endpoint=False)
    sinogram = radon(posterior_matrix, theta=theta, circle=False)

    # Score: Max variance across angles (or max projection sum)
    radon_score = np.max(np.sum(sinogram**2, axis=0))
    best_angle = theta[np.argmax(np.sum(sinogram**2, axis=0))]

    return radon_score, best_angle


def calculate_linear_weighted_correlation(posterior_matrix):
    """Calculates the maximum linear weighted correlation."""
    n_time_bins, n_position_bins = posterior_matrix.shape
    if n_time_bins < 2 or n_position_bins < 2 or np.sum(posterior_matrix) == 0:
        return 0  # Not enough data or empty matrix

    time_indices = np.arange(n_time_bins)
    position_indices = np.arange(n_position_bins)

    # Create meshgrid for calculations
    T, P = np.meshgrid(time_indices, position_indices, indexing="ij")

    # Calculate weighted means
    total_prob = np.sum(posterior_matrix)
    if total_prob == 0:
        return 0

    mean_t = np.sum(T * posterior_matrix) / total_prob
    mean_p = np.sum(P * posterior_matrix) / total_prob

    # Calculate weighted covariance and variances
    cov_tp = np.sum((T - mean_t) * (P - mean_p) * posterior_matrix) / total_prob
    var_t = np.sum(((T - mean_t) ** 2) * posterior_matrix) / total_prob
    var_p = np.sum(((P - mean_p) ** 2) * posterior_matrix) / total_prob

    # Calculate correlation coefficient
    if var_t > 0 and var_p > 0:
        correlation = cov_tp / np.sqrt(var_t * var_p)
    else:
        correlation = 0

    return correlation


def calculate_custom_replay_score(posterior, time_bin_width, position_bin_edges, d, V_range, rho_range):
    """
    Calculates the replay score based on the custom Radon-like method.

    Args:
        posterior (np.ndarray): Shape (n_time_bins, n_position_bins).
                                Posterior probability matrix P(x|n).
        time_bin_width (float): Duration (Δt) of each time bin in seconds.
        position_bin_edges (np.ndarray): Shape (n_position_bins + 1,).
                                         Boundaries of position bins (e.g., in cm).
        d (float): Distance threshold (e.g., 15 cm).
        V_range (tuple or list): Range of velocities (V) to test (e.g., [min_V, max_V, n_V_steps]).
                                 Units should be consistent with position/time (e.g., cm/s).
        rho_range (tuple or list): Range of starting positions (ρ) to test (e.g., [min_rho, max_rho, n_rho_steps]).
                                   Units should be consistent with position (e.g., cm).

    Returns:
        tuple: (R_max, V_max, rho_max, R_map)
            R_max (float): The maximum replay score found.
            V_max (float): The velocity corresponding to R_max.
            rho_max (float): The starting position corresponding to R_max.
            R_map (np.ndarray): 2D array of scores R(V, ρ).
    """
    n_time_bins, n_position_bins = posterior.shape
    min_pos_cm = position_bin_edges[0]
    max_pos_cm = position_bin_edges[-1]

    # --- Create grids for V and rho ---
    Vs = np.linspace(V_range[0], V_range[1], int(V_range[2]))
    rhos = np.linspace(rho_range[0], rho_range[1], int(rho_range[2]))

    R_map = np.zeros((len(Vs), len(rhos)))  # To store R(V, rho)

    # --- Iterate through parameter space ---
    for i, V in enumerate(Vs):
        for j, rho in enumerate(rhos):
            total_probability_sum = 0.0

            # --- Iterate through time bins ---
            for k in range(n_time_bins):
                predicted_pos_cm = rho + V * k * time_bin_width

                # --- Check if trajectory is off track ---
                if predicted_pos_cm < min_pos_cm or predicted_pos_cm > max_pos_cm:
                    # Use median probability for this time bin
                    median_prob = np.median(posterior[k, :])
                    total_probability_sum += median_prob
                else:
                    # --- Find position bins within distance d ---
                    min_bound_cm = predicted_pos_cm - d
                    max_bound_cm = predicted_pos_cm + d

                    # Find indices of bins that overlap with [min_bound_cm, max_bound_cm]
                    # A bin j overlaps if:
                    # (bin_edges[j] < max_bound_cm) and (bin_edges[j+1] > min_bound_cm)
                    overlapping_bin_indices = np.where((position_bin_edges[:-1] < max_bound_cm) & (position_bin_edges[1:] > min_bound_cm))[0]

                    # Ensure indices are within valid range [0, n_position_bins-1]
                    overlapping_bin_indices = overlapping_bin_indices[(overlapping_bin_indices >= 0) & (overlapping_bin_indices < n_position_bins)]

                    # --- Sum probability within the band ---
                    if len(overlapping_bin_indices) > 0:
                        prob_in_band = np.sum(posterior[k, overlapping_bin_indices])
                        total_probability_sum += prob_in_band
                    # If no bins overlap (shouldn't happen if predicted_pos is on track), add 0

            # --- Calculate average score R for this (V, rho) ---
            if n_time_bins > 0:
                R_map[i, j] = total_probability_sum / n_time_bins

    # --- Find the maximum score and corresponding parameters ---
    if np.any(R_map):  # Check if R_map is not all zeros
        max_idx_flat = np.argmax(R_map)
        max_idx_V, max_idx_rho = np.unravel_index(max_idx_flat, R_map.shape)
        R_max = R_map[max_idx_V, max_idx_rho]
        V_max = Vs[max_idx_V]
        rho_max = rhos[max_idx_rho]
    else:
        R_max = 0.0
        V_max = np.nan
        rho_max = np.nan

    return R_max, V_max, rho_max, R_map
