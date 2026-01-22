"""
Tuning entanglementdecoupling analysis
This model disentangles the tuning of a neuron to two simultaneously recorded
stimulus variables using conditional independence tests.

Definitions:
v1: driver variable
v2: passenger variable
NH_v1: null hypothesis that the neuron's activity is purely driven by v1

settings:
- bin_edges
- method for comparing tuning curves: 'euclidean' or 'cosine'

TODO: Merge this class with Laurence's TunEd class which does linear shift stats.
ideally this class can perform tuned on any two given variables, with either bootstrap or linear shift stats.

Based on Campagner et al., 2023 Nature
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TunED:
    """Conditional independence test"""

    def __init__(self, settings):
        self.settings = settings

    def estimate_p_conditional(self, v1, v2):
        """Estimate conditional probabilities P(v1 | v2) and P(v2 | v1) from behavioral data"""

        # Compute joint histogram P(v1, v2)
        joint_prob, _, _ = np.histogram2d(v1, v2, bins=[self.settings["bin_edges"], self.settings["bin_edges"]], density=True)

        # Compute marginal probability P(v2)
        Pv1 = np.sum(joint_prob, axis=0)
        Pv2 = np.sum(joint_prob, axis=1)

        # Compute conditional probability P(v1 | v2) (avoid division by zero)
        Pv2_v1 = joint_prob / (np.ones(len(Pv2)).reshape(-1, 1) * (Pv1 + 1e-8))
        Pv1_v2 = joint_prob.T / (np.ones(len(Pv1)).reshape(-1, 1) * (Pv2 + 1e-8))  # P(v1|v2)

        return Pv1_v2, Pv2_v1

    def compute_expected_tuning(self, mu_v1, p_v1_given_v2):
        """
        Compute expected tuning curves under NH that driver is v1.
        INPUTS: mu_v1: mean tuning curve for v1
                   p_v1_given_v2: conditional probability of v1 given v2
        OUTPUT: mu_NH_v2: expected tuning curve for v2 under null hypothesis that driver is v1
        NB: NaN values in the input tuning curve are ignored in the computation
        """
        mu_NH_v2 = np.full(p_v1_given_v2.shape[1], np.nan)
        mask = ~np.isnan(mu_v1)
        if not mask.any():
            return mu_NH_v2

        mu_finite = np.where(mask, mu_v1, 0.0)                # NaNs -> 0 for numerator
        numer = (mu_finite[:, None] * p_v1_given_v2).sum(axis=0)
        denom = p_v1_given_v2[mask, :].sum(axis=0)

        valid = denom > 0
        mu_NH_v2[valid] = numer[valid] / denom[valid]
        
        return mu_NH_v2

    def compare_curves(self, mu_v1, mu_v2, mu_NH_v1, mu_NH_v2):
        """Compute the difference between real and expected tuning curves using:
        - euclidean distance
        - cosine similarity"""

        if self.settings["compare_method"] == "euclidean":
            mask = ~np.isnan(mu_NH_v1) & ~np.isnan(mu_v1)
            dv1 = np.sqrt(np.sum((mu_v1[mask] - mu_NH_v1[mask]) ** 2))
            mask = ~np.isnan(mu_NH_v2) & ~np.isnan(mu_v2)
            dv2 = np.sqrt(np.sum((mu_v2[mask] - mu_NH_v2[mask]) ** 2))

            # Store difference
            return dv1 - dv2

        elif self.settings["compare_method"] == "cosine":
            mask = ~np.isnan(mu_NH_v1) & ~np.isnan(mu_v1)
            cosim1 = cosine_similarity(mu_v1[mask].reshape(1, -1), mu_NH_v1[mask].reshape(1, -1))[0][0]
            mask = ~np.isnan(mu_NH_v2) & ~np.isnan(mu_v2)
            cosim2 = cosine_similarity(mu_v2[mask].reshape(1, -1), mu_NH_v2[mask].reshape(1, -1))[0][0]

            return cosim1 - cosim2
