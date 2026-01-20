"""
Tuning entanglementdecoupling analysis
This model disentangles the tuning of a neuron to two simultaneously recorded
stimulus variables using conditional independence tests.

Definitions:
v1: driver variable
v2: passenger variable
NH: null hypothesis that the neuron's activity is purely driven by v1

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

    def __init__(self, v1, v2, mu_v1, mu_v2, settings):
        self.v1 = v1  # driver variable
        self.v2 = v2  # passenger variable
        self.mu_v1 = mu_v1  # tuning curve for v1
        self.mu_v2 = mu_v2  # tuning curve for v2
        self.settings = settings

    def independence_test(self):
        """Perform conditional independence test between v1 and v2"""

        # Estimate P(v1 | v2) and P(v2 | v1)
        self.estimate_p_conditional()

        # Compute expected tuning curves under NH
        self.mu_NH_v2 = self.compute_expected_tuning(self.mu_v1, self.Pv1_v2)  # tuning to v2 given that driver is v1
        self.mu_NH_v1 = self.compute_expected_tuning(self.mu_v2, self.Pv2_v1)  # tuning to v1 given that driver is v2

        # Compare actual vs expected tuning for significance
        self.distance = self.compare_curves()

    def estimate_p_conditional(self):
        """Estimate conditional probabilities P(v1 | v2) and P(v2 | v1) from behavioral data"""

        # Compute joint histogram P(v1, v2)
        joint_prob, _, _ = np.histogram2d(self.v1, self.v2, bins=[self.settings["bin_edges"], self.settings["bin_edges"]], density=True)

        # Compute marginal probability P(v2)
        Pv1 = np.sum(joint_prob, axis=0)
        Pv2 = np.sum(joint_prob, axis=1)

        # Compute conditional probability P(v1 | v2) (avoid division by zero)
        self.Pv2_v1 = joint_prob / (np.ones(len(Pv2)).reshape(-1, 1) * (Pv1 + 1e-8))
        self.Pv1_v2 = joint_prob.T / (np.ones(len(Pv1)).reshape(-1, 1) * (Pv2 + 1e-8))  # P(v1|v2)

    def compute_expected_tuning(mu_v1, p_v1_given_v2):
        """INPUTS: mu_v1: mean tuning curve for v1
            p_v1_given_v2: conditional probability of v1 given v2
        OUTPUT: mu_NH_v2: expected tuning curve for v2 under null hypothesis that driver is v1"""
        Nbins_x = p_v1_given_v2.shape[1]
        mu_NH_v2 = np.sum((mu_v1[:, None] @ np.ones((1, Nbins_x))) * p_v1_given_v2, axis=0)  # Turn ty_y into a column vector and multiply by Py_x
        return mu_NH_v2

    def compare_curves(self):
        """Compute the difference between real and expected tuning curves using:
        - euclidean distance
        - cosine similarity"""

        if self.settings["compare_method"] == "euclidean":
            dv1 = np.sqrt(np.sum((self.mu_v1 - self.mu_NH_v1) ** 2))
            dv2 = np.sqrt(np.sum((self.mu_v2 - self.mu_NH_v2) ** 2))

            # Store difference
            return dv1 - dv2

        elif self.settings["compare_method"] == "cosine":
            cosim1 = cosine_similarity(self.mu_v1.reshape(1, -1), self.mu_NH_v1.reshape(1, -1))[0][0]
            cosim2 = cosine_similarity(self.mu_v2.reshape(1, -1), self.mu_NH_v2.reshape(1, -1))[0][0]

            return cosim1 - cosim2
