"""This script defines the EscapeTuning dataclass"""

from dataclasses import dataclass
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import define_bin_edges, parse_side, parse_residual_string


@dataclass(frozen=False)
class Replay:
    selected_cells: bool
    template_seq: float
    settings: object
    time_mask: bool
    # rank order correlation results
    rank_order_corr: float
    # bayesian decoder results
    bayesian_posterior: float
    radon_score: float
    radon_angle: float
    linear_corr: float
    R_max: float
    V_max: float
    rho_max: float
    R_map: float
