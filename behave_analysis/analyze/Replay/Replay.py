"""This script defines the EscapeTuning dataclass"""

from dataclasses import dataclass
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import define_bin_edges, parse_side, parse_residual_string


@dataclass(frozen=False)
class Replay:
    selected_cells: bool = False
    template_seq: float = 0.0
    settings: object = None
    time_mask: bool = False
    # rank order correlation results
    rank_order_corr: float = 0.0
    # bayesian decoder results
    bayesian_posterior: float = 0.0
    radon_score: float = 0.0
    radon_angle: float = 0.0
    linear_corr: float = 0.0
    R_max: float = 0.0
    V_max: float = 0.0
    rho_max: float = 0.0
    R_map: float = 0.0
