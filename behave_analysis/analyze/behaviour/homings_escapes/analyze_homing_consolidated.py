"""
Consolidated homing/escape detection pipeline (Phase 1-3).

Simplified workflow:
1. Load session data (speed, hdir, position, labels, metadata)
2. Extract runs (speed threshold + gap merging)
3. Extract features for ALL extracted runs and manually-labelled runs
4. Label extracted runs as exploration (no manual overlap) or target (manual overlap)
5. Compute feature distributions (exploration vs manually-labelled homing/escape)
6. Phase 2: Learn gates from manual distributions OR use manual thresholds
7. Phase 3: Compute overlap metrics (fraction labelled, containment)
8. Syd viewer with manual curation (remove run)
"""

import os

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger
import json
import matplotlib.pyplot as plt
from scipy import stats
from behave_analysis.analyze.behaviour.homings_escapes.homing_load_manual_labels import load_manual_labels
from behave_analysis.database.computer_ID import get_computer_specific_paths
from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.utils.creating_directories import make_directory

@dataclass
class RunFeatures:
    """Features for a single run (extracted or manual)."""
    session_name: str
    run_idx: int
    onset_frame: int
    offset_frame: int
    run_duration: float  # seconds
    is_manual_homing: bool = False  # True if this is a manually-labelled homing
    is_manual_escape: bool = False  # True if this is a manually-labelled escape
    is_extracted: bool = False    # True if this is an extracted run
    is_exploration: bool = False  # True if extracted run has no manual overlap
    
    # Features
    speed_mean: float = np.nan
    speed_peak: float = np.nan
    speed_min: float = np.nan
    head_turn_angle_initial: float = np.nan
    net_distance: float = np.nan
    net_dx: float = np.nan
    net_dy: float = np.nan
    displacement_vertical_ratio: float = np.nan
    mean_initial_acceleration: float = np.nan
    initial_hdir_change_abs: float = np.nan
    mean_initial_angular_head_velocity: float = np.nan
    

@dataclass
class SessionData:
    """Minimal per-session data required for analysis."""
    name: str
    fps: float
    pixel_per_cm: float
    speed: np.ndarray  # shape (n_frames,)
    hdir: np.ndarray   # shape (n_frames,)
    xy_position: np.ndarray  # shape (n_frames, 2)
    is_homing: np.ndarray    # shape (n_frames,) - bool
    is_escape: np.ndarray    # shape (n_frames,) - bool
    barrier_loc: np.ndarray  # shape (n_frames, ...) or None
    shelter_loc: np.ndarray  # shape (n_frames, ...) or None
    barrier_present: np.ndarray  # shape (n_frames,) - bool
    barrier_flipped: np.ndarray  # shape (n_frames,) - bool
    outofshelter_idx: Optional[np.ndarray] = None  # shape (n_frames,) - bool, optional


class HomingAnalyzer:
    """Consolidated homing/escape detection pipeline."""
    
    def __init__(self, experiments_list: List, settings: object):
        """
        Args:
            experiments_list: List of experiment objects
            settings: Settings object containing analysis parameters
        """
        self.experiments = experiments_list
        self.settings = settings
        self.session_data: Dict[str, SessionData] = {}
        self.all_runs: List[RunFeatures] = []
        self.extracted_runs: List[RunFeatures] = []
        self.manual_runs: List[RunFeatures] = []
        self.feature_ranking: List[Tuple[str, Dict]] = []
        self.phase2_gates: Dict[str, Dict] = {}
        self.phase2_selected_features: List[str] = []
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        base_path,_ = get_computer_specific_paths(session_path='', return_ceph=True)
        self.homings_base_path = make_directory(os.path.join(base_path, "Homings"))
        
    def load_all_sessions(self) -> None:
        """Load session data from experiment objects."""
        from behave_analysis.visualize.visualize_utils import open_tracking_data
        from behave_analysis.process.process import Process
        
        logger.info(f"Loading {len(self.experiments)} sessions...")
        
        for exp in self.experiments:
            session_name = exp.nick_name + '_' + exp.experiment_name + '_' + exp.experiment_date
            try:
                session = Process(exp).load_session()
                tracking_data = open_tracking_data(session)
                video_df_path = Path(session.base_path) / session.processed_path / 'full_video_dataframe.csv'
                video_df = pl.read_csv(video_df_path)
                
                # check if manual labels exist
                if os.path.isfile(session.base_path + '/' + session.processed_path + '/Borris/scored_homings.csv'):
                    onset, _, offset = load_manual_labels(session)
                    ishoming = np.zeros(len(video_df), dtype=bool)
                    for on, off in zip(onset, offset):
                        ishoming[on:off + 1] = True

                fps = float(session.video.fps) if hasattr(session.video, 'fps') else 30.0
                pixel_per_cm = float(session.video.pixels_per_cm) if hasattr(session.video, 'pixels_per_cm') else 1.0
                
                self.session_data[session_name] = SessionData(
                    name=session_name,
                    fps=fps,
                    pixel_per_cm=pixel_per_cm,
                    speed=video_df['speed'].to_numpy(),
                    hdir=video_df['hdir'].to_numpy(),
                    xy_position=np.column_stack([
                        tracking_data['avg_loc'][:, 0],
                        tracking_data['avg_loc'][:, 1]
                    ]),
                    is_homing=ishoming,
                    is_escape=video_df['EscapePeriod'].to_numpy(),
                    barrier_loc=tracking_data.get('barrier_loc', None),
                    shelter_loc=tracking_data.get('shelter_loc', None),
                    barrier_present=video_df['barrier_present'].to_numpy(),
                    barrier_flipped=video_df['barrier_flipped'].to_numpy(),
                    outofshelter_idx=video_df['OutofshelterIdx'].to_numpy(),
                )
                logger.info(f"  ✓ {session_name}")
            except Exception as e:
                logger.error(f"  ✗ {session_name}: {e}")
    
    def preloaded_session_data(self, video_df: pl.DataFrame, tracking_data: Dict, session) -> None:
        """If running homing analyzer from analyze_behave or other piepeline where data has been loaded already, 
        this function takes the preloaded data and formats it into the session_data dictionary."""

        if not hasattr(session, "date"):
            import re
            m = re.search(r"(\d{4})_?(\d{2})_?(\d{2})T(\d{2})_?(\d{2})_?(\d{2})$", session.file_path)
            date = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        else:
            date = session.date
        session_name = session.mouse + '_' + session.experiment + '_' + date

        # check if manual labels exist
        ishoming = np.zeros(len(video_df), dtype=bool)
        if os.path.isfile(session.base_path + '/' + session.processed_path + '/Borris/scored_homings.csv'):
            onset, _, offset = load_manual_labels(session)
            for on, off in zip(onset, offset):
                ishoming[on:off + 1] = True

        # ensure runs are only foudn in valid_time by setting all other frames to 0 speed
        speed = video_df['speed'].to_numpy()
        speed[video_df['valid_time'].to_numpy() == False] = 0.0

        # ensure runs are only found outside shelter by setting in shelter frames to 0 speed
        if session.shelter_location is not None:
            in_shelter = video_df['OutofshelterIdx'].to_numpy() == False
            speed[in_shelter] = 0.0

        self.session_data[session_name] = SessionData(
                    name=session_name,
                    fps=float(session.video.fps),
                    pixel_per_cm=float(session.video.pixels_per_cm),
                    speed=speed,
                    hdir=video_df['hdir'].to_numpy(),
                    xy_position=np.column_stack([
                        tracking_data['avg_loc'][:, 0],
                        tracking_data['avg_loc'][:, 1]
                    ]),
                    is_homing=ishoming,
                    is_escape=video_df['EscapePeriod'].to_numpy(),
                    barrier_loc=tracking_data.get('barrier_loc', None),
                    shelter_loc=tracking_data.get('shelter_loc', None),
                    barrier_present=video_df['barrier_present'].to_numpy(),
                    barrier_flipped=video_df['barrier_flipped'].to_numpy(),
                    outofshelter_idx=video_df['OutofshelterIdx'].to_numpy(),
                )
        
    def extract_runs(self, speed_threshold: float = 2.0, gap_tolerance_frames: int = 10) -> None:
        """
        Extract runs based on speed threshold with gap merging.
        No double-counting.
        """
        logger.info(f"Extracting runs (threshold={speed_threshold} cm/s, gap_tolerance={gap_tolerance_frames} frames)...")
        
        self.extracted_runs = []
        
        for session_name, data in self.session_data.items():
            speed = data.speed.copy()
            speed[~np.isfinite(speed)] = 0
            above_threshold = speed >= speed_threshold
            
            # Merge gaps
            padded = np.concatenate([[False], above_threshold, [False]])
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1
            
            # Merge runs separated by < gap_tolerance_frames
            if len(starts) > 0:
                merged_starts = [starts[0]]
                merged_ends = [ends[0]]
                
                for i in range(1, len(starts)):
                    gap = starts[i] - ends[i - 1] - 1
                    if gap < gap_tolerance_frames:
                        merged_ends[-1] = ends[i]
                    else:
                        merged_starts.append(starts[i])
                        merged_ends.append(ends[i])
                
                for run_idx, (on, off) in enumerate(zip(merged_starts, merged_ends)):
                    run_feat = RunFeatures(
                        session_name=session_name,
                        run_idx=run_idx,
                        onset_frame=int(on),
                        offset_frame=int(off),
                        run_duration=float((off - on + 1) / data.fps),
                        is_extracted=True,
                    )
                    self.extracted_runs.append(run_feat)
        
        logger.info(f"  ✓ Extracted {len(self.extracted_runs)} runs across {len(self.session_data)} sessions")
    
    def extract_manual_runs(self) -> None:
        """Extract manually-labelled homing and escape runs."""
        logger.info("Extracting manually-labelled homing/escape runs...")
        
        self.manual_runs = []
        
        for session_name, data in self.session_data.items():
            # Homings
            homing_events = self._contiguous_events(data.is_homing)
            for run_idx, (on, off) in enumerate(homing_events):
                run_feat = RunFeatures(
                    session_name=session_name,
                    run_idx=run_idx,
                    onset_frame=int(on),
                    offset_frame=int(off),
                    run_duration=float((off - on + 1) / data.fps),
                    is_manual_homing=True,
                )
                self.manual_runs.append(run_feat)
            
            # Escapes
            escape_events = self._contiguous_events(data.is_escape)
            for run_idx, (on, off) in enumerate(escape_events):
                run_feat = RunFeatures(
                    session_name=session_name,
                    run_idx=run_idx,
                    onset_frame=int(on),
                    offset_frame=int(off),
                    run_duration=float((off - on + 1) / data.fps),
                    is_manual_escape=True,
                )
                self.manual_runs.append(run_feat)
        
        logger.info(f"  ✓ Extracted {len(self.manual_runs)} manually-labelled runs")
    
    def _contiguous_events(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert boolean mask to list of (onset, offset) contiguous events."""
        mask = np.asarray(mask, dtype=bool)
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))
    
    def _interval_overlap(self, on1: int, off1: int, on2: int, off2: int) -> Dict[str, float]:
        """Compute overlap metrics between two frame intervals."""
        inter = max(0, min(off1, off2) - max(on1, on2) + 1)
        len1 = off1 - on1 + 1
        len2 = off2 - on2 + 1
        union = len1 + len2 - inter
        return {
            'intersection_frames': float(inter),
            'fraction_of_first': float(inter / len1) if len1 > 0 else 0.0,
            'fraction_of_second': float(inter / len2) if len2 > 0 else 0.0,
            'iou': float(inter / union) if union > 0 else 0.0,
        }
    
    def label_extracted_runs(self, overlap_threshold: float = 0.5) -> None:
        """
        Label extracted runs as exploration if they have no significant manual overlap.
        """
        logger.info(f"Labeling extracted runs (overlap_threshold={overlap_threshold})...")
        
        for run in self.extracted_runs:
            # Find all manual runs in the same session
            session_manual = [m for m in self.manual_runs if m.session_name == run.session_name]
            
            # Check overlap with any manual run
            best_overlap = 0.0
            for manual_run in session_manual:
                ov = self._interval_overlap(run.onset_frame, run.offset_frame, 
                                            manual_run.onset_frame, manual_run.offset_frame)
                # Use fraction of extracted run that overlaps manual
                best_overlap = max(best_overlap, ov['fraction_of_first'])
            
            if best_overlap < overlap_threshold:
                run.is_exploration = True
        
        n_exploration = sum(1 for r in self.extracted_runs if r.is_exploration)
        logger.info(f"  ✓ {n_exploration} extracted runs labeled as exploration")
    
    def extract_features(self) -> None:
        """
        Extract features for all runs (extracted and manual).
        """
        logger.info("Extracting features for all runs...")
        
        self.all_runs = []
        
        # Extract features for extracted runs
        for run in self.extracted_runs:
            self._compute_run_features(run)
            self.all_runs.append(run)
        
        # Extract features for manual runs
        for run in self.manual_runs:
            self._compute_run_features(run)
            self.all_runs.append(run)
        
        logger.info(f"  ✓ Extracted features for {len(self.all_runs)} runs")
    
    def _compute_run_features(self, run: RunFeatures) -> None:
        """Compute all features for a single run."""
        data = self.session_data[run.session_name]
        fps = data.fps
        px_cm = data.pixel_per_cm
        
        # Extract run segment
        on, off = run.onset_frame, run.offset_frame
        speed_seg = data.speed[on:off + 1]
        hdir_seg = data.hdir[on:off + 1]
        xy_seg = data.xy_position[on:off + 1]
        
        # Speed stats
        run.speed_mean = float(np.nanmean(speed_seg))
        run.speed_peak = float(np.nanpercentile(speed_seg, 90)) # float(np.nanmax(speed_seg))
        run.speed_min = float(np.nanpercentile(speed_seg, 10)) # float(np.nanmin(speed_seg))

        # Heading at run onset (wrapped to [-pi, pi])
        run.head_turn_angle_initial = float(np.arctan2(np.sin(hdir_seg[0]), np.cos(hdir_seg[0])))
        
        # Distance stats
        disp = xy_seg[-1] - xy_seg[0]
        run.net_distance = float(np.linalg.norm(disp) / px_cm)
        run.net_dx = float(disp[0] / px_cm)
        run.net_dy = float(disp[1] / px_cm)
        run.displacement_vertical_ratio = float(disp[1] / (np.linalg.norm(disp) + 1e-8))
        
        # Initial acceleration (first 1.0s)
        window_frames = int(round(self.settings.homings_features_initial_window_s * fps))
        if len(speed_seg) > window_frames:
            accel = (speed_seg[window_frames] - speed_seg[0]) / 1.0  # cm/s per second
            run.mean_initial_acceleration = float(accel)
        else:
            run.mean_initial_acceleration = np.nan
        
        # Initial head direction change (0.5-1.5s window, circular mean)
        on_0p5 = int(round(0.5 * fps))
        off_1p5 = int(round((self.settings.homings_features_initial_window_s+0.5) * fps))
        if off_1p5 <= len(hdir_seg):
            hdir_window = hdir_seg[on_0p5:off_1p5]
            hdir_change = np.abs(self._circular_mean(hdir_window) - hdir_seg[0])
            hdir_change = np.minimum(hdir_change, 2*np.pi - hdir_change)  # wrap to [-pi, pi]
            run.initial_hdir_change_abs = float(hdir_change)
        else:
            run.initial_hdir_change_abs = np.nan
        
        # Mean initial angular head velocity (first 1.0s, 3-frame step)
        if len(hdir_seg) > 3:
            step = 3
            hdir_diff = np.abs(np.diff(self._wrap_to_pi(hdir_seg[:window_frames]), n=step))
            ang_vel = hdir_diff / (step / fps)  # rad/s
            run.mean_initial_angular_head_velocity = float(np.nanmean(ang_vel))
        else:
            run.mean_initial_angular_head_velocity = np.nan
        
        run.speed_variance = float(np.nanvar(speed_seg))
        run.hdir_variance = float(self._circular_variance(hdir_seg))
    
    def _wrap_to_pi(self, a: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi]."""
        return np.arctan2(np.sin(a), np.cos(a))
    
    def _circular_mean(self, angles: np.ndarray) -> float:
        """Compute circular mean of angles (in radians)."""
        angles = np.asarray(angles, dtype=float)
        angles = angles[np.isfinite(angles)]
        if len(angles) == 0:
            return 0.0
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        return float(np.arctan2(sin_mean, cos_mean))
    
    def _circular_variance(self, theta):
        """Compute circular variance of angles (in radians)."""
        theta = np.asarray(theta, dtype=float)
        theta = theta[np.isfinite(theta)]
        if theta.size == 0:
            return np.nan
        R = np.abs(np.mean(np.exp(1j * theta))) # mean resultant length in [0, 1]
        return 1.0 - R # circular variance in [0, 1]

    def compute_feature_distributions(self) -> Dict:
        """
        Analyze feature distributions for two groups:
        - Exploration (extracted runs with no manual overlap)
        - Target (manually-labelled homing/escape)
        """
        logger.info("Computing feature distributions (exploration vs manually-labelled)...")
        
        exploration_runs = [r for r in self.extracted_runs if r.is_exploration]
        target_runs = self.manual_runs
        
        feature_names = [
            'speed_mean', 'speed_peak', 'speed_variance', 'hdir_variance',
            'head_turn_angle_initial',
            'net_distance', 'net_dx', 'net_dy', 'displacement_vertical_ratio',
            'mean_initial_acceleration', 'initial_hdir_change_abs', 'mean_initial_angular_head_velocity'
        ]
        
        results = {}
        self.feature_ranking = []
        
        for fname in feature_names:
            exp_vals = np.array([getattr(r, fname) for r in exploration_runs], dtype=float)
            tgt_vals = np.array([getattr(r, fname) for r in target_runs], dtype=float)
            
            exp_vals = exp_vals[np.isfinite(exp_vals)]
            tgt_vals = tgt_vals[np.isfinite(tgt_vals)]
            
            if len(exp_vals) == 0 or len(tgt_vals) == 0:
                continue
            
            # Cohen's d
            pooled_std = np.sqrt((np.var(exp_vals) + np.var(tgt_vals)) / 2.0)
            cohens_d = (np.mean(tgt_vals) - np.mean(exp_vals)) / (pooled_std + 1e-8)
            
            # AUC (pooled, rank-based): equivalent to U/(n_pos*n_neg)
            n_exp, n_tgt = len(exp_vals), len(tgt_vals)
            ranks = stats.rankdata(np.concatenate([tgt_vals, exp_vals]))
            rank_sum_tgt = np.sum(ranks[:n_tgt])
            u_tgt = rank_sum_tgt - (n_tgt * (n_tgt + 1)) / 2.0
            auc = float(u_tgt / (n_tgt * n_exp)) if (n_tgt > 0 and n_exp > 0) else np.nan

            # Session-wise AUC to avoid cross-session scale confounds
            session_aucs = []
            for s_name in self.session_data.keys():
                s_exp = np.array([
                    getattr(r, fname) for r in self.extracted_runs
                    if r.session_name == s_name and r.is_exploration
                ], dtype=float)
                s_tgt = np.array([
                    getattr(r, fname) for r in self.manual_runs
                    if r.session_name == s_name
                ], dtype=float)
                s_exp = s_exp[np.isfinite(s_exp)]
                s_tgt = s_tgt[np.isfinite(s_tgt)]
                if len(s_exp) == 0 or len(s_tgt) == 0:
                    continue
                s_ranks = stats.rankdata(np.concatenate([s_tgt, s_exp]))
                s_rank_sum_tgt = np.sum(s_ranks[:len(s_tgt)])
                s_u_tgt = s_rank_sum_tgt - (len(s_tgt) * (len(s_tgt) + 1)) / 2.0
                s_auc = float(s_u_tgt / (len(s_tgt) * len(s_exp)))
                session_aucs.append(s_auc)

            # p-value: keep non-parametric rank test for significance reporting
            _, pval = stats.ranksums(tgt_vals, exp_vals)
            
            results[fname] = {
                'n_exploration': n_exp,
                'n_target': n_tgt,
                'exp_mean': float(np.mean(exp_vals)),
                'tgt_mean': float(np.mean(tgt_vals)),
                'cohens_d': float(cohens_d),
                'auc': float(auc),
                'auc_session_mean': float(np.mean(session_aucs)) if len(session_aucs) > 0 else np.nan,
                'auc_session_n': int(len(session_aucs)),
                'pval': float(pval),
            }
            
            self.feature_ranking.append((fname, results[fname]))
        
        # Sort by |Cohen's d|
        self.feature_ranking.sort(key=lambda x: abs(x[1]['cohens_d']), reverse=True)
        
        logger.info("  Feature ranking (by |Cohen's d|):")
        for idx, (fname, stats_dict) in enumerate(self.feature_ranking):
            logger.info(
                f"    {idx+1}. {fname}: d={stats_dict['cohens_d']:.3f}, "
                f"AUC={stats_dict['auc']:.3f}, p={stats_dict['pval']:.3e}"
            )
        
        return results
    
    def plot_feature_distributions(self, only_top_n: int = 6, n_bins: int = 20) -> None:
        """
        Plot feature distributions as session-wise line histograms.

        Manual target (homing/escape) is blue and exploration is orange.
        For each feature, both groups use identical bins defined by curated x-limits
        when available; otherwise they fall back to pooled finite min/max.
        """
        if not hasattr(self, 'feature_ranking') or len(self.feature_ranking) == 0:
            raise RuntimeError("Run compute_feature_distributions() first")

        xlim_map = {
            'net_distance': (0.0, 90.0),
            'net_dy': (-75.0, 75.0),
            'speed_mean': (0.0, 60.0),
            'speed_peak': (0.0, 150.0),
            'mean_initial_acceleration': (0.0, 50.0),
            'displacement_vertical_ratio': (-1.0, 1.0),
            'run_duration': (0.0, 50.0),
            'net_dx': (-75.0, 75.0),
            'head_turn_angle_initial': (-np.pi, np.pi),
            'initial_hdir_change_abs': (0.0, np.pi),
            'mean_initial_angular_head_velocity': (0.0, 5.0),
            'speed_variance': (0.0, 300.0),
            'hdir_variance': (0.0, 1.5),
        }
        
        n_plots = min(only_top_n, len(self.feature_ranking))
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axs = axs.flatten()

        session_names = list(self.session_data.keys())

        for idx, (fname, _) in enumerate(self.feature_ranking[:n_plots]):
            ax = axs[idx]

            pooled = np.array([
                getattr(r, fname)
                for r in (self.extracted_runs + self.manual_runs)
                if np.isfinite(getattr(r, fname, np.nan))
            ], dtype=float)
            if pooled.size == 0:
                ax.axis('off')
                continue

            if fname in xlim_map:
                x_min, x_max = xlim_map[fname]
            else:
                x_min, x_max = float(np.nanmin(pooled)), float(np.nanmax(pooled))
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                    x_min, x_max = float(x_min) - 1.0, float(x_max) + 1.0

            bin_edges = np.linspace(x_min, x_max, int(n_bins) + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            for session_name in session_names:
                session_explore = [
                    r for r in self.extracted_runs
                    if r.session_name == session_name and r.is_exploration
                ]
                session_target = [
                    r for r in self.manual_runs
                    if r.session_name == session_name
                ]

                exp_vals = np.array([getattr(r, fname) for r in session_explore], dtype=float)
                tgt_vals = np.array([getattr(r, fname) for r in session_target], dtype=float)

                exp_vals = exp_vals[np.isfinite(exp_vals)]
                tgt_vals = tgt_vals[np.isfinite(tgt_vals)]

                if exp_vals.size > 1:
                    exp_counts, _ = np.histogram(exp_vals, bins=bin_edges)
                    exp_density = exp_counts.astype(float) / max(1.0, float(np.sum(exp_counts)))
                    ax.plot(bin_centers, exp_density, color='orange', alpha=0.45, linewidth=1.3)

                if tgt_vals.size > 1:
                    tgt_counts, _ = np.histogram(tgt_vals, bins=bin_edges)
                    tgt_density = tgt_counts.astype(float) / max(1.0, float(np.sum(tgt_counts)))
                    ax.plot(bin_centers, tgt_density, color='blue', alpha=0.45, linewidth=1.3)

            axs[idx].set_xlabel(fname)
            axs[idx].set_ylabel('fraction in bin')
            axs[idx].set_xlim(x_min, x_max)
        
        for idx in range(n_plots, len(axs)):
            axs[idx].axis('off')

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='blue', lw=2.0, label='manual homing/escape (per session)'),
            Line2D([0], [0], color='orange', lw=2.0, label='exploration (per session)'),
        ]
        fig.legend(handles=legend_handles, loc='upper right', fontsize=9)
        
        plt.tight_layout()

        fig_path = os.path.join(self.homings_base_path, "feature_distributions.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved feature distributions to {fig_path}")
    
    def fit_classification_gates(self, target_recall: float = 0.90) -> Dict[str, Dict]:
        """
        Fit single-feature binary gates using manually-labelled runs.
        Returns learned gates.
        """
        logger.info(f"Fitting gates (target_recall={target_recall})...")
        
        exploration_runs = [r for r in self.extracted_runs if r.is_exploration]
        target_runs = self.manual_runs
        
        self.phase2_gates = {}
        ranking_stats = {fname: sd for fname, sd in self.feature_ranking}

        for fname, _ in self.feature_ranking:
            tgt_vals = np.array([getattr(r, fname) for r in target_runs], dtype=float)
            exp_vals = np.array([getattr(r, fname) for r in exploration_runs], dtype=float)
            
            tgt_vals = tgt_vals[np.isfinite(tgt_vals)]
            exp_vals = exp_vals[np.isfinite(exp_vals)]
            
            if len(tgt_vals) == 0 or len(exp_vals) == 0:
                continue
            
            # Determine direction
            direction = '>=' if np.mean(tgt_vals) >= np.mean(exp_vals) else '<='
            
            # Find threshold
            pool = np.unique(np.concatenate([tgt_vals, exp_vals]))
            if len(pool) > 200:
                pool = np.quantile(pool, np.linspace(0.01, 0.99, 200))
            
            best_gate_recall = None  # meets target_recall, highest precision
            best_gate_f1 = None      # best F1 overall (fallback)
            best_f1_score = -1.0

            for thr in pool:
                if direction == '>=':
                    tp = np.sum(tgt_vals >= thr)
                    fp = np.sum(exp_vals >= thr)
                else:
                    tp = np.sum(tgt_vals <= thr)
                    fp = np.sum(exp_vals <= thr)
                
                recall = tp / len(tgt_vals) if len(tgt_vals) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

                gate_entry = {
                    'feature': fname,
                    'dir': direction,
                    'threshold': float(thr),
                    'recall': float(recall),
                    'precision': float(precision),
                }

                if recall >= target_recall:
                    if best_gate_recall is None or precision > best_gate_recall['precision']:
                        best_gate_recall = gate_entry
                
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_gate_f1 = gate_entry
            
            best_gate = best_gate_recall if best_gate_recall is not None else best_gate_f1
            if best_gate is not None:
                # Enrich with distribution-level stats so the saved file is self-contained
                fd = ranking_stats.get(fname, {})
                best_gate['cohens_d']        = float(fd.get('cohens_d', np.nan))
                best_gate['auc']             = float(fd.get('auc', np.nan))
                best_gate['auc_session_mean']= float(fd.get('auc_session_mean', np.nan))
                best_gate['n_target']        = int(fd.get('n_target', 0))
                best_gate['n_exploration']   = int(fd.get('n_exploration', 0))
                self.phase2_gates[fname] = best_gate
                logger.info(
                    f"  {fname}: keep if value {best_gate['dir']} {best_gate['threshold']:.4f} "
                    f"(recall={best_gate['recall']:.3f}, precision={best_gate['precision']:.3f})"
                )
        
        self.phase2_selected_features = list(self.phase2_gates.keys())

        gates_path = os.path.join(self.homings_base_path, "fitted_homing_classification_gates.json")
        with open(gates_path, 'w') as f:
            json.dump(self.phase2_gates, f, indent=2)
        logger.info(f"  ✓ Saved {len(self.phase2_gates)} gates to {gates_path}")

        return self.phase2_gates
    
    def run_classification(
        self,
        manual_gates: Optional[Dict[str, Dict]] = None,
        use_learned_gates: bool = True,
    ) -> List[Tuple[int, int]]:
        """
        Homing classification.
        
        Args:
            manual_gates: Optional dict of {feature_name: {'dir': '>=', 'threshold': value}}
            use_learned_gates: If True, load gates from disk and filter by target_recall
            target_recall: Minimum recall a gate must achieve to be used (only applied when use_learned_gates=True)
        """
        if not use_learned_gates and manual_gates is None:
            raise ValueError("Must provide manual_gates if use_learned_gates=False")

        if use_learned_gates:
            gates_path = os.path.join(self.homings_base_path, "fitted_homing_classification_gates.json")
            # if not gates_path.exists():
            #     raise RuntimeError(f"Gates file not found: {gates_path}. Run fit_classification_gates() first.")
            with open(gates_path, 'r') as f:
                all_gates = json.load(f)
            gates = {
                        k: v for k, v in all_gates.items()
                        if v.get('recall', 0.0)           >= self.settings.homings_classification_recall_threshold
                        and v.get('precision', 0.0)        >= self.settings.homings_classification_precision_threshold
                        and v.get('auc', 0.0)              >= self.settings.homings_classification_auc_threshold
                        and abs(v.get('cohens_d', 0.0))    >= self.settings.homings_classification_cohens_d_threshold
                    }
            self.classification_gates = gates
            logger.info(
                        f"  Loaded {len(all_gates)} gates; {len(gates)} pass filters "
                        f"(recall>={self.settings.homings_classification_recall_threshold}, precision>={self.settings.homings_classification_precision_threshold}, "
                        f"auc>={self.settings.homings_classification_auc_threshold}, |d|>={self.settings.homings_classification_cohens_d_threshold})"
                    )
        else:
            gates = manual_gates
        logger.info(f"Running Homing classification with {len(gates)} gates...")
        
        candidates = []
        candidates_by_session = {s: [] for s in self.session_data.keys()}
        candidate_meta_by_session = {s: [] for s in self.session_data.keys()}
        for run in self.extracted_runs:
            passed = True
            for fname, gate_info in gates.items():
                val = getattr(run, fname, np.nan)
                if not np.isfinite(val):
                    passed = False
                    break
                
                if gate_info['dir'] == '>=':
                    if val < gate_info['threshold']:
                        passed = False
                        break
                else:  # '<='
                    if val > gate_info['threshold']:
                        passed = False
                        break
            
            if passed:
                candidates.append((run.onset_frame, run.offset_frame))
                candidates_by_session[run.session_name].append((run.onset_frame, run.offset_frame))
                candidate_meta_by_session[run.session_name].append({
                    'run_idx': run.run_idx,
                    'onset': run.onset_frame,
                    'offset': run.offset_frame,
                    'feature_values': {fname: float(getattr(run, fname, np.nan)) for fname in gates.keys()},
                })

        self.candidates_by_session = candidates_by_session
        self.candidate_meta = candidate_meta_by_session
        self.gates_used = gates
        
        logger.info(f"  ✓ {len(candidates)} runs passed Homing classification gates")
        return candidates
    
    def compute_manualvsauto_overlap(
        self,
        candidates: Optional[List[Tuple[int, int]]] = None,
        candidates_by_session: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        recovered_fraction_threshold: float = 0.30,
    ) -> Dict:
        """
        Compute Phase 3 overlap metrics.

        For each extracted run:
            - fraction of frames that were manually labelled
            - recovered if best overlap with any manual run is >= recovered_fraction_threshold

        For each manually-labelled run:
            - best fraction of the manual run covered by any extracted run
            - recovered if best overlap with any extracted run is >= recovered_fraction_threshold

        Session summary reports:
            - mean fraction of extracted frames that are manually labelled
            - fraction of extracted runs recovered
            - fraction of manual runs recovered
        """
        if candidates_by_session is None:
            if hasattr(self, 'candidates_by_session'):
                candidates_by_session = self.candidates_by_session
            elif candidates is not None and len(self.session_data) == 1:
                only_session = list(self.session_data.keys())[0]
                candidates_by_session = {only_session: list(candidates)}
            else:
                logger.error(
                    "Session-wise candidates are required for accurate Phase 3 overlap. "
                    "Run run_phase2_classification() first or pass candidates_by_session."
                )
                return {}

        if candidates is not None and candidates_by_session is not None and hasattr(self, 'phase2_candidates_by_session'):
            logger.warning(
                "Ignoring flat candidates list in Phase 3 because session-wise candidates are available."
            )

        logger.info("Computing Phase 3 overlap metrics...")

        results = {'by_session': {}, 'summary': {}}

        for session_name, data in self.session_data.items():
            manual_homing = self._contiguous_events(data.is_homing)
            manual_escape = self._contiguous_events(data.is_escape)
            all_manual = manual_homing + manual_escape

            session_candidates = list(candidates_by_session.get(session_name, []))

            extracted_metrics = []
            recovered_extracted_count = 0

            for c_on, c_off in session_candidates:
                n_frames_extracted = c_off - c_on + 1
                n_frames_manual_overlap = 0
                best_fraction_extracted_covered = 0.0
                n_manual_runs_overlapping = 0

                for m_on, m_off in all_manual:
                    inter = max(0, min(c_off, m_off) - max(c_on, m_on) + 1)
                    if inter > 0:
                        n_frames_manual_overlap += inter
                        n_manual_runs_overlapping += 1
                        fraction_extracted_covered = inter / n_frames_extracted if n_frames_extracted > 0 else 0.0
                        best_fraction_extracted_covered = max(best_fraction_extracted_covered, fraction_extracted_covered)

                fraction_manually_labelled = n_frames_manual_overlap / n_frames_extracted if n_frames_extracted > 0 else np.nan
                is_recovered = best_fraction_extracted_covered >= recovered_fraction_threshold
                if is_recovered:
                    recovered_extracted_count += 1

                extracted_metrics.append({
                    'onset': c_on,
                    'offset': c_off,
                    'n_manual_overlap_frames': n_frames_manual_overlap,
                    'fraction_manually_labelled': fraction_manually_labelled,
                    'best_fraction_extracted_covered': best_fraction_extracted_covered,
                    'is_recovered': is_recovered,
                    'recovered_threshold': recovered_fraction_threshold,
                    'n_manual_runs_overlapping': n_manual_runs_overlapping,
                })

            manual_metrics = []
            recovered_manual_count = 0
            multi_extracted_count = 0

            for m_on, m_off in all_manual:
                n_frames_manual = m_off - m_on + 1
                n_frames_extracted_overlap = 0
                best_fraction_manual_covered = 0.0
                n_extracted_runs_overlapping = 0

                for c_on, c_off in session_candidates:
                    inter = max(0, min(c_off, m_off) - max(c_on, m_on) + 1)
                    if inter > 0:
                        n_frames_extracted_overlap += inter
                        n_extracted_runs_overlapping += 1
                        fraction_manual_covered = inter / n_frames_manual if n_frames_manual > 0 else 0.0
                        best_fraction_manual_covered = max(best_fraction_manual_covered, fraction_manual_covered)

                is_recovered = best_fraction_manual_covered >= recovered_fraction_threshold
                if is_recovered:
                    recovered_manual_count += 1
                if n_extracted_runs_overlapping > 1:
                    multi_extracted_count += 1

                manual_metrics.append({
                    'onset': m_on,
                    'offset': m_off,
                    'n_extracted_overlap_frames': n_frames_extracted_overlap,
                    'best_fraction_manual_covered': best_fraction_manual_covered,
                    'is_recovered': is_recovered,
                    'recovered_threshold': recovered_fraction_threshold,
                    'n_extracted_runs_overlapping': n_extracted_runs_overlapping,
                })

            mean_frac_manual = np.mean([m['fraction_manually_labelled'] for m in extracted_metrics]) if extracted_metrics else np.nan
            fraction_extracted_recovered = recovered_extracted_count / len(session_candidates) if session_candidates else np.nan
            fraction_manual_recovered = recovered_manual_count / len(all_manual) if all_manual else np.nan

            results['by_session'][session_name] = {
                'n_extracted': len(session_candidates),
                'n_manual': len(all_manual),
                'extracted_metrics': extracted_metrics,
                'manual_metrics': manual_metrics,
                'mean_fraction_extracted_that_are_manually_labelled': mean_frac_manual,
                'fraction_extracted_recovered': fraction_extracted_recovered,
                'recovered_extracted_count': recovered_extracted_count,
                'fraction_manual_recovered': fraction_manual_recovered,
                'recovered_manual_count': recovered_manual_count,
                'recovered_threshold': recovered_fraction_threshold,
                'fraction_manual_with_multiple_extracted': multi_extracted_count / len(all_manual) if all_manual else np.nan,
            }

        logger.info("  ✓ Phase 3 overlap computed")
        return results
    
    def create_syd_viewer(
        self,
        candidates: List[Tuple[int, int]],
        session_name: Optional[str] = None,
        include_manual_events_in_iteration: bool = True,
    ):
        """
        Create interactive Syd viewer for exploring extracted runs and manual labels.
        Optional: save removed runs to CSV.
        
        Args:
            candidates: List of (onset, offset) tuples if only plotting one session, or a dict of {session_name: [(onset, offset), ...]} if plotting multiple sessions
            session_name: Filter to single session (if None, includes all), or if pasing a list of tuples give name as str
        """
        from syd import make_viewer
        
        if session_name is None:
            if isinstance(candidates, list):
                logger.error("Session name must be provided when candidates is a flat list. Pass candidates as a dict of {session_name: [(onset, offset), ...]} instead.")
                return
            session_names = list(candidates.keys())
            candidates_by_session = candidates
        else:
            session_names = [session_name]
            candidates_by_session = {session_name: list(candidates)}
        
        viewer = make_viewer()
        viewer.add_integer('trial', min=0, max=500, value=0)
        viewer.add_integer('speed_thresh', min=0, max=20, value=self.settings.homings_speed_threshold)
        viewer.add_integer('session_idx', min=0, max=len(session_names) - 1, value=0)
        viewer.add_integer('include_manual_events', min=0, max=1, value=1 if include_manual_events_in_iteration else 0)
        
        def see_next(state):
            viewer.update_integer("trial", value=state["trial"] + 1)
        viewer.add_button("see_next", label="See next", callback=see_next)
        
        def plot(state):
            idx = int(state['trial'])
            s_name = session_names[int(state['session_idx'])]
            data = self.session_data[s_name]
            fps = data.fps
            n_frames = len(data.speed)
            cand_list = np.array(candidates_by_session.get(s_name, []), dtype=int)
            if cand_list.size == 0:
                cand_on = np.array([], dtype=int)
                cand_off = np.array([], dtype=int)
            else:
                cand_on = cand_list[:, 0]
                cand_off = cand_list[:, 1]

            manual_mask_full = np.logical_or(data.is_homing.astype(bool), data.is_escape.astype(bool))
            man_events = self._contiguous_events(manual_mask_full)
            if len(man_events) == 0:
                man_on = np.array([], dtype=int)
                man_off = np.array([], dtype=int)
            else:
                man_on = np.array([m[0] for m in man_events], dtype=int)
                man_off = np.array([m[1] for m in man_events], dtype=int)

            candidate_mask = np.zeros(n_frames, dtype=bool)
            for on_i, off_i in zip(cand_on, cand_off):
                candidate_mask[on_i:off_i + 1] = True

            manual_mask = manual_mask_full

            iterate_manual = int(state.get('include_manual_events', 1)) == 1
            if iterate_manual:
                all_onsets = np.concatenate([cand_on, man_on]) if len(man_on) > 0 else cand_on.copy()
                all_offsets = np.concatenate([cand_off, man_off]) if len(man_off) > 0 else cand_off.copy()
                all_source = np.concatenate([
                    np.zeros(len(cand_on), dtype=int),
                    np.ones(len(man_on), dtype=int)
                ]) if len(man_on) > 0 else np.zeros(len(cand_on), dtype=int)
            else:
                all_onsets = cand_on.copy()
                all_offsets = cand_off.copy()
                all_source = np.zeros(len(cand_on), dtype=int)

            if len(all_onsets) == 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 3))
                ax.axis('off')
                mode_txt = 'candidates + manual' if iterate_manual else 'candidates only'
                ax.text(0.02, 0.8, f"No events to iterate in {s_name} ({mode_txt})", fontsize=11)
                return fig

            order = np.argsort(all_onsets)
            all_onsets = all_onsets[order]
            all_offsets = all_offsets[order]
            all_source = all_source[order]

            if idx > len(all_onsets) - 1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 3))
                ax.axis('off')
                ax.text(0.02, 0.8, f"Trial index {idx} exceeds number of events ({len(all_onsets)}) in session {s_name}.")
                return fig

            onset = int(all_onsets[idx])
            offset = int(all_offsets[idx])

            pad = int(round(fps))
            start = max(0, onset - pad)
            stop = min(n_frames, offset + pad + 1)

            t = (np.arange(start, stop) - start) / fps
            x = data.xy_position[start:stop, 0]
            y = data.xy_position[start:stop, 1]
            speed = data.speed[start:stop]
            hdir = data.hdir[start:stop]
            if data.barrier_present[start] == False:
                    condition = 'shelter_only'
            elif data.barrier_present[start] == True:
                condition = 'barrier_pre_flip'
                if data.barrier_flipped[start] == True:
                    condition = 'barrier_post_flip'

            cand_win = candidate_mask[start:stop]
            man_win = manual_mask[start:stop]

            fig, axs = plt.subplots(3, 2, figsize=(10, 11), gridspec_kw={'height_ratios': [2, 1, 1]})

            axs[0, 0].plot(x, y, color='0.75', lw=1)
            axs[0, 0].scatter(x[cand_win], y[cand_win], s=8, color='tab:blue', alpha=0.8, label='candidate')
            axs[0, 0].scatter(x[0], y[0], s=35, color='green', label='window start')
            axs[0, 0].scatter(x[-1], y[-1], s=35, color='red', label='window end')
            Arena(ax = axs[0, 0], condition = condition,
                barrier_coordinates = data.barrier_loc[:-1],
                shelter_coordinates = data.shelter_loc, full_image = False)

            axs[0, 1].scatter(x[man_win], y[man_win], s=8, color='tab:orange', alpha=0.8, label='manual')
            axs[0, 1].plot(x, y, color='0.85', lw=1)
            Arena(ax = axs[0, 1], condition = condition,
                barrier_coordinates = data.barrier_loc[:-1],
                shelter_coordinates = data.shelter_loc, full_image = False)

            src = 'candidate' if all_source[idx] == 0 else 'manual'
            axs[0, 0].set_title(f'{s_name} | trial {idx} ({src})')
            axs[0, 0].set_xlim(0, 1024)
            axs[0, 0].set_ylim(0, 1024)
            axs[0, 0].set_aspect('equal')
            axs[0, 0].invert_yaxis()
            axs[0, 0].legend(loc='upper right', fontsize=8)

            axs[0, 1].set_xlim(0, 1024)
            axs[0, 1].set_ylim(0, 1024)
            axs[0, 1].set_aspect('equal')
            axs[0, 1].invert_yaxis()

            axs[1, 0].plot(t, speed, color='black', lw=1)
            axs[1, 0].fill_between(t, 0, speed, where=cand_win, color='tab:blue', alpha=0.25)
            ax_r0 = axs[1, 0].twinx()
            ax_r0.plot(t, speed > state['speed_thresh'], color='tab:red', lw=1)

            axs[1, 1].plot(t, speed, color='black', lw=1)
            axs[1, 1].fill_between(t, 0, speed, where=man_win, color='tab:orange', alpha=0.18)
            ax_r1 = axs[1, 1].twinx()
            ax_r1.plot(t, speed > state['speed_thresh'], color='tab:red', lw=1)

            axs[1, 0].axvline((onset - start) / fps, color='tab:green', ls='--', lw=1)
            axs[1, 0].axvline((offset - start) / fps, color='tab:red', ls='--', lw=1)
            axs[1, 0].set_ylabel('speed')

            axs[2, 0].plot(t, hdir, color='slategray', lw=1)
            axs[2, 0].fill_between(t, np.nanmin(hdir), np.nanmax(hdir), where=cand_win, color='tab:blue', alpha=0.12)
            axs[2, 1].plot(t, hdir, color='slategray', lw=1)
            axs[2, 1].fill_between(t, np.nanmin(hdir), np.nanmax(hdir), where=man_win, color='tab:orange', alpha=0.10)
            axs[2, 0].axvline((onset - start) / fps, color='tab:green', ls='--', lw=1)
            axs[2, 0].axvline((offset - start) / fps, color='tab:red', ls='--', lw=1)
            axs[2, 0].set_ylabel('hdir')
            axs[2, 0].set_xlabel('time (s)')

            overlap = int(np.sum(candidate_mask[onset:offset + 1] & manual_mask[onset:offset + 1]))
            manual_count = int(np.sum(manual_mask[onset:offset + 1]))
            cand_count = int(np.sum(candidate_mask[onset:offset + 1]))
            axs[0, 1].set_title(f'candidate frames={cand_count} | manual frames={manual_count} | overlap={overlap}')

            plt.tight_layout()
            return fig

        viewer.set_plot(plot)

        return viewer
    
