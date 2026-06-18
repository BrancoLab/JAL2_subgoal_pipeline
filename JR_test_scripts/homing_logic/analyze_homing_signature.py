# File: behave_analysis/analyze/analyze_homing_signatures.py

"""Phase 1: Extract behavioral signatures of target runs (homings+escapes) vs exploration.

This module:
1. Loads manual labels from all 20 sessions
2. Defines "runs" as sustained locomotion bouts (speed > threshold)
3. Validates the hypothesis that homings+escapes are exclusively within runs
4. Extracts per-run features for target vs exploration runs
5. Computes feature distributions and ranks by discrimination power
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from scipy import stats
import dill as pickle

from behave_analysis.process.process import Process
from behave_analysis.visualize.visualize_utils import open_tracking_data


@dataclass
class RunFeatures:
    """Per-run feature container."""
    run_idx: int
    session_name: str
    onset_frame: int
    offset_frame: int
    is_target: bool  # True = homings+escapes, False = exploration
    
    # Speed features
    speed_mean: float
    speed_peak: float
    mean_initial_acceleration: float
    
    # Heading dynamics
    head_turn_angle_initial: float  # initial turn angle relative to shelter
    initial_hdir_change_abs: float
    mean_initial_angular_head_velocity: float
    
    # Trajectory geometry
    net_dx: float  # net displacement x
    net_dy: float  # net displacement y
    net_distance: float
    displacement_vertical_ratio: float  # dy / net_distance
    
    # Movement phases
    prelude_duration: float  # stationary time before run
    run_duration: float  # total run duration
    
    # Spatial context
    start_x: float
    start_y: float
    end_x: float
    end_y: float


class PhaseOneAnalyzer:
    """Phase 1 analyzer: Extract and compare signatures of target runs vs exploration."""
    
    def __init__(self, experiments_dict: Dict, output_dir: str = None):
        """
        Initialize analyzer.
        
        Args:
            experiments_dict: Dict mapping session_name -> Experiment object
            output_dir: Directory to save results (default: behave_analysis/analyze/homing_analysis_results/)
        """
        self.experiments_dict = experiments_dict
        self.output_dir = Path(output_dir) if output_dir else \
            Path("behave_analysis/analyze/homing_analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_data = {}  # {session_name: {...session_info...}}
        self.all_runs = []  # List[RunFeatures]
        self.hypothesis_results = {}
        
    def load_all_sessions(self):
        """Load data for all sessions."""
        logger.info(f"Loading {len(self.experiments_dict)} sessions...")
        for session_name, exp_obj in self.experiments_dict.items():
            logger.info(f"Loading session: {session_name}")
            try:
                session = Process(exp_obj).load_session()
                tracking_data = open_tracking_data(session)
                
                video_df_path = os.path.join(
                    session.base_path, session.processed_path, 'full_video_dataframe.csv'
                )
                video_df = pl.read_csv(video_df_path)

                # load escape_obj
                esc_path = os.path.join(session.base_path, session.processed_path, "escapes", "escapes_obj.pkl")
                if os.path.exists(esc_path):
                    logger.info("Escape object found. Loading...")
                    with open(esc_path, "rb") as dill_file:
                        escapes = pickle.load(dill_file)
                onset_frames = escapes.onset_frames
                offset_frames = escapes.offset_frames
                escape_bool = np.zeros(len(video_df), dtype=bool)
                for onset, offset in zip(onset_frames, offset_frames):
                    escape_bool[onset: offset + 1] = True

                video_df = video_df.with_columns(pl.Series("EscapePeriod", escape_bool.astype(bool)))
                
                # Load manual labels
                # manual_labels = self._load_manual_labels(session)
                
                self.session_data[session_name] = {
                    'session': session,
                    'tracking_data': tracking_data,
                    'video_df': video_df,
                    # 'manual_labels': manual_labels,  # (onsets, offsets in frames, 0-indexed)
                    'fps': session.video.fps,
                    'pixel_per_cm': session.video.pixels_per_cm,
                }
                logger.success(f"  Loaded {session_name}")
            except Exception as e:
                logger.error(f"Failed to load {session_name}: {e}")
    
    def _load_manual_labels(self, session):
        """Load manual labels from BORIS CSV."""
        boris_path = os.path.join(
            session.base_path, session.processed_path, "Borris", "scored_homings.csv"
        )
        if not os.path.exists(boris_path):
            logger.warning(f"  No manual labels found at {boris_path}")
            return (np.array([], dtype=int), np.array([], dtype=int))
        
        df = pd.read_csv(boris_path)
        onsets = df[df["Behavior type"] == "START"]["Image index"].to_numpy(dtype=int) - 1
        offsets = df[df["Behavior type"] == "STOP"]["Image index"].to_numpy(dtype=int) - 1
        return (onsets, offsets)
    
    def create_frame_labels(self):
        """Create frame-level labels: target (homings+escapes) vs exploration."""
        logger.info("Creating frame-level labels...")
        for session_name, data in self.session_data.items():
            n_frames = len(data['video_df'])
            
            # Initialize frame labels
            is_target = np.zeros(n_frames, dtype=bool)
            is_in_shelter = ~data['video_df']['OutofshelterIdx'].to_numpy()
            
            # Mark target frames (from manual labels)
            is_target = np.logical_or(data['video_df']['homingPeriod'].to_numpy(), data['video_df']['EscapePeriod'].to_numpy())
            
            # Exploration = outside shelter AND outside target periods
            is_exploration = (~is_in_shelter) & ~is_target
            
            data['frame_labels'] = {
                'is_target': is_target,
                'is_exploration': is_exploration,
                'is_in_shelter': is_in_shelter,
            }
            
            logger.info(f"  {session_name}: {np.sum(is_target)} target frames, "
                       f"{np.sum(is_exploration)} exploration frames")
    
    def extract_runs(self, speed_threshold: float = 2.0, gap_tolerance_frames: int = 25):
        """
        Extract all run segments based on speed threshold.
        A run is a continuous period where speed > threshold with tolerated gaps.

        Args:
            speed_threshold: Speed threshold in cm/s
            gap_tolerance_frames: Max consecutive frames below threshold to tolerate within a run
        """
        logger.info(f"Extracting runs (speed > {speed_threshold} cm/s, gap tolerance {gap_tolerance_frames}f)...")

        all_runs_by_session = {}

        for session_name, data in self.session_data.items():
            speed = data['video_df']['speed'].to_numpy()

            above_threshold = speed > speed_threshold

            # --- edge-safe segment extraction ---
            # Pad with False so sessions that start/end inside a run get correct edges
            padded = np.concatenate([[False], above_threshold, [False]])
            diff = np.diff(padded.astype(int))
            raw_starts = np.where(diff == 1)[0]   # first True frame of each segment
            raw_ends   = np.where(diff == -1)[0] - 1  # last True frame

            # --- gap filling: merge segments separated by <= gap_tolerance_frames ---
            if len(raw_starts) == 0:
                all_runs_by_session[session_name] = []
                logger.info(f"  {session_name}: Found 0 runs")
                continue

            merged_starts = [raw_starts[0]]
            merged_ends   = [raw_ends[0]]
            for s, e in zip(raw_starts[1:], raw_ends[1:]):
                if s - merged_ends[-1] - 1 <= gap_tolerance_frames:
                    merged_ends[-1] = e          # extend current run over the gap
                else:
                    merged_starts.append(s)
                    merged_ends.append(e)

            runs = list(zip(merged_starts, merged_ends))
            all_runs_by_session[session_name] = runs
            logger.info(f"  {session_name}: Found {len(runs)} runs")

        self.all_runs_by_session = all_runs_by_session
        return all_runs_by_session
    
    def validate_run_hypothesis(self) -> Dict:
        """
        Validate that homings+escapes are exclusively within extracted runs.
        
        Returns:
            Dict with validation results per session
        """
        logger.info("Validating run hypothesis...")
        
        results = {}
        for session_name, data in self.session_data.items():
            onsets = np.where(np.diff(data['frame_labels']['is_target'].astype(int)) == 1)[0] + 1  # target start frames
            
            if len(onsets) == 0:
                logger.warning(f"  {session_name}: No manual labels")
                results[session_name] = {
                    'hypothesis_confirmed': None,
                    'pct_target_in_runs': None,
                    'num_target_events': 0,
            }
                continue
            
            runs = self.all_runs_by_session[session_name]
            
            # Create mask of all frames in runs
            n_frames = len(data['video_df'])
            in_run = np.zeros(n_frames, dtype=bool)
            for run_start, run_end in runs:
                in_run[run_start:run_end+1] = True
            
            # Check if target frames are in runs
            is_target = data['frame_labels']['is_target']
            target_frames = np.where(is_target)[0]
            target_in_runs = in_run[is_target]
            pct_in_runs = np.mean(target_in_runs) * 100 if len(target_frames) > 0 else 0
            
            hypothesis_confirmed = pct_in_runs > 95  # >95% threshold
            
            results[session_name] = {
                'hypothesis_confirmed': hypothesis_confirmed,
                'pct_target_in_runs': pct_in_runs,
                'num_target_events': len(onsets),
                'num_runs': len(runs),
            }
            
            logger.info(f"  {session_name}: {pct_in_runs:.1f}% of target in runs "
                       f"(hypothesis={'✓' if hypothesis_confirmed else '✗'})")
        
        self.hypothesis_results = results
        return results
    
    def extract_features_for_all_runs(self) -> List[RunFeatures]:
        """Extract features for each run segment."""
        logger.info("Extracting per-run features...")
        
        all_features = []
        
        for session_name, data in self.session_data.items():
            tracking_data = data['tracking_data']
            video_df = data['video_df']
            fps = data['fps']
            runs = self.all_runs_by_session[session_name]
            
            is_target = data['frame_labels']['is_target']
            
            for run_idx, (onset, offset) in enumerate(runs):
                try:
                    px_per_cm = data.get('pixel_per_cm', None)
                    if px_per_cm is None:
                        session_obj = data.get('session', None)
                        px_per_cm = getattr(getattr(session_obj, 'video', None), 'pixels_per_cm', 1.0)
                    if px_per_cm is None or px_per_cm == 0:
                        px_per_cm = 1.0
                    features = self._extract_run_features(
                        run_idx, session_name, onset, offset,
                        tracking_data, video_df, is_target, fps, px_per_cm
                    )
                    all_features.append(features)
                except Exception as e:
                    logger.warning(f"  Failed to extract features for {session_name} run {run_idx}: {e}")
        
        self.all_run_features = all_features
        logger.info(f"Extracted features for {len(all_features)} runs")
        return all_features
    
    def _extract_run_features(self, run_idx, session_name, onset, offset,
                              tracking_data, video_df, is_target, fps, pixel_per_cm) -> RunFeatures:
        """Extract features for a single run."""
        def _wrap_to_pi(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        def _circular_mean(angles):
            if len(angles) == 0:
                return np.nan
            return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

        def _wrapped_angular_velocity(angles, fps_value, step_frames=3):
            ang_vel = np.full(len(angles), np.nan, dtype=float)
            if len(angles) <= step_frames:
                return ang_vel
            wrapped_delta = _wrap_to_pi(angles[step_frames:] - angles[:-step_frames])
            ang_vel[step_frames:] = np.abs(wrapped_delta) * fps_value / step_frames
            return ang_vel
        
        run_duration = (offset - onset + 1) / fps
        run_len = offset - onset + 1
        
        # Extract trajectory during run
        x = tracking_data['avg_loc'][onset:offset+1, 0]
        y = tracking_data['avg_loc'][onset:offset+1, 1]
        speed = video_df['speed'][onset:offset+1].to_numpy()
        hdir = video_df['hdir'][onset:offset+1].to_numpy()
        
        # Speed features
        speed_mean = np.mean(speed)
        speed_peak = np.max(speed)

        initial_window_frames = max(1, int(round(fps)))
        initial_window_end = min(run_len, initial_window_frames)

        accel = np.full(run_len, np.nan, dtype=float)
        if run_len > 1:
            accel[1:] = np.diff(speed) * fps
        mean_initial_acceleration = np.nanmean(accel[1:initial_window_end]) if initial_window_end > 1 else 0.0
        if not np.isfinite(mean_initial_acceleration):
            mean_initial_acceleration = 0.0
        
        # Trajectory geometry
        net_dx = (x[-1] - x[0])/pixel_per_cm
        net_dy = (y[-1] - y[0])/pixel_per_cm 
        net_distance = np.sqrt(net_dx**2 + net_dy**2)
        
        displacement_vertical_ratio = net_dy / net_distance if net_distance > 0 else 0
        
        # Head dynamics
        head_turn_angle_initial = hdir[0]  # heading angle at start relative to shelter (from video_df)

        initial_hdir_start = min(run_len, max(0, int(round(0.5 * fps))))
        initial_hdir_end = min(run_len, max(initial_hdir_start + 1, int(round(1.5 * fps))))
        mean_hdir_initial_window = _circular_mean(hdir[initial_hdir_start:initial_hdir_end])
        if np.isfinite(mean_hdir_initial_window):
            initial_hdir_change_abs = np.abs(_wrap_to_pi(mean_hdir_initial_window - hdir[0]))
        else:
            initial_hdir_change_abs = 0.0

        head_angular_velocity = _wrapped_angular_velocity(hdir, fps_value=fps, step_frames=3)
        mean_initial_angular_head_velocity = np.nanmean(head_angular_velocity[1:initial_window_end]) if initial_window_end > 1 else 0.0
        if not np.isfinite(mean_initial_angular_head_velocity):
            mean_initial_angular_head_velocity = 0.0
        
        # Prelude: frames before run where speed < threshold
        prelude_duration = 0.0  # Can be computed if needed
        
        # Determine if target
        run_is_target = np.any(is_target[onset:offset+1])
        
        features = RunFeatures(
            run_idx=run_idx,
            session_name=session_name,
            onset_frame=onset,
            offset_frame=offset,
            is_target=run_is_target,
            speed_mean=speed_mean,
            speed_peak=speed_peak,
            mean_initial_acceleration=mean_initial_acceleration,
            head_turn_angle_initial=head_turn_angle_initial,
            initial_hdir_change_abs=initial_hdir_change_abs,
            mean_initial_angular_head_velocity=mean_initial_angular_head_velocity,
            net_dx=net_dx,
            net_dy=net_dy,
            net_distance=net_distance,
            displacement_vertical_ratio=displacement_vertical_ratio,
            prelude_duration=prelude_duration,
            run_duration=run_duration,
            start_x=x[0],
            start_y=y[0],
            end_x=x[-1],
            end_y=y[-1],
        )
        
        return features
    
    def compute_feature_distributions(self):
        """Compute distributions and rank features by discrimination power."""
        logger.info("Computing feature distributions and discrimination power...")
        
        target_features = [f for f in self.all_run_features if f.is_target]
        exploration_features = [f for f in self.all_run_features if not f.is_target]
        
        logger.info(f"  Target runs: {len(target_features)}")
        logger.info(f"  Exploration runs: {len(exploration_features)}")
        
        if len(target_features) == 0 or len(exploration_features) == 0:
            logger.warning("Insufficient runs for feature comparison")
            return {}
        
        # Feature names
        feature_names = [
            'speed_mean', 'speed_peak', 'mean_initial_acceleration',
            'head_turn_angle_initial', 'initial_hdir_change_abs', 'mean_initial_angular_head_velocity',
            'net_dx', 'net_dy', 'net_distance',
            'displacement_vertical_ratio', 'run_duration',
        ]
        
        discrimination_results = {}
        
        for fname in feature_names:
            target_vals = np.array([getattr(f, fname) for f in target_features])
            exploration_vals = np.array([getattr(f, fname) for f in exploration_features])
            
            # Remove NaNs and infs
            target_vals = target_vals[np.isfinite(target_vals)]
            exploration_vals = exploration_vals[np.isfinite(exploration_vals)]
            
            if len(target_vals) == 0 or len(exploration_vals) == 0:
                continue
            
            # Compute statistics
            target_mean, target_std = np.mean(target_vals), np.std(target_vals)
            exploration_mean, exploration_std = np.mean(exploration_vals), np.std(exploration_vals)
            
            # Cohen's d (effect size)
            cohens_d = (target_mean - exploration_mean) / np.sqrt(
                (target_std**2 + exploration_std**2) / 2
            ) if (target_std**2 + exploration_std**2) > 0 else 0
            
            # ROC-AUC (sklearn not assumed; use manual Mann-Whitney U)
            try:
                stat, pval = stats.mannwhitneyu(target_vals, exploration_vals, alternative='two-sided')
                # Approximate AUC from U statistic
                n1, n2 = len(target_vals), len(exploration_vals)
                auc = stat / (n1 * n2) if (n1 * n2) > 0 else 0.5
            except:
                auc = 0.5
                pval = 1.0
            
            discrimination_results[fname] = {
                'target_mean': float(target_mean),
                'target_std': float(target_std),
                'exploration_mean': float(exploration_mean),
                'exploration_std': float(exploration_std),
                'cohens_d': float(cohens_d),
                'auc': float(auc),
                'pval': float(pval),
                'target_vals': target_vals.tolist(),
                'exploration_vals': exploration_vals.tolist(),
            }
        
        # Rank by |Cohen's d|
        ranked = sorted(
            discrimination_results.items(),
            key=lambda x: abs(x[1]['cohens_d']),
            reverse=True
        )
        
        self.feature_ranking = ranked
        logger.info("Top 10 discriminative features:")
        for i, (fname, stats_dict) in enumerate(ranked[:10]):
            logger.info(f"  {i+1}. {fname}: Cohen's d={stats_dict['cohens_d']:.3f}, AUC={stats_dict['auc']:.3f}")
        
        return discrimination_results
    
    def plot_feature_scatter_matrix(self, output_prefix: str = 'feature', n_features: int = 6):
        """
        Plot pairwise scatter plots of top features, colored by target vs exploration.
        Creates an upper-triangular grid: each subplot shows (feature_i, feature_j) with points colored by is_target.
        
        Args:
            output_prefix: Prefix for output filename
            n_features: Number of top features to include in scatter matrix
        """
        logger.info("Plotting feature scatter matrix...")
        
        # Get top N features by rank
        top_features = [fname for fname, _ in self.feature_ranking[:n_features]]
        
        # Separate target and exploration runs
        target_features = [f for f in self.all_run_features if f.is_target]
        exploration_features = [f for f in self.all_run_features if not f.is_target]
        
        # Create n x n subplots (only upper triangle will have data)
        n = len(top_features)
        fig, axs = plt.subplots(n, n, figsize=(4*n, 4*n))
        
        for i, feat_x in enumerate(top_features):
            for j, feat_y in enumerate(top_features):
                ax = axs[i, j]
                
                if i > j:
                    # Lower triangle: hide
                    ax.axis('off')
                elif i == j:
                    # Diagonal: show distribution
                    target_vals = np.array([getattr(f, feat_x) for f in target_features])
                    exploration_vals = np.array([getattr(f, feat_x) for f in exploration_features])
                    target_vals = target_vals[np.isfinite(target_vals)]
                    exploration_vals = exploration_vals[np.isfinite(exploration_vals)]
                    
                    ax.hist(target_vals, bins=15, alpha=0.6, label='Target', color='blue', density=True)
                    ax.hist(exploration_vals, bins=15, alpha=0.6, label='Exploration', color='orange', density=True)
                    ax.set_ylabel(feat_x if i == 0 else '')
                    ax.set_title(feat_x)
                    if i == 0:
                        ax.legend(loc='upper right')
                else:
                    # Upper triangle: scatter plots
                    target_x = np.array([getattr(f, feat_x) for f in target_features])
                    target_y = np.array([getattr(f, feat_y) for f in target_features])
                    exploration_x = np.array([getattr(f, feat_x) for f in exploration_features])
                    exploration_y = np.array([getattr(f, feat_y) for f in exploration_features])
                    
                    # Remove NaNs/infs
                    target_mask = np.isfinite(target_x) & np.isfinite(target_y)
                    exploration_mask = np.isfinite(exploration_x) & np.isfinite(exploration_y)
                    target_x, target_y = target_x[target_mask], target_y[target_mask]
                    exploration_x, exploration_y = exploration_x[exploration_mask], exploration_y[exploration_mask]
                    
                    ax.scatter(target_x, target_y, alpha=0.5, s=30, color='blue', label='Target')
                    ax.scatter(exploration_x, exploration_y, alpha=0.5, s=30, color='orange', label='Exploration')
                    
                    if j == 0:
                        ax.set_ylabel(feat_y)
                    if i == 0:
                        ax.set_xlabel(feat_x)
                        ax.xaxis.tick_top()
                        ax.xaxis.set_label_position('top')
                    if i == 0 and j == n - 1:
                        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()

    def plot_feature_distributions(self, only_top_n: int = 6):
        """
        Plot distributions for top features as lines across all sessions.
        Each subplot shows overlaid histograms as lines: yellow for exploration, blue for target.
        One line per session, creating a spaghetti plot of distribution variation.
        """
        logger.info("Plotting feature distributions as lines across sessions...")
        
        # Plot top N features
        n_plots = min(only_top_n, len(self.feature_ranking))
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
        axs = axs.flatten()
        
        for idx, (fname, stats_dict) in enumerate(self.feature_ranking[:n_plots]):
            ax = axs[idx]
            
            # Plot lines for each session's target and exploration distributions
            for session_name in self.session_data.keys():
                session_features = [f for f in self.all_run_features if f.session_name == session_name]
                
                # Target runs for this session
                target_runs = [f for f in session_features if f.is_target]
                if len(target_runs) > 0:
                    target_vals = np.array([getattr(f, fname) for f in target_runs])
                    target_vals = target_vals[np.isfinite(target_vals)]
                    if len(target_vals) > 1:
                        counts, bins = np.histogram(target_vals, bins=15)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        ax.plot(bin_centers, counts/len(target_vals), color='blue', alpha=0.5, linewidth=1.5)
                
                # Exploration runs for this session
                exploration_runs = [f for f in session_features if not f.is_target]
                if len(exploration_runs) > 0:
                    exploration_vals = np.array([getattr(f, fname) for f in exploration_runs])
                    exploration_vals = exploration_vals[np.isfinite(exploration_vals)]
                    if len(exploration_vals) > 1:
                        counts, bins = np.histogram(exploration_vals, bins=15)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        ax.plot(bin_centers, counts/len(exploration_vals), color='orange', alpha=0.5, linewidth=1.5)
            
            ax.set_title(f"{fname}\n(Cohen's d={stats_dict['cohens_d']:.2f})")
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            if fname == "net_distance": ax.set_xlim(0, 90)
            if fname == "net_dy": ax.set_xlim(-75,75)
            if fname == "speed_mean": ax.set_xlim(0,60)
            if fname == "speed_peak": ax.set_xlim(0,150)
            if fname == "mean_initial_acceleration": ax.set_xlim(0,50)
            if fname == "displacement_vertical_ratio": ax.set_xlim(-1,1)
            if fname == "run_duration": ax.set_xlim(0,50)
            if fname == "net_dx": ax.set_xlim(-75,75)
            if fname == "head_turn_angle_initial": ax.set_xlim(-np.pi, np.pi)
            if fname == "initial_hdir_change_abs": ax.set_xlim(0, np.pi)
            if fname == "mean_initial_angular_head_velocity": ax.set_xlim(0, 5)
        
        # Add legend (shared across all subplots)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2.5, label='Target (n sessions)'),
            Line2D([0], [0], color='orange', lw=2.5, label='Exploration (n sessions)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
    
    def generate_report(self):
        """Generate Phase 1 report."""
        logger.info("Generating Phase 1 report...")
        
        report_path = self.output_dir / 'phase1_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 1: HOMING DETECTION ANALYSIS - RUN HYPOTHESIS & FEATURE EXTRACTION\n")
            f.write("=" * 80 + "\n\n")
            
            # Hypothesis results
            f.write("RUN HYPOTHESIS VALIDATION\n")
            f.write("-" * 80 + "\n")
            all_confirmed = all(
                v['hypothesis_confirmed'] for v in self.hypothesis_results.values()
                if v['hypothesis_confirmed'] is not None
            )
            f.write(f"Hypothesis Status: {'✓ CONFIRMED' if all_confirmed else '✗ FAILED'}\n")
            f.write(f"  Requirement: >95% of target frames within extracted runs\n\n")
            
            for session_name, result in self.hypothesis_results.items():
                f.write(f"  {session_name}:\n")
                f.write(f"    - Target in runs: {result['pct_target_in_runs']:.1f}%\n")
                f.write(f"    - Target events: {result['num_target_events']}\n")
                f.write(f"    - Runs extracted: {result['num_runs']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FEATURE DISCRIMINATION RANKING (Top 15)\n")
            f.write("-" * 80 + "\n")
            f.write("{:<6} {:<35} {:<12} {:<8} {:<12}\n".format(
                            "Rank", "Feature", "Cohen's d", "AUC", "p-value"
                        ))
            f.write("-" * 80 + "\n")
            
            for idx, (fname, stats_dict) in enumerate(self.feature_ranking[:15]):
                f.write(f"{idx+1:<6} {fname:<35} {stats_dict['cohens_d']:>11.3f} "
                       f"{stats_dict['auc']:>7.3f} {stats_dict['pval']:>11.3e}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS FOR PHASE 2\n")
            f.write("-" * 80 + "\n")
            f.write("Use top 3-5 features with highest |Cohen's d| as gate functions.\n")
            f.write("Suggested candidates:\n")
            for idx, (fname, _) in enumerate(self.feature_ranking[:5]):
                f.write(f"  {idx+1}. {fname}\n")
        
        logger.info(f"Report saved to {report_path}")
        return report_path

    def _get_target_events_from_labels(self, is_target: np.ndarray) -> List[Tuple[int, int]]:
        """Convert frame-level target labels into contiguous event ranges."""
        padded = np.concatenate([[False], is_target.astype(bool), [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    def _interval_overlap_stats(self, a0: int, a1: int, b0: int, b1: int) -> Dict[str, float]:
        """Compute overlap fractions and IoU for two frame intervals."""
        inter = max(0, min(a1, b1) - max(a0, b0) + 1)
        len_a = a1 - a0 + 1
        len_b = b1 - b0 + 1
        union = len_a + len_b - inter
        return {
            'intersection_frames': float(inter),
            'candidate_fraction': float(inter / len_a) if len_a > 0 else 0.0,
            'manual_fraction': float(inter / len_b) if len_b > 0 else 0.0,
            'iou': float(inter / union) if union > 0 else 0.0,
        }

    def _fit_single_feature_gate(self, feature_name: str, target_recall: float = 0.90) -> Dict:
        """Fit a single-feature threshold with direction by maximizing precision under recall constraint."""
        target_runs = [f for f in self.all_run_features if f.is_target]
        explore_runs = [f for f in self.all_run_features if not f.is_target]

        tvals = np.array([getattr(f, feature_name) for f in target_runs], dtype=float)
        evals = np.array([getattr(f, feature_name) for f in explore_runs], dtype=float)
        tvals = tvals[np.isfinite(tvals)]
        evals = evals[np.isfinite(evals)]

        if len(tvals) == 0 or len(evals) == 0:
            return {
                'feature': feature_name,
                'direction': '>=',
                'threshold': 0.0,
                'recall': 0.0,
                'precision': 0.0,
                'f1': 0.0,
            }

        direction = '>=' if np.mean(tvals) >= np.mean(evals) else '<='
        pool = np.unique(np.concatenate([tvals, evals]))
        if len(pool) > 200:
            q = np.linspace(0.01, 0.99, 200)
            pool = np.quantile(pool, q)

        best = None
        for thr in pool:
            if direction == '>=':
                tp = np.sum(tvals >= thr)
                fp = np.sum(evals >= thr)
            else:
                tp = np.sum(tvals <= thr)
                fp = np.sum(evals <= thr)

            fn = len(tvals) - tp
            recall = tp / len(tvals) if len(tvals) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            score = (recall >= target_recall, precision, f1, recall)
            if best is None or score > best['score']:
                best = {
                    'feature': feature_name,
                    'direction': direction,
                    'threshold': float(thr),
                    'recall': float(recall),
                    'precision': float(precision),
                    'f1': float(f1),
                    'score': score,
                }

        best.pop('score', None)
        return best

    def run_phase2_classification(
        self,
        top_n_features: int = 4,
        exclude_features: List[str] = None,
        target_recall: float = 0.90,
        max_run_duration_s: float = None,
        auto_duration_cap_quantile: float = 0.95,
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Phase 2: classify runs using top ranked features and learned one-feature gates."""
        if not hasattr(self, 'feature_ranking') or len(self.feature_ranking) == 0:
            raise RuntimeError("Run compute_feature_distributions() before Phase 2 classification.")

        if not hasattr(self, 'all_run_features') or len(self.all_run_features) == 0:
            raise RuntimeError("Run extract_features_for_all_runs() before Phase 2 classification.")

        exclude = set(exclude_features or [])
        ranked = [fname for fname, _ in self.feature_ranking if fname not in exclude]
        selected = ranked[:top_n_features]
        if len(selected) == 0:
            raise RuntimeError("No available features for Phase 2 classification after exclusions.")

        gates = {fname: self._fit_single_feature_gate(fname, target_recall=target_recall) for fname in selected}

        duration_cap = max_run_duration_s
        if duration_cap is None:
            target_durations = np.array([f.run_duration for f in self.all_run_features if f.is_target], dtype=float)
            target_durations = target_durations[np.isfinite(target_durations)]
            if len(target_durations) > 0:
                duration_cap = float(np.quantile(target_durations, auto_duration_cap_quantile))

        candidates = {s: [] for s in self.session_data.keys()}
        candidate_meta = {s: [] for s in self.session_data.keys()}

        for feat in self.all_run_features:
            passed = True
            run_gate_values = {}
            for fname in selected:
                gate = gates[fname]
                value = float(getattr(feat, fname))
                run_gate_values[fname] = value
                if gate['direction'] == '>=':
                    ok = value >= gate['threshold']
                else:
                    ok = value <= gate['threshold']
                if not ok:
                    passed = False
                    break

            if passed and duration_cap is not None and float(feat.run_duration) > duration_cap:
                passed = False

            if passed:
                candidates[feat.session_name].append((feat.onset_frame, feat.offset_frame))
                candidate_meta[feat.session_name].append({
                    'run_idx': feat.run_idx,
                    'onset': feat.onset_frame,
                    'offset': feat.offset_frame,
                    'feature_values': run_gate_values,
                })

        self.phase2_selected_features = selected
        self.phase2_gates = gates
        self.phase2_duration_cap_s = duration_cap
        self.phase2_candidates_by_session = candidates
        self.phase2_candidate_meta = candidate_meta

        logger.info(f"Phase 2 selected features: {selected}")
        for fname in selected:
            g = gates[fname]
            logger.info(
                f"  gate {fname}: keep if value {g['direction']} {g['threshold']:.4f} "
                f"(feature recall={g['recall']:.3f}, precision={g['precision']:.3f})"
            )
        if duration_cap is not None:
            logger.info(f"  run_duration cap: keep if run_duration <= {duration_cap:.3f} s")

        return candidates

    def compute_phase3_fractional_overlap(
        self,
        candidates_by_session: Dict[str, List[Tuple[int, int]]] = None,
        manual_fraction_threshold: float = 0.50,
        candidate_fraction_threshold: float = 0.50,
    ) -> Dict:
        """Phase 3: compute overlap between candidate runs and manual target events."""
        if candidates_by_session is None:
            candidates_by_session = getattr(self, 'phase2_candidates_by_session', None)
        if candidates_by_session is None:
            raise RuntimeError("Run run_phase2_classification() first or provide candidates_by_session.")

        summary = {}
        all_rows = []

        for session_name, data in self.session_data.items():
            manual_events = self._get_target_events_from_labels(data['frame_labels']['is_target'])
            candidates = candidates_by_session.get(session_name, [])

            if len(candidates) == 0:
                summary[session_name] = {
                    'num_manual': len(manual_events),
                    'num_candidates': 0,
                    'manual_recall_at_threshold': 0.0,
                    'candidate_precision_at_threshold': 0.0,
                }
                continue

            candidate_best_manual_frac = []
            candidate_best_candidate_frac = []
            candidate_best_iou = []
            for ci, (c0, c1) in enumerate(candidates):
                best = {'manual_fraction': 0.0, 'candidate_fraction': 0.0, 'iou': 0.0, 'manual_idx': -1}
                for mi, (m0, m1) in enumerate(manual_events):
                    ov = self._interval_overlap_stats(c0, c1, m0, m1)
                    if ov['iou'] > best['iou']:
                        best = {
                            'manual_fraction': ov['manual_fraction'],
                            'candidate_fraction': ov['candidate_fraction'],
                            'iou': ov['iou'],
                            'manual_idx': mi,
                        }
                candidate_best_manual_frac.append(best['manual_fraction'])
                candidate_best_candidate_frac.append(best['candidate_fraction'])
                candidate_best_iou.append(best['iou'])
                all_rows.append({
                    'session_name': session_name,
                    'candidate_idx': ci,
                    'onset': c0,
                    'offset': c1,
                    'best_manual_idx': best['manual_idx'],
                    'best_manual_fraction': best['manual_fraction'],
                    'best_candidate_fraction': best['candidate_fraction'],
                    'best_iou': best['iou'],
                })

            manual_best_candidate_frac = []
            for m0, m1 in manual_events:
                best = 0.0
                for c0, c1 in candidates:
                    ov = self._interval_overlap_stats(c0, c1, m0, m1)
                    best = max(best, ov['manual_fraction'])
                manual_best_candidate_frac.append(best)

            manual_recall = float(np.mean(np.array(manual_best_candidate_frac) >= manual_fraction_threshold)) if len(manual_best_candidate_frac) > 0 else 0.0
            candidate_precision = float(np.mean(np.array(candidate_best_candidate_frac) >= candidate_fraction_threshold)) if len(candidate_best_candidate_frac) > 0 else 0.0
            strict_pair_precision = float(np.mean((np.array(candidate_best_manual_frac) >= manual_fraction_threshold) & (np.array(candidate_best_candidate_frac) >= candidate_fraction_threshold))) if len(candidate_best_manual_frac) > 0 else 0.0

            summary[session_name] = {
                'num_manual': len(manual_events),
                'num_candidates': len(candidates),
                'manual_recall_at_threshold': manual_recall,
                'candidate_precision_at_threshold': candidate_precision,
                'strict_pair_precision': strict_pair_precision,
                'mean_best_manual_fraction_per_candidate': float(np.mean(candidate_best_manual_frac)) if len(candidate_best_manual_frac) > 0 else 0.0,
                'mean_best_candidate_fraction_per_candidate': float(np.mean(candidate_best_candidate_frac)) if len(candidate_best_candidate_frac) > 0 else 0.0,
                'mean_best_iou_per_candidate': float(np.mean(candidate_best_iou)) if len(candidate_best_iou) > 0 else 0.0,
            }

        self.phase3_overlap_summary = summary
        self.phase3_overlap_rows = all_rows
        return {'summary': summary, 'rows': all_rows}

    def plot_phase3_event_view(
        self,
        session_name: str,
        candidate_idx: int = 0,
        pad_s: float = 1.0,
    ):
        """Phase 3 visualizer for one candidate event against the best-matching manual event."""
        if not hasattr(self, 'phase2_candidates_by_session'):
            raise RuntimeError("Run run_phase2_classification() before using the Phase 3 visualizer.")

        if session_name not in self.session_data:
            raise KeyError(f"Unknown session: {session_name}")

        candidates = self.phase2_candidates_by_session.get(session_name, [])
        if len(candidates) == 0:
            raise RuntimeError(f"No Phase 2 candidate runs for session {session_name}")

        candidate_idx = int(np.clip(candidate_idx, 0, len(candidates) - 1))
        c0, c1 = candidates[candidate_idx]

        data = self.session_data[session_name]
        fps = float(data['fps'])
        tracking = data['tracking_data']
        video_df = data['video_df']
        is_target = data['frame_labels']['is_target']
        manual_events = self._get_target_events_from_labels(is_target)

        best_mi = -1
        best = {'manual_fraction': 0.0, 'candidate_fraction': 0.0, 'iou': 0.0}
        for mi, (m0, m1) in enumerate(manual_events):
            ov = self._interval_overlap_stats(c0, c1, m0, m1)
            if ov['iou'] > best['iou']:
                best_mi = mi
                best = ov

        n_frames = len(video_df)
        pad = int(round(pad_s * fps))
        start = max(0, c0 - pad)
        stop = min(n_frames, c1 + pad + 1)
        frames = np.arange(start, stop)
        t = (frames - c0) / fps

        x = tracking['avg_loc'][start:stop, 0]
        y = tracking['avg_loc'][start:stop, 1]
        speed = video_df['speed'][start:stop].to_numpy()
        hdir = video_df['hdir'][start:stop].to_numpy()

        cand_mask = np.zeros(stop - start, dtype=bool)
        cand_mask[max(0, c0 - start):min(stop - start, c1 - start + 1)] = True

        man_mask = np.zeros(stop - start, dtype=bool)
        manual_label = "none"
        if best_mi >= 0:
            m0, m1 = manual_events[best_mi]
            man_mask[max(0, m0 - start):min(stop - start, m1 - start + 1)] = True
            manual_label = f"manual #{best_mi} ({m0}-{m1})"

        fig, axs = plt.subplots(2, 2, figsize=(11, 8), gridspec_kw={'height_ratios': [2, 1]})

        axs[0, 0].plot(x, y, color='0.75', lw=1)
        axs[0, 0].scatter(x[cand_mask], y[cand_mask], s=9, color='tab:blue', alpha=0.85, label='candidate')
        axs[0, 0].scatter(x[man_mask], y[man_mask], s=9, color='tab:orange', alpha=0.75, label='manual')
        axs[0, 0].set_title(f"{session_name} candidate #{candidate_idx} trajectory")
        axs[0, 0].set_aspect('equal')
        axs[0, 0].invert_yaxis()
        axs[0, 0].legend(loc='best', fontsize=8)

        axs[0, 1].plot(t, speed, color='black', lw=1)
        axs[0, 1].fill_between(t, 0, speed, where=cand_mask, color='tab:blue', alpha=0.25)
        axs[0, 1].fill_between(t, 0, speed, where=man_mask, color='tab:orange', alpha=0.20)
        axs[0, 1].axvline(0.0, color='k', ls='--', lw=1)
        axs[0, 1].set_ylabel('speed')
        axs[0, 1].set_title('Speed with candidate/manual masks')

        axs[1, 0].plot(t, hdir, color='slategray', lw=1)
        axs[1, 0].fill_between(t, np.nanmin(hdir), np.nanmax(hdir), where=cand_mask, color='tab:blue', alpha=0.15)
        axs[1, 0].fill_between(t, np.nanmin(hdir), np.nanmax(hdir), where=man_mask, color='tab:orange', alpha=0.12)
        axs[1, 0].axvline(0.0, color='k', ls='--', lw=1)
        axs[1, 0].set_xlabel('time from candidate onset (s)')
        axs[1, 0].set_ylabel('hdir')

        txt = (
            f"best overlap with {manual_label}\n"
            f"manual_fraction={best['manual_fraction']:.3f}\n"
            f"candidate_fraction={best['candidate_fraction']:.3f}\n"
            f"IoU={best['iou']:.3f}"
        )
        axs[1, 1].axis('off')
        axs[1, 1].text(0.0, 0.95, txt, va='top', ha='left', fontsize=10)

        plt.tight_layout()
        return fig

    def plot_run_aligned_dynamics_per_session(
        self,
        features_to_plot=None,
        pre_s: float = 1.0,
        post_s: float = 2.0,
        target_overlap_threshold: float = 0.0,
        with_std: bool = True,
        output_prefix: str = "run_aligned_grid",
    ):
        """
        One figure per session.
        Subplots = requested features.
        Each subplot shows:
        - target mean line
        - exploration mean line
        - optional std shading
        Signals are onset-aligned.
        """
        logger.info("Plotting onset-aligned dynamics (feature subplots in one figure per session)...")

        def _wrap_to_pi(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        def _wrapped_angular_velocity(angles, fps_value, step_frames=3):
            ang_vel = np.full(len(angles), np.nan, dtype=float)
            if len(angles) <= step_frames:
                return ang_vel
            wrapped_delta = _wrap_to_pi(angles[step_frames:] - angles[:-step_frames])
            ang_vel[step_frames:] = np.abs(wrapped_delta) * fps_value / step_frames
            return ang_vel

        if features_to_plot is None:
            features_to_plot = ["speed", "delta_hdir", "hdir", "ang_vel", "accel"]

        label_map = {
            "speed": "speed",
            "delta_hdir": "delta hdir (rad/frame)",
            "hdir": "head direction (rad)",
            "ang_vel": "angular head velocity (rad/s)",
            "accel": "acceleration (speed/s)",
        }

        color_target = "tab:blue"
        color_explore = "orange"

        for session_name, data in self.session_data.items():
            video_df = data["video_df"]
            fps = float(data["fps"])
            runs = self.all_runs_by_session.get(session_name, [])
            is_target_frame = data["frame_labels"]["is_target"]

            if len(runs) == 0:
                logger.warning(f"{session_name}: no runs, skipping")
                continue

            pre_f = int(round(pre_s * fps))
            post_f = int(round(post_s * fps))
            win_len = pre_f + post_f + 1
            t = (np.arange(win_len) - pre_f) / fps

            speed = video_df["speed"].to_numpy()
            hdir = video_df["hdir"].to_numpy()

            delta_hdir = np.zeros_like(hdir, dtype=float)
            if len(hdir) > 1:
                delta_hdir[1:] = _wrap_to_pi(np.diff(hdir))

            ang_vel = _wrapped_angular_velocity(hdir, fps_value=fps, step_frames=3)

            accel = np.zeros_like(speed, dtype=float)
            if len(speed) > 1:
                accel[1:] = np.diff(speed) * fps

            signal_bank = {
                "speed": speed,
                "delta_hdir": delta_hdir,
                "hdir": hdir,
                "ang_vel": ang_vel,
                "accel": accel,
            }

            valid_features = [f for f in features_to_plot if f in signal_bank]
            unknown = [f for f in features_to_plot if f not in signal_bank]
            if unknown:
                logger.warning(f"{session_name}: unknown features skipped: {unknown}")
            if len(valid_features) == 0:
                logger.warning(f"{session_name}: no valid features to plot")
                continue

            target_stacks = {f: [] for f in valid_features}
            explore_stacks = {f: [] for f in valid_features}
            n_frames = len(speed)

            for onset, offset in runs:
                run_mask = is_target_frame[onset:offset + 1]
                overlap_frac = float(np.mean(run_mask)) if len(run_mask) > 0 else 0.0
                run_is_target = overlap_frac > target_overlap_threshold

                w0 = onset - pre_f
                w1 = onset + post_f

                for f in valid_features:
                    arr = signal_bank[f]
                    w = np.full(win_len, np.nan, dtype=float)

                    src0 = max(0, w0)
                    src1 = min(n_frames - 1, w1)
                    if src1 >= src0:
                        dst0 = src0 - w0
                        dst1 = dst0 + (src1 - src0) + 1
                        w[dst0:dst1] = arr[src0:src1 + 1]
                        if f == "hdir":
                            w[dst0:dst1] = np.abs(_wrap_to_pi(w[dst0:dst1] - arr[onset]))

                    if run_is_target:
                        target_stacks[f].append(w)
                    else:
                        explore_stacks[f].append(w)

            n_feat = len(valid_features)
            ncols = min(2, n_feat)
            nrows = int(np.ceil(n_feat / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows), squeeze=False)
            axs = axs.flatten()

            n_target_runs = len(target_stacks[valid_features[0]])
            n_explore_runs = len(explore_stacks[valid_features[0]])

            for i, f in enumerate(valid_features):
                ax = axs[i]

                if len(target_stacks[f]) > 0:
                    T = np.vstack(target_stacks[f])
                    mu_t = np.nanmean(T, axis=0)
                    sd_t = np.nanstd(T, axis=0)
                    ax.plot(t, mu_t, color=color_target, lw=2, label=f"target (n={n_target_runs})")
                    if with_std:
                        ax.fill_between(t, mu_t - sd_t, mu_t + sd_t, color=color_target, alpha=0.2)

                if len(explore_stacks[f]) > 0:
                    E = np.vstack(explore_stacks[f])
                    mu_e = np.nanmean(E, axis=0)
                    sd_e = np.nanstd(E, axis=0)
                    ax.plot(t, mu_e, color=color_explore, lw=2, label=f"exploration (n={n_explore_runs})")
                    if with_std:
                        ax.fill_between(t, mu_e - sd_e, mu_e + sd_e, color=color_explore, alpha=0.2)

                ax.axvline(0.0, color="k", ls="--", lw=1)
                ax.set_title(label_map.get(f, f))
                ax.set_xlabel("time from run onset (s)")
                ax.set_ylabel("value")
                ax.grid(alpha=0.25)
                ax.legend(loc="best", fontsize=8)

            # hide any extra empty axes
            for j in range(n_feat, len(axs)):
                axs[j].axis("off")

            fig.suptitle(f"{session_name} | onset-aligned target vs exploration", y=1.01)
            plt.tight_layout()
        
# Main entry point
if __name__ == "__main__":
    # This will be called from a notebook orchestrating all phases
    pass