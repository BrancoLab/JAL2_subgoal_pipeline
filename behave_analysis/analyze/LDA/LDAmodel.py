# OS libaries
from loguru import logger
import numpy as np
import polars as pl
import pickle
import matplotlib

matplotlib.use("Agg")

from tqdm import tqdm

# import functions
from behave_analysis.analyze.stats.linshit import LinearShift
from behave_analysis.analyze.LDA.LDA_plotting import (
    across_conditions_LDA_map,
    plot_LDA_model,
    plot_LDA_by_position,
)
from behave_analysis.analyze.LDA.LDA_utils import (
    BuildSavingFolder,
    check_if_we_do_LDA,
    fill_dict_with_zeros,
    choose_predictors,
    list_conditions,
)
from behave_analysis.analyze.LDA.LDA_preprocess import (
    select_relevant_frames,
    BinDfbyPos,
    BinArenaEqualParts,
    exclude_proximal_frames,
    prep_target_and_predictors,
)
from behave_analysis.utils.PersistentPool import PersistentPool
from behave_analysis.analyze.LDA.LDA_by_position.LDA_by_position import run_LDA_model_by_position
from behave_analysis.analyze.LDA.LDA_fitting import linear_discriminant_analysis, parallel_function


def LDA(self, settings):
    """
    A wrapper function that figures out all the conditions across which to run decoding analysis.
    It will iterate over all conditions and compartments.
    If the decoding analyss has not yet been run or the user asked to force_redo, the analysis will be run,
    if not it will jump straight to plotting the prediction accuracy maps for all conditions
    """

    # determine condition types
    self.number_of_homings, condition_types = list_conditions(settings)

    # if running dropout or linear shift - initialize the pool
    if np.logical_or(settings.dropout, settings.linear_shift):
        self.PPool = PersistentPool()

    # run LDA across condition types, across compartments and across conditions
    for cond in condition_types:  # e.g. 'experimental_conditions', 'behavioral_conditions'
        self.condition_types = cond
        for comp in settings.compartment_split:  # ['all','threat_zone','shelter_compartment', 'left_arena','right_arena','by_position']
            self.compartment = comp
            if comp == 'by_position':
                # figure out which angles we want to decode
                if np.logical_or(settings.run_LDA == 'all_vectors', settings.run_LDA == 'all_distance'):
                    logger.warning("You are running LDA by position to decode vectors or distances, but this dramatically reduces the amount of available data. Run it on 'all_angles' instead")
                self.number_of_bins = 9
                self.num_slices = 6
                self.num_circles = 3
                target = choose_predictors(settings, self.session, include_rand_points = False)
            else:
                self.number_of_bins = settings.number_of_bins
                target = choose_predictors(settings, self.session)
            for c in self.all_conditions:
                # e.g. 'all_time', 'pre_shelter', 'shelter_present', 'barrier_present', 'shelter_only', 'barrier_pre_flip', 'barrier_post_flip'
                self.condition = c
                self.savepath = BuildSavingFolder(self.dir, settings, self.cluster_type, self.condition_types, self.condition, self.compartment)
                check_if_we_do_LDA(self, settings)
                if np.logical_or(self.do_LDA, self.do_LS):
                    logger.info(
                        f"Run LDA on {self.cluster_type} data with condition {self.condition} in condition type {self.condition_types} in compartment {self.compartment}"
                    )
                    if comp == 'by_position':
                        run_LDA_model_by_position(self, settings, target)    
                    else:
                        run_LDA_model(self, settings, target)
                else:
                    logger.info(
                        f"LDA already run on this session for condition {self.condition} in condition type {self.condition_types} in compartment {self.compartment}"
                    )
                logger.info(f"Time for some overview plots")
                if comp == 'by_position':
                    plot_LDA_by_position(self, settings, target)
                else:
                    plot_LDA_model(self, settings)
        if comp != 'by_position':
            across_conditions_LDA_map(self, settings)
    
    if np.logical_or(settings.dropout, settings.linear_shift):
        self.PPool.close()

def run_LDA_model(self, settings, target_name):
    """
    A function that iterates across all angles and runs decoder analysis and linear shift statistics based on user settings
    It will also make bar plots of prediction accuracy across all angles and a map of prediction accuracy for the random points
    """

    prediction_accuracy = {}
    prediction_coef = {}
    LS_compiled = {}
    dropout_pa = {}
    LDA_y_output = {}

    # filter video_df for this condition
    filtered_video_df = select_relevant_frames(self)
    if settings.subsampling:
        bp, _ = BinArenaEqualParts(filtered_video_df, numpoints = 4, numrings = 1, radius = 460, video = self.session.video)
    else:
        bp = np.ones(len(filtered_video_df))
    filtered_video_df = filtered_video_df.hstack([pl.Series("binned_position", bp)])
    # remove all frames where binned position is zero as these are outside the arena!
    filtered_video_df = filtered_video_df.filter((filtered_video_df['binned_position'] > 0))

    for variable in target_name:

        if 'randP' not in variable:
            logger.info(f"Running LDA on {variable}")
            
            # we can run LDA only for times when the mouse is far from the point we're trying to decode the angle to
            if np.logical_and(settings.exclude_proximal > 0, variable != "hdir"):
                logger.warning(
                    "You are excluding proximal frames! This reduces the amount of data available - recommend only doing this for experimental conditions"
                )
                self.filtered_video_df = exclude_proximal_frames(
                    filtered_video_df, variable, self.tracking_data, dist_thresh=settings.exclude_proximal * self.session.video.pixels_per_cm
                )
            else:
                self.filtered_video_df = filtered_video_df

            # if no frames meet the criteria, make this condition blank
            if len(self.filtered_video_df) == 0:
                prediction_coef, prediction_accuracy, LDA_y_output, LS_compiled, dropout_pa = fill_dict_with_zeros(
                    self, prediction_coef, prediction_accuracy, LDA_y_output, dropout_pa, LS_compiled, variable
                )
            else:
                savename, target, X = prep_target_and_predictors(self, variable, settings)

                # run LDA on different angles
                if self.do_LDA:
                    pa, coef, frames, y_out = linear_discriminant_analysis(
                        X,
                        pos_ang=target,
                        epoch_num=settings.epoch_num,
                        fr=self.session.video.fps,
                        return_coef=True,
                        discriminant_type=settings.discriminant_type,
                        plotting=True,
                        self=self,
                        title=savename,
                        subsampling = settings.subsampling,
                    )
                    prediction_accuracy.update({variable: pa})
                    prediction_accuracy.update({variable + '_time': frames})
                    prediction_coef.update({variable: coef})
                    LDA_y_output.update({variable: y_out})

                # run LDA with individual cell dropout
                if self.do_dropout:
                    logger.info(f"Running LDA predictor dropout on {variable}")
                    X_drop = []
                    for drop in np.arange(1, np.shape(X)[1]):
                        X_drop.append(np.delete(X, drop, axis=1))
                    args_list = [(x, target) for x in X_drop]
                    dropouts = self.PPool.mp_pool.map(parallel_function, args_list)
                    dropout_pa.update({variable: dropouts})

                # run linear shift on different angles
                if self.do_LS:
                    logger.info(f"Running linear shift on LDA on {variable}")
                    LS_output = LinearShift(
                        X,
                        y=target,
                        PPool=self.PPool,
                        stat_computation_func=linear_discriminant_analysis,
                        step=40,
                        size_of_central_chunk=np.round(np.shape(X)[0] * 0.9),
                    )
                    LS_compiled.update({variable: LS_output})
                    del LS_output

        else:  # if the variable is a random point
            n_randP = self.video_df.select(pl.col("^head_randP_.*$")).width
            for j in tqdm(np.arange(self.video_df.select(pl.col("^head_randP_.*$")).width), desc=f"Running LDA on random point out of  {n_randP}"):
                # we can run LDA only for times when the mouse is far from the point we're trying to deocde the angle to
                if settings.exclude_proximal > 0:
                    self.filtered_video_df = exclude_proximal_frames(
                        filtered_video_df,
                        variable + str(j),
                        self.tracking_data,
                        dist_thresh=settings.exclude_proximal * self.session.video.pixels_per_cm,
                    )
                else:
                    self.filtered_video_df = filtered_video_df

                # if no frames meet the criteria, make this condition blank
                if len(self.filtered_video_df) == 0:
                    prediction_coef, prediction_accuracy, LS_compiled = fill_dict_with_zeros(
                        self, prediction_coef, prediction_accuracy, LS_compiled, variable + str(j)
                    )

                else:
                    savename, target, X = prep_target_and_predictors(self, str(variable + str(j)), settings)

                    # run LDA on different angles
                    if self.do_LDA:
                        pa, coef, frames, y_out = linear_discriminant_analysis(
                            X,
                            pos_ang=target,
                            epoch_num=settings.epoch_num,
                            fr=self.session.video.fps,
                            return_coef=True,
                            discriminant_type=settings.discriminant_type,
                            plotting=False, # TODO: needs to be false!
                            self=self,
                            title=savename,
                            subsampling = settings.subsampling,
                        )
                        prediction_accuracy.update({str(variable + str(j)): pa})
                        prediction_accuracy.update({'time_rP'+str(j): frames})
                        prediction_coef.update({str(variable + str(j)): coef})
                        LDA_y_output.update({str(variable + str(j)): y_out})

                    # run linear shift on different angles
                    if self.do_LS:
                        LS_output = LinearShift(
                            X,
                            y=target,
                            PPool=self.PPool,
                            stat_computation_func=linear_discriminant_analysis,
                            step=40,
                            size_of_central_chunk=np.round(np.shape(X)[0] * 0.9),
                        )
                        LS_compiled.update({str(variable + str(j)): LS_output})
                        del LS_output

    logger.info(f"Finally! It's time to save LDA output on {self.condition}")
    if self.do_LDA:
        with open(self.LDA_out, "wb") as fp:
            pickle.dump(prediction_accuracy, fp)
        coef_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_prediction_coef" + ".pkl"
        with open(coef_out, "wb") as fp:
            pickle.dump(prediction_coef, fp)
        y_path = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_y_out" + ".pkl"
        with open(y_path, "wb") as fp:
            pickle.dump(LDA_y_output, fp)

    if self.do_LS:
        with open(self.LS_out, "wb") as fp:
            pickle.dump(LS_compiled, fp)

    if self.do_dropout:
        with open(self.dropout_out, "wb") as fp:
            pickle.dump(dropout_pa, fp)
