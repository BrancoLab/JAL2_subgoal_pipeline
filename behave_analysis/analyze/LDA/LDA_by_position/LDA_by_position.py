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

from behave_analysis.analyze.LDA.LDA_utils import (
    fill_dict_with_zeros,
)
from behave_analysis.analyze.LDA.LDA_preprocess import (
    select_relevant_frames,
    BinDfbyPos,
    exclude_proximal_frames,
    prep_target_and_predictors,
)
from behave_analysis.analyze.LDA.LDA_fitting import linear_discriminant_analysis

def run_LDA_model_by_position(self, settings, target_name):
    """
    A function that iterates across all angles and runs decoder analysis and linear shift statistics based on user settings
    It will also make bar plots of prediction accuracy across all angles and a map of prediction accuracy for the random points
    """

    prediction_accuracy = {}
    prediction_coef = {}
    LS_compiled = {}
    dropout_pa = {}

    # filter video_df for this condition
    filtered_video_df = select_relevant_frames(self)
    bp, bc = BinDfbyPos(filtered_video_df, self.session.video.height, self.session.video.width, numpoints = self.pos_numpoints, return_bin_centre = True)
    filtered_video_df = filtered_video_df.hstack([pl.Series("binned_position", bp)])
    prediction_accuracy.update({'bin_centre' : bc})

    for variable in target_name:

        logger.info(f"Running LDA on {variable}")
        import time
        start_time = time.time()
        
        # we can run LDA only for times when the mouse is far from the point we're trying to decode the angle to
        if np.logical_and(settings.exclude_proximal > 0, variable != "hdir"):
            logger.warning(
                "You are excluding proximal frames! This reduces the amount of data available - recommend only doing this for experimental conditions"
            )
            self.filtered_video_df_full = exclude_proximal_frames(
                filtered_video_df, variable, self.tracking_data, dist_thresh=settings.exclude_proximal * self.session.video.pixels_per_cm
            )
        else:
            self.filtered_video_df_full = filtered_video_df

        # now iterate over the positions!
        for b in tqdm(np.unique(self.filtered_video_df_full['binned_position'].to_numpy()), desc=f"Running LDA on position out of {len(np.unique(self.filtered_video_df_full['binned_position'].to_numpy()))}"):
            if b == 0:
                continue
            self.filtered_video_df = self.filtered_video_df_full.filter(self.filtered_video_df_full['binned_position'] == b)
            # if no frames meet the criteria, make this condition blank
            if len(self.filtered_video_df) == 0:
                prediction_coef, prediction_accuracy, LS_compiled, dropout_pa = fill_dict_with_zeros(
                    self, prediction_coef, prediction_accuracy, dropout_pa, LS_compiled, variable + '_pos' + str(b)
                )
            else:
                savename, target, X = prep_target_and_predictors(self, variable, settings)
                savename = savename + '_pos' + str(b)
                preprocess_time = time.time()
                print("Time to preprocess is " + str(preprocess_time - start_time))

                # run LDA on different angles
                if self.do_LDA:
                    start_time = time.time()
                    pa, coef = linear_discriminant_analysis(
                        X,
                        pos_ang=target,
                        epoch_num=settings.epoch_num,
                        fr=self.session.video.fps,
                        return_coef=True,
                        discriminant_type=settings.discriminant_type,
                        plotting=False,
                        self=self,
                        title=savename,
                    )
                    prediction_accuracy.update({variable + '_pos' + str(b): pa})
                    prediction_coef.update({variable + '_pos' + str(b): coef})
                    print("Time to run single LDA iter: " + str(time.time() - start_time))

                # run LDA with individual cell dropout
                if self.do_dropout:
                    logger.warning("LDA dropout doesn't work yet for positional LDA")
                #     start_time = time.time()
                #     logger.info(f"Running LDA predictor dropout on {variable}")
                #     X_drop = []
                #     for drop in np.arange(1, np.shape(X)[1]):
                #         X_drop.append(np.delete(X, drop, axis=1))
                #     args_list = [(x, target) for x in X_drop]
                #     dropouts = self.PPool.mp_pool.map(parallel_function, args_list)
                #     dropout_pa.update({variable: dropouts})
                #     print("Time to run LDA on " + str(np.shape(X)[1]) + " dropouts: " + str(time.time() - start_time))

                # run linear shift on different angles
                if self.do_LS:
                    start_time = time.time()
                    LS_output = LinearShift(
                        X,
                        y=target,
                        PPool=self.PPool,
                        stat_computation_func=linear_discriminant_analysis,
                        step=40,
                        size_of_central_chunk=np.round(np.shape(X)[0] * 0.9),
                    )
                    LS_compiled.update({variable + '_pos' + str(b): LS_output})
                    del LS_output

    logger.info(f"Finally! It's time to save LDA output on {self.condition}")
    if self.do_LDA:
        with open(self.LDA_out, "wb") as fp:
            pickle.dump(prediction_accuracy, fp)
        coef_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_prediction_coef" + ".pkl"
        with open(coef_out, "wb") as fp:
            pickle.dump(prediction_coef, fp)

    if self.do_LS:
        with open(self.LS_out, "wb") as fp:
            pickle.dump(LS_compiled, fp)

    if self.do_dropout:
        with open(self.dropout_out, "wb") as fp:
            pickle.dump(dropout_pa, fp)