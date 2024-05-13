# OS libaries
from loguru import logger
import numpy as np
import polars as pl
import pickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from multiprocessing.pool import Pool
import multiprocessing

# import functions
from behave_analysis.analyze.filtering_data.filtering_functions import identify_angles, generate_bin_angles
from behave_analysis.analyze.stats.linshit import LinearShift
from behave_analysis.analyze.LDA.LDA_plotting import (
    across_conditions_LDA_map,
    plot_LDA_model,
)
from behave_analysis.analyze.LDA.LDA_utils import (
    BuildSavingFolder,
    plotConfusionMatrix,
    compute_prediction_accuracy,
    check_if_we_do_LDA,
    fill_dict_with_zeros,
)
from behave_analysis.analyze.LDA.LDA_preprocess import (
    select_relevant_frames,
    BinDfbyPos,
    BinDfbyAngle,
    ProcessPredictors,
    binDfbyEpoch,
    exclude_proximal_frames,
    zscore_predictors,
)
from behave_analysis.analyze.regression_decoders.pytorch.working_models.LSTM_within_LDA import fit_LSTM, predict_LSTM


def LDA(self, settings):
    """
    A wrapper function that figures out all the conditions across which to run decoding analysis.
    It will iterate over all conditions and compartments.
    If the decoding analyss has not yet been run or the user asked to force_redo, the analysis will be run,
    if not it will jump straight to plotting the prediction accuracy maps for all conditions
    """

    # figure out which angles we want to decode
    if np.logical_or(settings.run_LDA == "all", np.logical_and(type(settings.run_LDA) is list, settings.run_LDA[0] == "all")):
        angles = identify_angles(self.session)
        angles.append("randP")
    else:
        angles = settings.run_LDA

    # determine condition types
    if settings.condition_types == "experimental_conditions":
        condition_types = ["experimental_conditions"]
    elif settings.condition_types == "behavioral_conditions":
        condition_types = ["good_behavioral_conditions", "bad_behavioral_conditions"]
        # good means the times when mousie is doing correct homies,
        # bad is when mouse is doing incorrect homies
    elif "homing_number" in settings.condition_types:
        self.number_of_homings = int(settings.condition_types.replace("homing_number_", ""))
        condition_types = ["before_" + str(self.number_of_homings) + "good_homings", "after_" + str(self.number_of_homings) + "good_homings"]

    # run LDA across condition types, across compartments and across conditions
    for cond in condition_types:  # e.g. 'experimental_conditions', 'behavioral_conditions'
        self.condition_types = cond
        for comp in settings.compartment_split:  # ['all','threat_zone','shelter_compartment']
            self.compartment = comp
            for (
                c
            ) in (
                self.all_conditions
            ):  # e.g. 'all_time', 'pre_shelter', 'shelter_present', 'barrier_present', 'shelter_only', 'barrier_pre_flip', 'barrier_post_flip'
                self.condition = c
                self.savepath = BuildSavingFolder(self.dir, settings, self.cluster_type, self.condition_types, self.condition, self.compartment)
                check_if_we_do_LDA(self, settings)
                if np.logical_or(self.do_LDA, self.do_LS):
                    logger.info(
                        f"Run LDA on {self.cluster_type} data with condition {self.condition} in condition type {self.condition_types} in compartment {self.compartment}"
                    )
                    run_LDA_model(self, settings, angles)
                else:
                    logger.info(
                        f"LDA already run on this session for condition {self.condition} in condition type {self.condition_types} in compartment {self.compartment}"
                    )
                logger.info(f"Time for some overview plots")
                plot_LDA_model(self, settings)
        across_conditions_LDA_map(self, settings)


def run_LDA_model(self, settings, angles):
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

    for variable in angles:

        if variable != "randP":
            logger.info(f"Running LDA on {variable}")
            import time

            # start_time = time.time()
            # we can run LDA only for times when the mouse is far from the point we're trying to deocde the angle to
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
                prediction_coef, prediction_accuracy, LS_compiled, dropout_pa = fill_dict_with_zeros(
                    self, prediction_coef, prediction_accuracy, dropout_pa, LS_compiled, variable
                )
            else:
                # TODO: is there a more elegant way to not remake binned_pos each time?
                binned_pos = BinDfbyPos(self.filtered_video_df, self.session.video.height, self.session.video.width)
                binned_angles, frames, savename = BinDfbyAngle(self, variable, settings)
                X = ProcessPredictors(self, frames, settings)
                pos_ang = np.vstack((binned_angles.T, binned_pos))
                preprocess_time = time.time()
                # print('Time to preprocess is ' + str(preprocess_time - start_time))

                # run LDA on different angles
                if self.do_LDA:
                    pa, coef = linear_discriminant_analysis(
                        X,
                        pos_ang=pos_ang,
                        epoch_num=settings.epoch_num,
                        fr=self.session.video.fps,
                        return_coef=True,
                        discriminant_type=settings.discriminant_type,
                        plotting=True,
                        self=self,
                        title=savename,
                    )
                    prediction_accuracy.update({variable: pa})
                    prediction_coef.update({variable: coef})

                # run LDA with individual cell dropout
                if self.do_dropout:
                    logger.info(f"Running LDA predictor dropout on {variable}")
                    X_drop = []
                    for drop in np.arange(1, np.shape(X)[1]):
                        X_drop.append(np.delete(X,drop,axis=1))
                    args_list = [(x, pos_ang) for x in X_drop]
                    num_processes = multiprocessing.cpu_count()-1  # Adjust as needed
                    with Pool(num_processes) as pool:
                        dropouts = pool.map(parallel_function, args_list)
                    dropout_pa.update({variable: dropouts})

                # run linear shift on different angles
                if self.do_LS:
                    logger.info(f"Running linear shift on LDA on {variable}")
                    LS_output = LinearShift(
                        X,
                        y=pos_ang,
                        stat_computation_func=linear_discriminant_analysis,
                        step=40,
                        size_of_central_chunk=np.round(np.shape(X)[0] * 0.9),
                    )
                    LS_compiled.update({variable: LS_output})
                    lda_time = time.time()
                    print("Total Time to fit is " + str(lda_time - preprocess_time))
                    del LS_output

        else:  # if the variable is a random point
            n_randP = self.video_df.select(pl.col("^head_randP_.*$")).width
            for j in tqdm(np.arange(self.video_df.select(pl.col("^head_randP_.*$")).width), desc=f"Running LDA on random point out of  {n_randP}"):
                # we can run LDA only for times when the mouse is far from the point we're trying to deocde the angle to
                if settings.exclude_proximal > 0:
                    # logger.warning('You are excluding proximal frames! This reduces the amount of data available - recommend only doing this for experimental conditions')
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
                    binned_pos = BinDfbyPos(self.filtered_video_df, self.session.video.height, self.session.video.width)
                    binned_angles, frames, savename = BinDfbyAngle(self, str("head_randP_" + str(j)), settings)
                    X = ProcessPredictors(self, frames, settings)
                    pos_ang = np.vstack((binned_angles.T, binned_pos))

                    # run LDA on different angles
                    if self.do_LDA:
                        pa, coef = linear_discriminant_analysis(
                            X,
                            pos_ang=pos_ang,
                            epoch_num=settings.epoch_num,
                            fr=self.session.video.fps,
                            return_coef=True,
                            discriminant_type=settings.discriminant_type,
                            plotting=False,
                            self=self,
                            title=savename,
                        )
                        prediction_accuracy.update({str(variable + str(j)): pa})
                        prediction_coef.update({str(variable + str(j)): coef})

                    # run linear shift on different angles
                    if self.do_LS:
                        LS_output = LinearShift(
                            X,
                            y=pos_ang,
                            stat_computation_func=linear_discriminant_analysis,
                            step=40,
                            size_of_central_chunk=np.round(np.shape(X)[0] * 0.9),
                        )
                        LS_compiled.update({str(variable + str(j)): LS_output})
                        del LS_output

    logger.info(f"Time to save LDA output on {variable}")
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


## --------------- MAIN LDA FUNCTION


def linear_discriminant_analysis(
    X, pos_ang, epoch_num=6, fr=40, return_coef=False, discriminant_type="linear", plotting=False, self=None, title=None
):
    """
    A function for doing LDA on data.
    This function iterates over the crossvalidation epochs, runs the decoder, computes the confusion matrix and the average prediction accuracy

    WARNING: currently this function can be used for linear shift statistics but will do LDA, not QDA

    pos_ang can have multiple columns - only the first column is used to predictions, but the other columns are used for binning the data (e.g. by position)

    """

    # initialize variables
    # import time
    # start_time = time.time()
    n_bins = len(np.unique(pos_ang[0, :]))
    coef_matrix = np.empty((n_bins, np.shape(X)[1] - 1, epoch_num))  # -1 on the number of clusters, because we have an extra column in X
    conf_matrix_all_train = np.empty((n_bins, n_bins, epoch_num))
    conf_matrix_all_test = np.empty((n_bins, n_bins, epoch_num))

    # chunk into epochs
    X, Y, epochs = binDfbyEpoch(X, pos_ang, epoch_num)  # after this Y is just the angles to predict
    X = X[:, 1:]  # the first column is frame id and you no longer need it
    _, counts = np.unique(epochs, return_counts=True)

    if np.logical_or(np.amin(counts) < fr, len(np.unique(epochs)) < 2):
        prediction_accuracy = 0
    else:
        # LDA
        for counter, i in enumerate(np.unique(epochs).astype(int)):
            test_idx = epochs == (i)
            train_idx = epochs != (i)

            # figure set up
            if plotting:
                plt.figure(figsize=(20, 16))
                plt.subplots_adjust(hspace=0.3)

            # make train matrix of frames x clusters
            X1 = zscore_predictors(X[train_idx, :])

            # make test matrix of frames x clusters
            X2 = zscore_predictors(X[test_idx, :])

            # train model
            y1 = Y[train_idx]
            y2 = Y[test_idx]
            # pre_time = time.time()
            # print('Time to prep LDa is ' + str(pre_time - start_time))

            if discriminant_type == "LSTM":
                # convert y to values from -pi to pi
                bin_angles, bin_angle_center = generate_bin_angles(n_bins + 1)
                bin_angle_center = bin_angle_center[1:-1]

                # run LSTM
                model, seq_length = fit_LSTM(X1, bin_angle_center[y1 - 1], X2, bin_angle_center[y2 - 1])
                y_hat_train = predict_LSTM(model, X1, seq_length).reshape(-1)
                y_hat_test = predict_LSTM(model, X2, seq_length).reshape(-1)

                # convert predicted output back to bins
                y_hat_train = np.digitize(y_hat_train, bin_angles)
                y_hat_test = np.digitize(y_hat_test, bin_angles)

                # crop y to match predicted output
                if len(y_hat_train) != len(y1):
                    y1 = y1[len(y1) - len(y_hat_train) :]
                if len(y_hat_test) != len(y2):
                    y2 = y2[len(y2) - len(y_hat_test) :]

            else:
                if discriminant_type == "linear":
                    clf = LinearDiscriminantAnalysis()
                elif discriminant_type == "quadratic":
                    clf = QuadraticDiscriminantAnalysis()
                clf.fit(X1, y1)
                y_hat_train = clf.predict(X1)
                y_hat_test = clf.predict(X2)
                coef_matrix[:, :, counter] = clf.coef_

            # plot confusion matrix of prediction on training data
            conf_matrix_all_train[:, :, counter] = plotConfusionMatrix(y1, y_hat_train, "training data", plt.subplot2grid(shape=(4, 4), loc=(2, 0)))

            if plotting:
                # scatter residuals vs predictions
                ax = plt.subplot2grid(shape=(4, 4), loc=(2, 1))
                ax.scatter(y_hat_train, y_hat_train - y1)
                ax.set_xlabel("prediction")
                ax.set_ylabel("residual")
                ax.set_box_aspect(1)

                # plot histogram of frames per angle bin
                ax = plt.subplot2grid(shape=(4, 4), loc=(3, 0), colspan=2)
                ax.hist(y_hat_train, np.arange(1, n_bins + 2), alpha=0.75)
                ax.hist(y1, np.arange(1, n_bins + 2), alpha=0.75)
                ax.set_title("training data")
                ax.set_xlabel("classes")
                ax.set_ylabel("number of frames")

                # look at data side-by-side
                ax = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)
                x_time = np.arange(len(y1)) / (self.session.video.fps * 60)
                ax.plot(x_time, y_hat_train)
                ax.plot(x_time, y1)
                ax.legend(["prediction", "real"])
                ax.set_xlim((0, len(y1) / (self.session.video.fps * 60)))
                ax.set_title("training data")
                ax.set_ylabel("classes (binned angles)")
                ax.set_xlabel("time (mins)")

            # plot confusion matrix of prediction on test data

            conf_matrix_all_test[:, :, counter] = plotConfusionMatrix(y2, y_hat_test, "test data", plt.subplot2grid(shape=(4, 4), loc=(2, 2)))

            if plotting:
                # scatter residuals vs predictions
                ax = plt.subplot2grid(shape=(4, 4), loc=(2, 3))
                ax.scatter(y_hat_test, y_hat_test - y2)
                ax.set_xlabel("prediction")
                ax.set_ylabel("residual")
                ax.set_box_aspect(1)

                # plot histogram of frames per angle bin
                ax = plt.subplot2grid(shape=(4, 4), loc=(3, 2), colspan=2)
                ax.hist(y_hat_test, np.arange(1, n_bins + 2), alpha=0.75)
                ax.hist(y2, np.arange(1, n_bins + 2), alpha=0.75)
                ax.set_title("test data")
                ax.set_xlabel("classes")
                ax.set_ylabel("number of frames")

                # look at data side-by-side
                ax = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4)
                x_time = np.arange(len(y2)) / (self.session.video.fps * 60)
                ax.plot(x_time, y_hat_test)
                ax.plot(x_time, y2)
                ax.legend(["prediction", "real"])
                ax.set_xlim((0, len(y2) / (self.session.video.fps * 60)))
                ax.set_title("test data")
                ax.set_ylabel("classes")
                ax.set_xlabel("time (mins)")

                plt.tight_layout()
                filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i) + ".png"
                plt.savefig(filename)
                if self.show_plots:
                    plt.show()
                plt.close()

        if plotting:
            # plot average confusion matrix
            plt.figure(figsize=(20, 16))
            plt.subplots_adjust(hspace=0.3)
            ax = plt.subplot(1, 2, 1)
            ax.imshow(np.mean(conf_matrix_all_train, axis=2), cmap="Blues", vmin=0, vmax=1)
            ax.set_ylabel("real")
            ax.set_xlabel("predicted")
            ax.set_title("train")

            ax = plt.subplot(1, 2, 2)
            ax.imshow(np.mean(conf_matrix_all_test, axis=2), cmap="Blues", vmin=0, vmax=1)
            ax.set_ylabel("real")
            ax.set_xlabel("predicted")
            ax.set_title("test")

            filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + ".png"
            plt.savefig(filename)
            if self.show_plots:
                plt.show()
            plt.close()

        prediction_accuracy = compute_prediction_accuracy(np.mean(conf_matrix_all_test, axis=2))
        coef = np.mean(coef_matrix, axis=2)

        # if you want to look at how similar the weights are across epochs
        # peak_weight = np.amax(coef_matrix[:,:,0],axis = 0)
        # fig, axs = plt.subplots(1, np.shape(coef_matrix)[2])
        # for i in np.arange(np.shape(coef_matrix)[2]):
        #     axs[i].imshow(coef_matrix[:,np.argsort(peak_weight),i].T, aspect = 'auto')

        # fit_time = time.time()
        # print('Time to fit&predict LDA is ' + str(fit_time - pre_time))
    if return_coef:
        return prediction_accuracy, coef
    else:
        return prediction_accuracy

def parallel_function(args):
    '''A function that unpacks a tuple of X and y and runs LDA'''
    X,y = args
    out = linear_discriminant_analysis(X, y)
    return out