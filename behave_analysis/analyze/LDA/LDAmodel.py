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

# import functions
from behave_analysis.analyze.filtering_data.filtering_functions import identify_angles, generate_bin_angles
from behave_analysis.analyze.LDA.LDAlinearshift import LinearShift
from behave_analysis.analyze.LDA.LDA_plotting import (
    PlotLSPredictionAccuracy,
    PlotPredictionAccuracy,
    PredictionAccuracyMapped,
    across_conditions_LDA_map,
)
from behave_analysis.analyze.LDA.LDA_utils import BuildSavingFolder, plotConfusionMatrix, compute_prediction_accuracy, check_if_we_do_LDA
from behave_analysis.analyze.LDA.LDA_preprocess import (
    select_relevant_frames,
    BinDfbyPos,
    BinDfbyAngle,
    ProcessPredictors,
    binDfbyEpoch,
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
                self.LDA_out, self.LS_out, self.do_LDA, self.do_LS = check_if_we_do_LDA(self, settings)
                if np.logical_or(self.do_LDA, self.do_LS):
                    logger.info(
                        f"Run LDA on {self.cluster_type} data with condition {self.condition} in condition type {self.condition_types} in compartment {self.compartment}"
                    )
                    run_LDA_model(self, settings, angles)
                else:
                    logger.info(
                        f"LDA already run on this session for condition {self.condition} in condition type {self.condition_types} in compartment {self.compartment}"
                    )
        across_conditions_LDA_map(self, settings)


def run_LDA_model(self, settings, angles):
    """
    A function that iterates across all angles and runs decoder analysis and linear shift statistics based on user settings
    It will also make bar plots of prediction accuracy across all angles and a map of prediction accuracy for the random points
    """

    prediction_accuracy = {}
    LS_compiled = {}
    title = []

    self.filtered_video_df = select_relevant_frames(self)
    # if no frames meet the criteria, make this condition blank
    if len(self.filtered_video_df) == 0:
        pa = 0
        LS_out = 0
        for variable in angles:
            if variable != "randP":
                title = np.append(title, variable)
                if self.do_LDA:
                    prediction_accuracy.update({variable: pa})
                if self.do_LS:
                    LS_compiled.update({variable: LS_out})
            else:
                for j in np.arange(self.video_df.select(pl.col("^head_randP_.*$")).width):
                    title = np.append(title, str("randP" + str(j)))
                    var = str(variable + str(j))
                    if self.do_LDA:
                        prediction_accuracy.update({var: pa})
                    if self.do_LS:
                        LS_compiled.update({var: LS_out})

    else:
        binned_pos = BinDfbyPos(self)
        for variable in angles:
            if variable != "randP":
                logger.info(f"Running LDA on {variable}")
                binned_angles, frames, savename = BinDfbyAngle(self, variable, settings)
                X = ProcessPredictors(self, frames, settings)

                # run LDA on different angles
                if self.do_LDA:
                    pa = linear_discriminant_analysis(
                        X,
                        Y=binned_angles.T,
                        binned_pos=binned_pos,
                        discriminant_type=settings.discriminant_type,
                        plotting=True,
                        settings=settings,
                        self=self,
                        title=savename,
                    )
                    prediction_accuracy.update({variable: pa})

                # run linear shift on different angles
                if self.do_LS:
                    logger.info(f"Running linear shift on LDA on {variable}")
                    LS_output = LinearShift(
                        X, y=binned_angles.T, stat_computation_func=linear_discriminant_analysis, size_of_central_chunk=np.round(np.shape(X)[0] / 3)
                    )
                    LS_compiled.update({variable: LS_output})
                    del LS_output
                title = np.append(title, variable)

            else:
                n_randP = self.video_df.select(pl.col("^head_randP_.*$")).width
                for j in tqdm(
                    np.arange(self.video_df.select(pl.col("^head_randP_.*$")).width), desc=f"Running LDA on random point out of  {n_randP}"
                ):
                    binned_angles, frames, savename = BinDfbyAngle(self, str("head_randP_" + str(j)), settings)
                    X = ProcessPredictors(self, frames, settings)

                    # run LDA on different angles
                    if self.do_LDA:
                        pa = linear_discriminant_analysis(
                            X,
                            Y=binned_angles.T,
                            binned_pos=binned_pos,
                            discriminant_type=settings.discriminant_type,
                            plotting=False,
                            settings=settings,
                            self=self,
                            title=savename,
                        )
                        prediction_accuracy.update({str(variable + str(j)): pa})

                    # run linear shift on different angles
                    if self.do_LS:
                        LS_output = LinearShift(
                            X,
                            y=binned_angles.T,
                            stat_computation_func=linear_discriminant_analysis,
                            size_of_central_chunk=np.round(np.shape(X)[0] / 3),
                        )
                        LS_compiled.update({str(variable + str(j)): LS_output})
                        del LS_output
                    title = np.append(title, str("randP" + str(j)))

    # make a plot of prediction accuracy across variables
    if self.do_LDA:
        with open(self.LDA_out, "wb") as fp:
            pickle.dump(prediction_accuracy, fp)
    else:
        with open(self.LDA_out, "rb") as dill_file:
            prediction_accuracy = pickle.load(dill_file)
            
    # NOTE - Commenting this out of main branch as it doesn't work on Laurence's machine
    # BUG - Let's fix this 
    # PlotPredictionAccuracy(self, prediction_accuracy, title)

    # make a plot of prediction accuracy across variables with linear shift stats
    if settings.linear_shift:
        if self.do_LS:
            with open(self.LS_out, "wb") as fp:
                pickle.dump(LS_compiled, fp)
        else:
            with open(self.LS_out, "rb") as dill_file:
                LS_compiled = pickle.load(dill_file)
        PlotLSPredictionAccuracy(self, LS_compiled, title)

    # map random points on arena:
    if len(list(filter(lambda x: "randP" in x, prediction_accuracy.keys()))) > 10:
        PredictionAccuracyMapped(self, prediction_accuracy)


## --------------- MAIN LDA FUNCTION


def linear_discriminant_analysis(X, Y, binned_pos, discriminant_type="linear", plotting=False, settings=None, self=None, title=None):
    """
    A function for doing LDA on data.
    This function iterates over the crossvalidation epochs, runs the decoder, computes the confusion matrix and the average prediction accuracy

    WARNING: currently this function can be used for linear shift statistics but will do LDA
    """

    # initialize variables
    n_bins = len(np.unique(Y))
    epoch_num = settings.epoch_num
    conf_matrix_all_train = np.empty((n_bins, n_bins, epoch_num))
    conf_matrix_all_test = np.empty((n_bins, n_bins, epoch_num))

    # chunk into epochs
    X, Y, epochs = binDfbyEpoch(X, Y, binned_pos, epoch_num)
    X = X[:, 1:]  # the first column is frame id and you no longer need it
    _, counts = np.unique(epochs, return_counts=True)
    if np.logical_or(np.amin(counts) < self.session.video.fps, len(np.unique(epochs)) < 2):
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
            X1 = X[train_idx, :]

            # make test matrix of frames x clusters
            X2 = X[test_idx, :]

            # train model
            y1 = Y[train_idx]
            y2 = Y[test_idx]

            if discriminant_type == "LSTM":
                # convert y to values from -pi to pi
                bin_angles, bin_angle_center = generate_bin_angles(settings.number_of_bins)
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

            # plot confusion matrix of prediction on training data
            conf_matrix_all_train[:, :, counter] = plotConfusionMatrix(y1, y_hat_train, "training data", plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

            if plotting:
                # plot histogram of frames per angle bin
                ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
                ax.hist(y_hat_train, np.arange(1, n_bins + 2))
                ax.hist(y1, np.arange(1, n_bins + 2))
                ax.set_title("training data")

                # look at data side-by-side
                ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
                ax.plot(y_hat_train)
                ax.plot(y1)
                ax.legend(["prediction", "real"])
                ax.set_title("training data")
                ax.set_ylabel("binned angles")
                ax.set_xlabel("time")

            # plot confusion matrix of prediction on test data

            conf_matrix_all_test[:, :, counter] = plotConfusionMatrix(y2, y_hat_test, "test data", plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

            if plotting:
                # plot histogram of frames per angle bin
                ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
                ax.hist(y_hat_test, np.arange(1, n_bins + 2))
                ax.hist(y2, np.arange(1, n_bins + 2))
                ax.set_title("test data")

                # look at data side-by-side
                ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
                ax.plot(y_hat_test)
                ax.plot(y2)
                ax.legend(["prediction", "real"])
                ax.set_title("test data")
                ax.set_ylabel("binned angles")
                ax.set_xlabel("time")

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

    return prediction_accuracy
