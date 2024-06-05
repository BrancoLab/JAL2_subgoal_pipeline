# OS libaries
from loguru import logger
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import time

# import functions
from behave_analysis.analyze.LDA.LDA_plotting import (
    real_predicted_trace,
    real_predicted_hist,
    residual_distribution,
)
from behave_analysis.analyze.LDA.LDA_utils import (
    plotConfusionMatrix,
    compute_prediction_accuracy,
    compute_prediction_accuracy_vect,
)
from behave_analysis.analyze.LDA.LDA_preprocess import (
    binDfbyEpoch,
    zscore_predictors,
)
from behave_analysis.analyze.regression_decoders.pytorch.working_models.LSTM_within_LDA import fit_LSTM, predict_LSTM



## --------------- MAIN LDA FUNCTION

def linear_discriminant_analysis(
    X, pos_ang, epoch_num=6, fr=40, return_coef=False, discriminant_type="linear", plotting=False, self=None, title=None
):
    """
    A function for doing LDA on data.
    This function iterates over the crossvalidation epochs, runs the decoder, computes the confusion matrix and the average prediction accuracy

    WARNING: currently this function can be used for linear shift statistics but will do LDA, not QDA

    pos_ang can have multiple columns - only the first column is the target of our predictions, but the other columns are used for binning the data (e.g. by position)

    """
    # confirm that if you have invalid y values, you delete those rows now!
    # (this can happen when doing LDa for distance if the mouse is in invalid distances)
    bad_frames = np.logical_or(pos_ang[0, :] < 1, pos_ang[0, :] > len(self.bin_centre))
    X = X[bad_frames == False, :]
    pos_ang = pos_ang[:, bad_frames == False]

    # initialize variables
    n_bins = len(np.unique(pos_ang[0, :]))
    coef_matrix = np.empty((n_bins, np.shape(X)[1] - 1, epoch_num))  # -1 on the number of clusters, because we have an extra column in X
    conf_matrix_all_train = np.empty((n_bins, n_bins, epoch_num))
    conf_matrix_all_test = np.empty((n_bins, n_bins, epoch_num))

    # chunk into epochs
    X, Y, epochs = binDfbyEpoch(X, pos_ang, epoch_num)  # after this Y is just the angles to predict
    X = X[:, 1:]  # the first column is frame id and you no longer need it
    _, counts = np.unique(epochs, return_counts=True)
    print(len(Y))

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
            start_time = time.time()
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
                logger.warning(
                    f"You're attempting to run the LDA pipeline with LSTM decoding - this may not work on distance or vectors. It certainly doesn't work with linshift"
                )
                # run LSTM
                model, seq_length = fit_LSTM(X1, self.bin_centre[y1 - 1], X2, self.bin_centre[y2 - 1])
                y_hat_train = predict_LSTM(model, X1, seq_length).reshape(-1)
                y_hat_test = predict_LSTM(model, X2, seq_length).reshape(-1)

                # convert predicted output back to bins
                y_hat_train = np.digitize(y_hat_train, self.bins)
                y_hat_test = np.digitize(y_hat_test, self.bins)

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
                if "dist" in title:
                    titleclass = "dist (cm)"
                elif "vect" in title:
                    titleclass = "class"
                else:
                    titleclass = "angle (rad)"

                if "vect" not in title:
                    y_hat_train = self.bin_centre[y_hat_train - 1]
                    y1 = self.bin_centre[y1 - 1]

                # scatter residuals vs predictions
                ax = plt.subplot2grid(shape=(4, 4), loc=(2, 1))
                residual_distribution(ax, y1, y_hat_train, titleclass)

                # plot histogram of frames per angle bin
                ax = plt.subplot2grid(shape=(4, 4), loc=(3, 0), colspan=2)
                real_predicted_hist(ax, y1, y_hat_train, "train data", titleclass)

                # look at data side-by-side
                ax = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)
                real_predicted_trace(ax, y1, y_hat_train, self.session.video.fps, "train data", titleclass)

            # plot confusion matrix of prediction on test data

            conf_matrix_all_test[:, :, counter] = plotConfusionMatrix(y2, y_hat_test, "test data", plt.subplot2grid(shape=(4, 4), loc=(2, 2)))

            if plotting:
                if "vect" not in title:
                    y_hat_test = self.bin_centre[y_hat_test - 1]
                    y2 = self.bin_centre[y2 - 1]

                # scatter residuals vs predictions
                ax = plt.subplot2grid(shape=(4, 4), loc=(2, 3))
                residual_distribution(ax, y2, y_hat_test, titleclass)

                # plot histogram of frames per angle bin
                ax = plt.subplot2grid(shape=(4, 4), loc=(3, 2), colspan=2)
                real_predicted_hist(ax, y2, y_hat_test, "test data", titleclass)

                # look at data side-by-side
                ax = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4)
                real_predicted_trace(ax, y2, y_hat_test, self.session.video.fps, "test data", titleclass)

                plt.tight_layout()
                filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i) + ".png"
                plt.savefig(filename)
                if self.show_plots:
                    plt.show()
                plt.close()

            # fit_time = time.time()
            # print('Time to fit&predict LDA is ' + str(fit_time - pre_time))

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

        if hasattr(self, "target_key"):  # TODO this will break in linshit
            prediction_accuracy = compute_prediction_accuracy_vect(np.mean(conf_matrix_all_test, axis=2), self.target_key[:, np.unique(Y) - 1])
        else:
            prediction_accuracy = compute_prediction_accuracy(np.mean(conf_matrix_all_test, axis=2))
    coef = np.mean(coef_matrix, axis=2)

    # if you want to look at how similar the weights are across epochs
    # peak_weight = np.amax(coef_matrix[:,:,0],axis = 0)
    # fig, axs = plt.subplots(1, np.shape(coef_matrix)[2])
    # for i in np.arange(np.shape(coef_matrix)[2]):
    #     axs[i].imshow(coef_matrix[:,np.argsort(peak_weight),i].T, aspect = 'auto')

    if return_coef:
        return prediction_accuracy, coef
    else:
        return prediction_accuracy


def parallel_function(args):
    """A function that unpacks a tuple of X and y and runs LDA"""
    X, y = args
    out = linear_discriminant_analysis(X, y)
    return out
