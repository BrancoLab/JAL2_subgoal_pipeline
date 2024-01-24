import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from behave_analysis.analyze.regression_decoders.sklearn_decoders.sk_models import rf_model, gbr_model
from behave_analysis.analyze.regression_decoders.sklearn_decoders.input import gen_random_pred_array, split_data, remove_hdir_cells


def sklearn_main(session, video_df, cluster_matrix, cluster_labels, d_reduction=False, remove_hdir=True):
    """
    NOTE: PCA seems to make the models worse
    """
    if remove_hdir:
        logger.info("Removing hdir cells")
        cluster_matrix = remove_hdir_cells(session, cluster_matrix, cluster_labels)

    # Create a random array of angles to use as a predictor to check model is not overfitting
    random_y = gen_random_pred_array(cluster_matrix)

    # Select the predictors
    hdir = np.asarray(video_df["hdir"]).reshape(len(video_df["hdir"]))
    hsa = np.asarray(video_df["hsa"]).reshape(len(video_df["hsa"]))

    # Choose three random locations to use as predictors, speed is an issue here
    random_y_loc1 = np.asarray(video_df["head_randP_1"]).reshape(len(video_df["head_randP_1"]))
    random_y_loc2 = np.asarray(video_df["head_randP_50"]).reshape(len(video_df["head_randP_50"]))
    random_y_loc3 = np.asarray(video_df["head_randP_100"]).reshape(len(video_df["head_randP_100"]))

    # Create a dictionary of predictors
    predictors = {
        "hdir": hdir,
        "hsa": hsa,
        "random": random_y,
        "random_loc1": random_y_loc1,
        "random_loc2": random_y_loc2,
        "random_loc3": random_y_loc3,
    }

    # Run PCA if selected
    if d_reduction:
        pca = PCA(n_components=15)
        pca.fit(cluster_matrix)
        cluster_matrix = pca.transform(cluster_matrix)

    # init empty dictionaries to store results
    predicted_angles = {}
    r2_scores = {}

    # Run the models
    for key, predictor in predictors.items():
        X_train, X_test, Y_train, Y_test = split_data(cluster_matrix, predictor, test_size=0.2)
        logger.info(f"Running decoders for predictor: {key}")
        rff_r2, rff_y_pred = rf_model(X_train, Y_train, X_test, Y_test)
        gbr_r2, gbr_y_pred = gbr_model(X_train, Y_train, X_test, Y_test)

        # Make a dictionary for each predictor and each model
        # predicted_angles[key] = {"rff": rff_y_pred, "svr": svr_y_pred, "gbr": gbr_y_pred}
        # r2_scores[key] = {"rff": rff_r2, "svr": svr_r2, "gbr": gbr_r2}

        plot_and_save_results(Y_test, rff_y_pred, gbr_y_pred, rff_r2, gbr_r2, key)

        logger.info(f"Finished running for predictor: {key}")


def plot_and_save_results(y_test, rff_pred, gbr_pred, rff_r2, gbr_r2, key):
    # Plot the first 500 samples
    index = 500
    plt.figure(figsize=(10, 10))
    plt.plot(y_test[:index], label="True")
    plt.plot(rff_pred[:index], label="RFF")
    plt.plot(gbr_pred[:index], label="GBR")
    plt.legend()
    plt.title(f"First 500 samples with results R2: RFF: {rff_r2}, GBR: {gbr_r2}")
    # plt.show()
    plt.savefig(f"{key}_first_500_samples.png") 
    
