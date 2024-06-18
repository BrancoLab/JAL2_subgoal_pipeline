"""
TODO:
-- Remove shelter times?
-- Handle the different conditions of the homings, i.e split the heatmap by the different conditions
-- O
"""

from pathlib import Path

import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import ttest_rel
from tqdm import tqdm

from behave_analysis.utils.color_funcs import get_color_based_on_neural_activity
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.single_trial.tests import UnitTests

class SingleTrialRegression:
    """A class that performs single trial regression analysis on the data"""

    def __init__(self, design_matrix: pd.DataFrame, save_path: Path, dependents_df: pd.DataFrame, tracking_data, homing_list, spike_homing_list, condition_per_homing, cluster_ids):
        """Initialize the single trial regression analysis object
        
        Args:
            design_matrix (pd.DataFrame): The design matrix containing the neural data
            save_path (Path): The path to save the results
            dependents_df (pd.DataFrame): The dependent variables to run the regression on
            tracking_data (pd.DataFrame): The tracking data
            homing_list (list): A list of homings
            spike_homing_list (list): A list of spike data per homing, each item is a np matrix 
            condition_per_homing (list): A list of conditions per homing""" 
        
        logger.info("Initializing the single trial regression analysis object")
        self.cluster_ids = cluster_ids
        self.design_matrix = design_matrix
        self.homing_list = homing_list
        self.spike_homing_list = spike_homing_list
        self.condition_per_homing = condition_per_homing
        self.tracking_data = tracking_data
        self.save_path = save_path
        self.dependent_names = list(dependents_df.columns)
        logger.info("The dependent variables to run in this regression are: {}", self.dependent_names)
        self.dependents_df = dependents_df
        self.number_of_neurons = self.design_matrix.shape[1] - 1  # Subtract 1 for the homing id column
        self.angular_dependent_vars = [
            "h_bar_north_a",
            "hdir",
            "hsa",
            "h_bar_south_a",
        ]  # define here all potential angular variables so we can switch between coordinate systems
        self.encompasing_set = ["h_bar_north_a", "hdir", "hsa", "h_bar_south_a", "mouse_y_position", "velocity"]
    
        # --------------------- Functions for running the regression ---------------------
        # Initialize the plotting object
        self.plotter = RegressionPlotting(save_path)
    
        sig_index_coefficients_indices, og_coefficients_sig_in_both = self.run(
            run_all_dependent_variables=True, shift_neural_data=False, explore_coeffs_with_other_predictors=True, run_hiarchical_regression=False
        )
        
        # Plot a heatplot of the design matrix for the north index
        self.plotter.plot_clustered_heatmap(
            self.design_matrix,
            dependents_df=self.dependents_df,
            significant_neuron_ids=sig_index_coefficients_indices,
            og_coefficients=og_coefficients_sig_in_both,
            index_label="index_north",
        )

        # Plot the escape trajectories with the neural activity
        sig_cluster_ids = self.retrieve_clu_ids_sig_to_index(sig_index_coefficients_indices, cluster_ids)
        self.plotter.plot_all_homings_with_neural_activity(
            homing_list=self.homing_list, 
            spike_data_per_homing=self.spike_homing_list, 
            tracking_data=self.tracking_data, 
            condition_per_homing=self.condition_per_homing,
            cluster_ids=self.cluster_ids,
            sig_clu_ids=sig_cluster_ids,
        ) 
    # --------------------- Functions for running the regression ---------------------

    def run(
        self, run_all_dependent_variables: bool, shift_neural_data: bool, explore_coeffs_with_other_predictors: bool, run_hiarchical_regression: bool
    ) -> None:
        """This function runs the regression with different modes depending on the analysis you want to conduct:

        Modes:
        - run_all_dependent_variables: Run the model for all dependent variables
        - shift_neural_data: Shift the neural data and run the model to see how the R2 score changes with chance
        - explore_coeffs_with_other_predictors: Explore the coefficients with other predictors to see how they change"""


        if run_all_dependent_variables:
            r2_score_for_all_dependents, _, _, mse = self.run_all_dependent_variables()
            self.plotter.plot_the_r2_scores_for_all_dependents(r2_scores=r2_score_for_all_dependents)
            logger.success("The model has been run for all dependent variables")

        if shift_neural_data:
            # NOTE - Still only works for one index location at the moment
            shifts, og_r2, shifted_r2_ols = self.run_all_shifts()
            self.plotter.plot_shift_results(shifts, og_r2, shifted_r2_ols)
            logger.success("The model has been run for all shifts")

        if explore_coeffs_with_other_predictors:

            # Where og is the original gangster and comp is the comparison model
            INDEX = "index_north"
            # index = "index_south"

            og_r2_score, og_coefficients, og_p_values = self.run_just_one_dependent_variable(INDEX, self.design_matrix)
            comparison_design_matrix = self.add_other_predictors_to_design_matrix(self.design_matrix, self.encompasing_set)
            comp_r2_score, comp_predictors_coeffs, comp_predictors_p_value = self.run_just_one_dependent_variable(INDEX, comparison_design_matrix)

            # Select only the neural data coefficients and p values (Excluding intercept and non neural coeffs)
            og_coefficients = og_coefficients[: self.number_of_neurons]
            og_p_values = og_p_values[: self.number_of_neurons]
            comp_predictors_coeffs = comp_predictors_coeffs[: self.number_of_neurons]
            comp_predictors_p_value = comp_predictors_p_value[: self.number_of_neurons]

            # Take the indexes that are significant in both models
            sig_in_both_models = np.where((og_p_values < 0.05) & (comp_predictors_p_value < 0.05))[0]

            sig_index_coefficients_indices = np.where(og_p_values < 0.05)  # Which coefficients are significant in the original model
            self.plotter.plot_proportion_of_coeffs_that_remain_significant(og_p_values, comp_predictors_p_value, og_r2_score, comp_r2_score)
            # Check whether the significant coefficients change significantly between the two models
            x1 = np.ones(len(og_coefficients))
            x2 = 2 * np.ones(len(comp_predictors_coeffs))
            self.plotter.plot_coefficients_between_models(
                x1, x2, og_coefficients, comp_predictors_coeffs, sig_index_coefficients_indices, ttest_func=self.repeat_observation_ttest
            )
            logger.success("The model has been run to compare base model with encompassing model")

        if run_hiarchical_regression:
            h_results = self.run_hiararchical_regression(INDEX)
            self.plotter.plot_adjusted_r2_scores_for_hierarchy(h_results)
            logger.success("The model has been run for the hierarchical regression")

        # check if sig_index_coefficients_indices is defined
        assert "sig_index_coefficients_indices" in locals(), "sig_index_coefficients_indices is not defined"

        return sig_in_both_models, og_coefficients[sig_in_both_models]

    def retrieve_clu_ids_sig_to_index(self, sig_index, cluster_ids: np.ndarray) -> np.ndarray:
        """Retrieve the cluster ids that are significant to the index response variable"""
        return cluster_ids[sig_index]

    def run_just_one_dependent_variable(self, dependent_var_name: str, design_matrix: pd.DataFrame) -> tuple:
        """Run the model for just one dependent variable"""

        # Create a save location for the dependent variable
        make_directory(self.save_path / "ols_regression" / dependent_var_name)

        ols_fold_results = self.run_ols_model_with_cross_val(
            design_matrix=design_matrix, dependent_variable=self.dependents_df[dependent_var_name].to_numpy(), dependent_var_name=dependent_var_name
        )
        dic = self.unpack_fold_results_and_average(ols_fold_results)
        r2_scores, coefficients, p_values = dic["mean_r2"], dic["mean_coefficients"], dic["mean_p_values"]

        return r2_scores, coefficients, p_values

    def run_all_dependent_variables(self):
        """Run the model for different dependent variables and store the mean R2 scores
        to see which dependent variable is best predicted by the neural data alone"""

        # Init some storage vars
        r2_scores = {}
        p_values = {}
        coefficients = {}
        mse = {}

        # Extract the dependent variables and make save directories for each of them to store the results
        for dependent_var_name in self.dependent_names:
            make_directory(self.save_path / "ols_regression" / dependent_var_name)

        # Run the model for each dependent variable
        for var_idx, var_name in enumerate(self.dependent_names):
            logger.info("Running the model for dependent variable: {}", var_name)
            multiple_dependent_scenario = False

            # If the dependent variable is not angular dependent then run a standard OLS model
            if var_name in self.angular_dependent_vars:
                multiple_dependent_scenario = True

            # A dictionary where each key is a fold index
            ols_fold_results = self.run_ols_model_with_cross_val(
                design_matrix=self.design_matrix,
                dependent_variable=self.dependents_df[var_name].to_numpy(),
                dependent_var_name=var_name,
                multi_dependent=multiple_dependent_scenario,
            )

            # Unpack the results and average them
            dic = self.unpack_fold_results_and_average(ols_fold_results)
            r2_scores[var_name], p_values[var_name], coefficients[var_name], mse[var_name] = (
                dic["mean_r2"],
                dic["mean_p_values"],
                dic["mean_coefficients"],
                dic["mean_mse"],
            )

        return r2_scores, p_values, coefficients, mse

    def run_hiararchical_regression(self, INDEX):
        """Conduct a looping hierarchical regression analysis where predictors are added one by one and the adjusted R2 is calculated"""
        # hardcode subsets
        s0 = []  # empty set just neural data
        s1 = ["hdir"]
        s2 = ["hdir", "velocity"]
        s3 = ["hdir", "velocity", "mouse_y_position"]
        s4 = ["hdir", "velocity", "mouse_y_position", "hsa"]
        s5 = ["hdir", "velocity", "mouse_y_position", "hsa", "h_bar_north_a"]
        full_set = [s0, s1, s2, s3, s4, s5, self.encompasing_set]

        # store the results
        hirarchical_results = {}

        for i, subset in enumerate(full_set):
            # First add the subset to the design matrix
            design_matrix = self.add_other_predictors_to_design_matrix(self.design_matrix, dependents_to_add=subset)

            # Run cross validation
            ols_fold_results = self.run_ols_model_with_cross_val(
                design_matrix=design_matrix,
                dependent_variable=self.dependents_df[INDEX].to_numpy(),
                dependent_var_name=INDEX,
            )

            unpacked_results = self.unpack_fold_results_and_average(ols_fold_results)
            unpacked_r2 = unpacked_results["mean_r2"]
            number_of_predictors = design_matrix.shape[1] - 1  # Subtract 1 for the homing id column
            avg_observations = len(self.design_matrix)
            adjusted_r2 = self.calculate_adjusted_r2(unpacked_r2, avg_observations, number_of_predictors)
            hirarchical_results[i] = adjusted_r2

        return hirarchical_results

    # ------------------------ Changing the design matrix or targets ------------------

    def add_other_predictors_to_design_matrix(self, design_matrix: pd.DataFrame, dependents_to_add: list) -> pd.DataFrame:
        """Add other predictors to the design matrix to see how they affect the coefficients and p values compared to the neural data alone

        If empty list is passed then the function will just return the original design matrix

        Logic:
        -- If the dependent variable is not angular then divide by the standard deviation
        -- If the dependent variable is angular then convert to sin and cos"""

        # Check if dependents to add is an empty list
        if dependents_to_add:
            design_matrix = design_matrix.copy()
            for dependent in dependents_to_add:

                # Scale non angular data when adding into design matrix
                if dependent not in self.angular_dependent_vars:
                    std = np.std(self.dependents_df[dependent].to_numpy())
                    design_matrix[dependent] = self.dependents_df[dependent].to_numpy() / std

                # Convert angular dependent variables to sin and cos
                else:
                    circ_var = self.dependents_df[dependent].to_numpy()
                    sin_component = np.sin(circ_var)
                    cos_component = np.cos(circ_var)
                    design_matrix[f"{dependent}_sin"] = sin_component
                    design_matrix[f"{dependent}_cos"] = cos_component

        return design_matrix

    def convert_angular_dependent_variables_to_cartesian(self, dependent_variable: np.ndarray) -> np.ndarray:
        """Split the dependent variable into two components, sin and cos"""
        sin_component = np.sin(dependent_variable)
        cos_component = np.cos(dependent_variable)
        result = np.column_stack((sin_component, cos_component))
        assert result.shape[1] == 2, "The dependent variable has not been split into two components. Just checking the shape has two columns"
        return result

    # --------------------- Cross validation functions ---------------------

    def unpack_fold_results_and_average(self, fold_results: dict):
        """For each cross validation fold, average the returned statistics to obtain a single value for each metric"""
        r2_scores = []
        coefficients = []
        p_values = []
        mse = []

        for _, results in fold_results.items():
            r2_scores.append(results["test_r2"])
            coefficients.append(results["train_coefficients"])
            p_values.append(results["train_p_values"])
            mse.append(results["test_mse"])

        # --------- below code to handle outliers of fold effecting the mean -----------------
        # Should produce a more robust mean

        # Convert to numpy arrays for easy manipulation
        r2_scores = np.array(r2_scores)

        # Identify and filter out outliers in r2_scores
        r2_median = np.median(r2_scores)
        deviation_from_median = np.abs(r2_scores - r2_median)
        median_absolute_deviation = np.median(deviation_from_median)
        # Using a threshold of 3 times the median absolute deviation to identify outliers
        threshold = 3 * median_absolute_deviation
        filtered_r2_scores = r2_scores[deviation_from_median < threshold]

        # Compute the mean of the filtered r2_scores
        mean_r2 = np.mean(filtered_r2_scores) if len(filtered_r2_scores) > 0 else np.mean(r2_scores)

        # Compute mean MSE
        mean_mse = np.mean(mse)

        # return dic of results
        return {
            "mean_r2": mean_r2,
            "mean_mse": mean_mse,
            "mean_coefficients": np.mean(coefficients, axis=0),
            "mean_p_values": np.mean(p_values, axis=0),
        }

    def run_ols_model_with_cross_val(
        self, design_matrix: pd.DataFrame, dependent_variable: np.ndarray, dependent_var_name: str, n_splits: int = 5, multi_dependent: bool = False
    ):
        """Run an OLS statsmodel regression on the data, conducting a 4-fold cross validation"""

        # drop the homing id column from the design matrix to leave only the predictors
        X = design_matrix.drop(columns=["homing_id"])
        y = dependent_variable
        groups = design_matrix["homing_id"].to_numpy()  # Grouping according to homing_id

        if multi_dependent:
            y = self.convert_angular_dependent_variables_to_cartesian(y)

        # Set up cross validation by homing id
        group_kfold = GroupKFold(n_splits)
        assert len(np.unique(groups)) == len(np.unique(design_matrix["homing_id"])), "Number of groups is not equal to the number of homing ids"

        # Saveing info
        ols_save_path = self.save_path / "ols_regression" / dependent_var_name
        ols_fold_results = {}

        # Split using GroupKFold, ensuring groups do not overlap between folds
        for fold, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if multi_dependent:
                ols_fold_results[fold] = self.multi_output_ols_regression(
                    X_train, y_train, fold, ols_save_path, X_test, y_test, name_of_dependent=dependent_var_name
                )

            if not multi_dependent:
                ols_fold_results[fold] = self.ols_regression_statsmodel(
                    X_train, y_train, fold, ols_save_path, X_test, y_test, name_of_dependent=dependent_var_name
                )

        return ols_fold_results

    # ------------------- Statistical functions -------------------

    def repeat_observation_ttest(
        self, indices_of_sig_coeffs: np.ndarray, original_model_coeffs: np.ndarray, comparison_model_coeffs: np.ndarray
    ) -> tuple:
        """Perform a paired t test to see if the coefficients between the original model and the comparison model are significantly different

        The null hypothesis is that the coefficients are the same between the two models.
            If the p value is less than 0.05 then the change is significant which is bad.
            If the p value is greater than 0.05 then the change is not significant which is good

        In order to counteract a mean of zero issue due to a scaling around zero we take the absolute value of the coefficients

        Returns:
            tuple: t_stat, p_value both floats"""
        og_abs = np.abs(original_model_coeffs[indices_of_sig_coeffs])
        comp_abs = np.abs(comparison_model_coeffs[indices_of_sig_coeffs])
        t_stat, p_value = ttest_rel(og_abs, comp_abs)
        return t_stat, p_value

    def calculate_adjusted_r2(self, r2_score: float, n: int, p: int) -> float:
        """Calculate the adjusted R2 score

        Args:
            r2_score (float): The R2 score
            n (int): The number of samples
            p (int): The number of predictors"""
        adjusted_r2 = 1 - (1 - r2_score) * (n - 1) / (n - p - 1)
        return adjusted_r2

    def scaled_MSE_by_MAD(self, ygt: np.ndarray, ypred: np.ndarray) -> float:
        """Calculate the scaled MSE by the MAD"""

        # If the dependent var is angular we need to handle the two components
        if len(ygt.shape) == 2:
            ygt_mad = np.mean(np.abs(ygt - np.median(ygt, axis=0)), axis=0)
            mse = mean_squared_error(ygt, ypred, multioutput="raw_values")
            smse_mad = mse / (ygt_mad**2)
            return np.mean(smse_mad)

        # If non angular then just calculate the smae
        else:
            ygt_mad = np.mean(np.abs(ygt - np.median(ygt)))
            mse = mean_squared_error(ygt, ypred)
            smse_mad = mse / (ygt_mad**2)
            return smse_mad

    # ------------------- Regression functions -------------------

    def ols_regression_sklearn(self, X_train, y_train, fold, save_path: Path, X_test, y_test) -> dict:
        """Performing the OLS regression

        NOTES:
        - first coefficient is the intercept
        - first p-value is for the intercept
        - second p-value is for the first coefficient

        Returns:
            dict: A dictionary containing the training and test R2 scores, coefficients and p-values
        """

        # ------------------- Training data -------------------
        ols_model = LinearRegression(fit_intercept=True)
        ols_model.fit(X_train, y_train)
        train_coefficients = ols_model.coef_
        train_r2 = ols_model.score(X_train, y_train)

        # Predict data for training data
        train_pred = ols_model.predict(X_train)

        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")

        # ------------------- Test data -------------------
        # Refit the model on the test data
        test_pred = ols_model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_coefficients = ols_model.coef_

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        # NOTE - p-values are not available in sklearn - matching the statsmodel output so that the dictionary is consistent

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": np.zeros(len(train_coefficients)),
            "train_r2": train_r2,
            "test_coefficients": test_coefficients,
            "test_p_values": np.zeros(len(train_coefficients)),
            "test_r2": test_r2,
            "test_predictions": test_pred,
        }

    def ols_regression_statsmodel(self, X_train, y_train, fold, save_path: Path, X_test, y_test, name_of_dependent: str) -> dict:
        """Performing the OLS regression

        NOTES:
         - first coefficient is the intercept
         - first p-value is for the intercept
         - second p-value is for the first coefficient

         Returns:
            dict: A dictionary containing the training and test R2 scores, coefficients and p-values
        """

        # Check for NaNs
        UnitTests.check_for_nans_and_inf(X_train, X_test)

        # ------------------- Training data -------------------
        X_train = sm.add_constant(X_train)
        train_mod = sm.OLS(y_train, X_train)
        train_results = train_mod.fit()  # needed for when not using regularized attribute
        # train_results = sm.OLS(y_train, X_train).fit_regularized(method='elastic_net', L1_wt=1.0) # L1_wt=1.0 is lasso, L1_wt=0.0 is ridge - Where lasso will assign weights to one of two colinear variables
        train_coefficients = train_results.params[1:]  # Exclude the intercept which is the first coefficient
        train_p_values = train_results.pvalues[1:]  # Exclude the intercept which is the first p-value
        train_r2 = train_results.rsquared
        # print(train_results.summary())

        # Predict data for training data
        train_pred = train_results.predict(X_train)

        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")

        # ------------------- Test data -------------------
        X_test = sm.add_constant(X_test)
        test_pred = train_results.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_mse = self.scaled_MSE_by_MAD(y_test, test_pred)

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data")

        plt.suptitle(f"R2 Train: {train_r2}, R2 Test: {test_r2} - {name_of_dependent}")

        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": train_p_values,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_predictions": test_pred,
            "test_mse": test_mse,
        }

    def svr_regression(self, X_train, y_train, fold, save_path: Path, X_test, y_test):
        """Performing the SVR regression"""

        svr_model = SVR(kernel="rbf", C=0.8, epsilon=0.05)  # You can tune these parameters
        svr_model.fit(X_train, y_train)

        # Predict and calculate R2 for training data
        train_pred = svr_model.predict(X_train)
        train_r2 = svr_model.score(X_train, y_train)

        # test data
        test_pred = svr_model.predict(X_test)
        test_r2 = svr_model.score(X_test, y_test)

        # Plot train and test predictions into one plot, different subplots
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"svr Fold {fold + 1} Train Data")

        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Svr Fold {fold + 1} Test Data")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")

        return test_r2

    def multi_output_ols_regression(self, X_train, y_train, fold, save_path: Path, X_test, y_test, name_of_dependent: str) -> dict:
        """NOTE - Pvalues and coefficients are fake this is just to get the R2 score for the multi output regression model"""

        # ------------------- Training data -------------------
        reg = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
        train_r2 = reg.score(X_train, y_train)
        pred_train = reg.predict(X_train)

        # Recombine the sin and cos components into a single angular variable
        pred_train_y = np.arctan2(pred_train[:, 0], pred_train[:, 1])  # Convert back to radians
        real_train_y = np.arctan2(y_train[:, 0], y_train[:, 1])  # Convert back to radians

        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(real_train_y)), real_train_y, label="True")
        plt.plot(np.arange(len(pred_train_y)), pred_train_y, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")

        # ------------------- Test data ------------------------------------
        pred_test = reg.predict(X_test)
        test_r2 = reg.score(X_test, y_test)
        test_mse = self.scaled_MSE_by_MAD(y_test, pred_test)

        # Recombine the sin and cos components into a single angular variable
        pred_test_y = np.arctan2(pred_test[:, 0], pred_test[:, 1])  # Convert back to radians
        real_test_y = np.arctan2(y_test[:, 0], y_test[:, 1])  # Convert back to radians

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(real_test_y)), real_test_y, label="True")
        plt.plot(np.arange(len(pred_test_y)), pred_test_y, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data. {name_of_dependent} - Test R2: {test_r2}")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        # Make up the p values and coefficients in this case
        train_coefficients = np.zeros(X_train.shape[1])
        train_p_values = np.zeros(X_train.shape[1])

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": train_p_values,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_predictions": pred_test_y,
            "test_mse": test_mse,
        }

    # -------------------- Shifted data analysis --------------------

    def run_all_shifts(self):
        """As a quick pass, shift the neural data backwards and forwards and see how the R2 score changes with each shift"""

        shifted_r2_ols = []
        og_r2 = None
        shifts = np.arange(-400, 400, 10)  # Shift forwards and backwards by 400 frames (40 frames is 1 second)
        for shift_amount in tqdm(shifts, desc="Processing shifts"):
            if shift_amount == 0:
                shifted_design_matrix = self.design_matrix
            else:
                shifted_design_matrix = self.shift_spikes(self.design_matrix, shift_amount)
            r2, _, _ = self.run_just_one_dependent_variable("index_south", shifted_design_matrix)
            if shift_amount == 0:
                og_r2 = r2
            shifted_r2_ols.append(r2)

        logger.success("Shifts have been completed")

        return shifts, og_r2, shifted_r2_ols

    def shift_spikes(self, design_matrix: pd.DataFrame, shift_amount: int) -> pd.DataFrame:
        """Shifts the spikes in the design matrix by a certain amount, wrapping around the end of the matrix"""
        new_index = (design_matrix.index + shift_amount) % len(design_matrix)
        return design_matrix.reindex(new_index).reset_index(drop=True)


class RegressionPlotting:
    """Class to handle the plotting of the regression results"""

    def __init__(self, save_path: Path):
        self.save_path = save_path

    def plot_clustered_heatmap(
        self, design_matrix: np.ndarray, dependents_df: pd.DataFrame, significant_neuron_ids, og_coefficients, index_label: str
    ) -> None:
        """Creates a heatmap of the design matrix ranked by coefficients from largest to smallest along side the index dependent variable

        Args:
            design_matrix (np.ndarray): _description_
            dependents_df (pd.DataFrame): _description_
            significant_neuron_ids (_type_): _description_
            og_coefficients (_type_): _description_
            index_label (str): either index_north or index_south
        """
        logger.info(f"Plotting a sorted heatmap based on the coefficients of the neurons for the dependent variable: {index_label}")

        # Sort the neuron ids by the largest coefficient to the smallest
        dic_map = {id: coeff for id, coeff in zip(significant_neuron_ids, og_coefficients)}
        sorted_dic = dict(sorted(dic_map.items(), key=lambda item: item[1], reverse=True))
        sorted_ids = list(sorted_dic.keys())
        assert len(sorted_ids) == len(significant_neuron_ids), "The sorted ids are not the same length as the significant neuron ids"
        design_matrix = design_matrix.drop(columns=["homing_id"])
        design_matrix = design_matrix.iloc[:, sorted_ids]

        # select the first 2000 frames
        frames = 2000

        # Normalize the matrix
        # Max firing rate across all neurons and frames
        max_firing_rate = design_matrix.max()
        assert len(max_firing_rate) == len(significant_neuron_ids), "The max firing rate array is not the same length as the significant neuron ids"
        normalized_matrix = design_matrix / max_firing_rate
        normalized_matrix = normalized_matrix.T  # Transpose to have the neurons on the y axis

        # take the first 2000 frames as a smaller chunk to make the plot more readable
        if 0:
            np_matrix = normalized_matrix.to_numpy()
            normalized_matrix = np_matrix[:, :frames]  # selct all neurons and the first 2000 frames
            index = dependents_df[index_label][:frames]
        else:
            index = dependents_df[index_label]

        # Plot a narrow subplot above imshow to show the dependent variable varying across the frames
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, height_ratios=[1, 4])  # Adjust height ratios as needed

        # Plot the first subplot -----------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(index, label=index_label, color="black")
        ax1.set_title(f"{index_label} Over Time")
        ax1.set_xlabel("Frame id of concatenated homings")
        ax1.set_ylabel(f"{index_label} value")
        ax1.legend()
        ax1.set_xlim(0, len(index))

        # Plot the second subplot -----------------------------------------------------------------
        ax2 = fig.add_subplot(gs[1, :])
        _ = ax2.imshow(normalized_matrix, cmap="viridis", aspect="auto")
        ax2.set_title("Heatmap of the Matrix")
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Cluster Index")

        plt.tight_layout()
        plt.savefig(self.save_path / f"{index_label}_clustered_heatmap.png")
        plt.close()

    def plot_shift_results(self, shifts, og_r2, shifted_r2_ols):
        """Plot the mean R2 scores for different shifts"""

        shifts = np.array(shifts)

        plt.figure(figsize=(12, 6))
        plt.plot(shifts, shifted_r2_ols, label="OLS R2", marker="o")
        plt.axhline(y=og_r2, color="b", linestyle="--", label="OLS Original R2")
        plt.xlabel("Shift Amount by frame")
        plt.ylabel("Mean R2 Score")
        plt.title("Mean R2 Score vs. Shift Amount")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path / "shifted_mean_r2_scores.png")
        plt.close()

    def plot_the_r2_scores_for_all_dependents(self, r2_scores):
        """Plot the R2 scores for the different dependent variables"""

        dependent_vars = list(r2_scores.keys())
        ols_r2_scores = [r2_scores[dependent_var] for dependent_var in dependent_vars]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(dependent_vars))
        bar_width = 0.35
        plt.bar(x + bar_width / 2, ols_r2_scores, bar_width, label="R2 scores")

        # Plot the mse score as text above the bar
        for i, r2 in enumerate(ols_r2_scores):
            height = ols_r2_scores[i]
            plt.text(x[i] + bar_width / 2, height + 0.01, str(np.around(ols_r2_scores[i], 2)), color="black", ha="center")

        plt.xlabel("Dependent Variable")
        plt.ylabel("Mean R2 Score")
        plt.title("Mean R2 Score for Different Dependent Variables.")
        plt.xticks(x + bar_width / 2, dependent_vars)
        plt.xticks(rotation=20)
        plt.xticks(fontsize=10)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path / "dependent_variable_r2_scores.png")
        plt.close()

    def plot_lines_between_points_on_plot(
        self, ax, x1: np.ndarray, x2: np.ndarray, original_model_coeffs: np.ndarray, comparison_model_coeffs: np.ndarray, iterator: np.ndarray
    ) -> None:
        """Connect the corresponding points with lines on a 2d plot to see changes across different models

        Args:
            ax: The axis to plot on
            x1: The x values for the first model
            x2: The x values for the second model
            original_model_coeffs: The coefficients for the first model
            comparison_model_coeffs: The coefficients for the second model
            iterator: The indices to iterate over to connect the points"""

        assert len(x1) == len(x2) == len(original_model_coeffs) == len(comparison_model_coeffs), "Lengths of arrays are not the same"
        for i in iterator:
            ax.plot([x1[i], x2[i]], [original_model_coeffs[i], comparison_model_coeffs[i]], color="gray", linestyle="--", linewidth=0.5)

    def plot_coefficients_between_models(self, x1, x2, original_model_coeffs, comparison_model_coeffs, sig_og_model_coeffs_indices, ttest_func):
        """Plot all and only the significant coefficients between the two models in separate plots"""

        # Make coefficients absolute
        original_model_coeffs = np.abs(original_model_coeffs)
        comparison_model_coeffs = np.abs(comparison_model_coeffs)

        # Create two axes
        fig, ax = plt.subplots(1, figsize=(10, 8))

        # Plot only the significant coefficients on the second plot
        ax.scatter(x1[sig_og_model_coeffs_indices], original_model_coeffs[sig_og_model_coeffs_indices], label="Neural data only")
        ax.scatter(
            x2[sig_og_model_coeffs_indices],
            comparison_model_coeffs[sig_og_model_coeffs_indices],
            label="Neural data + other predictors that are correlated with the index",
        )
        ax.legend()
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Base model", "Encompassing model"])
        # rotate the x labels
        plt.xticks(rotation=20)
        ax.set_ylabel("Coefficients")
        ax.set_title("A check of whether the coeffs that are sig for both models change significantly when adding other predictors")
        self.plot_lines_between_points_on_plot(ax, x1, x2, original_model_coeffs, comparison_model_coeffs, sig_og_model_coeffs_indices)

        # Put sig test of difference in title
        t_stat, p_value = ttest_func(sig_og_model_coeffs_indices, original_model_coeffs, comparison_model_coeffs)
        are_results_significant = "significant" if p_value < 0.05 else "not significant"
        formatted_p_value = format(p_value, ".4f")
        rounded_p_value = round(p_value, 4)
        fig.suptitle(f"Sig coeff (absolute) p-value: {rounded_p_value}. \n Difference is {are_results_significant}")

        plt.savefig(self.save_path / "coefficients_between_models.png")
        plt.close()

    def plot_proportion_of_coeffs_that_remain_significant(
        self, original_model_pvalues: np.ndarray, comparison_model_pvalues: np.ndarray, og_r2_score, comp_r2_score, alpha=0.05
    ) -> None:
        """Plots a bar chart showing the proportion of coefficients that remain significant between the two models, neural coefficients only"""

        # Check the number of coefficients that are significant in each model
        og_moel_significant_coeffs = np.sum(original_model_pvalues < alpha)
        all_predictors_significant = np.sum(comparison_model_pvalues < alpha)

        # Check the numebr of cells that remain sig after adding all predictors
        num_sig = 0
        num_sig = np.sum(
            [num_sig + 1 for i in range(len(original_model_pvalues)) if original_model_pvalues[i] < 0.05 and comparison_model_pvalues[i] < 0.05]
        )

        # Plot the number of cells that remain significant
        _, ax = plt.subplots()
        ax.bar(
            ["Base model", "Encompassing", "consistently sig"],
            [og_moel_significant_coeffs, all_predictors_significant, num_sig],
        )
        # rotate the x labels
        plt.xticks(rotation=10)
        ax.set_ylabel("Number of significant coefficients")
        ax.set_title(f" {num_sig} / {len(original_model_pvalues)} coeffs are significant under both models")
        plt.savefig(self.save_path / "proportion_of_significant_coeffs.png")
        plt.close()

    def plot_adjusted_r2_scores_for_hierarchy(self, hirarchical_results: dict):
        """Plot the adjusted R2 scores for the hierarchical regression analysis"""

        adjusted_r2_scores = hirarchical_results.values()
        x = np.arange(len(adjusted_r2_scores))
        plt.figure(figsize=(10, 6))
        plt.bar(x, adjusted_r2_scores)
        plt.xlabel("Model Complexity")
        plt.ylabel("Adjusted R2 Score")
        plt.title("Adjusted R2 Score for Hierarchical Regression Analysis")
        plt.xticks(x, ["s0", "s1", "s2", "s3", "s4", "s5", "full_set"])
        plt.grid(True)
        plt.savefig(self.save_path / "adjusted_r2_scores.png")
        plt.close()

    # ------------------- Plotting functions take from spatial efficienty + mine -------------------
    # refactor potential to make these shared components

    def plot_all_homings_with_neural_activity(self, homing_list, spike_data_per_homing, tracking_data, condition_per_homing, cluster_ids, sig_clu_ids):
        """Plot all homings with neural activity by neuron

        Args:
            homing_list (list): List of homing dataframes
            spike_data_per_homing (list): List of spike np matrices
            plot_sig_only (bool, optional): Plot only the significant neurons that are tuned to the index variable. Defaults to True.
            
        # NOTE should we use hdir location instead of body location
        """
        
        logger.info("Plotting all homings + neural activity by neuron")
        new_path = self.save_path / "neural_activity_plots"
        make_directory(new_path)
        num_neurons = spike_data_per_homing[0].shape[1]
        assert len(cluster_ids) == num_neurons, "The number of cluster ids is not the same as the number of neurons"

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        for cidx, clu_id in enumerate(cluster_ids):
            print(f"Plotting neuron {clu_id}")
            
            # Booleans to check if the condition has been plotted - Lazy way to avoid repeating the same plot
            b1 = False
            b2 = False
            b3 = False
            
            for idx, (homing, spikes) in enumerate(zip(homing_list, spike_data_per_homing)):
                con = condition_per_homing[idx]
                neuron_filter = spikes[:, cidx]  
                
                if con == "shelter_only": 
                    if not b1:
                        self.base_plotting(ax1, tracking_data, condition=con)
                        b1 = True
                    self.plot_escape_trajectories_with_neural_activity(neural_data=neuron_filter, behavioural_data=homing, ax=ax1)
                    
                if con == "barrier_pre_flip": 
                    if not b2:
                        self.base_plotting(ax2, tracking_data, condition=con)
                        b2 = True
                    self.plot_escape_trajectories_with_neural_activity(neural_data=neuron_filter, behavioural_data=homing, ax=ax2)
                    
                if con == "barrier_post_flip": 
                    if not b3:
                        self.base_plotting(ax3, tracking_data, condition=con)
                        b3 = True
                    self.plot_escape_trajectories_with_neural_activity(neural_data=neuron_filter, behavioural_data=homing, ax=ax3)
                    
            # Add a title to the plot
            ax1.set_title("Shelter Only")
            ax2.set_title("Barrier Pre Flip")
            ax3.set_title("Barrier Post Flip")
            if cidx in sig_clu_ids:
                fig.suptitle(f"Neural Activity overlaid homings for Neuron {clu_id} - Significant to index variable")
            else:
                fig.suptitle(f"Neural Activity overlaid homings for Neuron {clu_id} - Not Significant to Index Variable")
            plt.savefig(new_path / f"neural_activity_neuron_{clu_id}.png")
            # plt.savefig(new_path / f"neural_activity_neuron_{neuron}.eps")
            
            # Clear the axes for the next neuron
            ax1.cla()
            ax2.cla()
            ax3.cla()
        
        plt.close()

    def plot_escape_trajectories_with_neural_activity(self, neural_data, behavioural_data, ax):
        """Plot a single homing trajectory with the neural activity colour coded on the trail for a single neuron"""

        assert len(neural_data) == len(behavioural_data), "The length of the neural data and behavioural data is not the same"

        # Extract positional data for each homing
        # x_loc = tracking_data['head_loc'][onset_frame:offset_frame, 0]
        # y_loc = tracking_data['head_loc'][onset_frame:offset_frame, 1]
        # clu_neural_data = neural_data #

        y_loc = behavioural_data["mouse_y_position"]
        x_loc = behavioural_data["mouse_x_position"]

        length_of_homing = len(neural_data)
        trail_color = np.empty([length_of_homing, 3]) # 3 for RGB

        # For each frame retrieve the colour of the trail based on the neural activity
        for frame in range(length_of_homing):

            # reuse the function and see if it works with neural data
            trail_color[frame, :] = get_color_based_on_neural_activity(
                neural_data=neural_data[frame], 
            )
        
        ax.scatter(x_loc, y_loc, s=5, c=trail_color, alpha = 0.7) # c is a 2d array where each row is an RGB value

        return ax

    def base_plotting(self, ax, tracking, condition):
        """hard code condition for ease"""

        arena_radius = 460
        
        # If there is a shelter present, draw it
        if "shelter_loc" in tracking.keys():
            for i in [0, 1]:
                ax.plot(
                    [tracking["shelter_loc"][0][0], tracking["shelter_loc"][1][0]],
                    [tracking["shelter_loc"][i][1], tracking["shelter_loc"][i][1]],
                    color=[1, 0, 0],
                )
                ax.plot(
                    [tracking["shelter_loc"][i][0], tracking["shelter_loc"][i][0]],
                    [tracking["shelter_loc"][0][1], tracking["shelter_loc"][1][1]],
                    color=[0, 0, 0],
                )

        # If there is a barrier present, draw it
        if not np.logical_or(condition == "shelter_only", condition == "pre_shelter"):
            if len(tracking["barrier_loc"]) > 0:
                if np.logical_or(np.logical_or(condition == "barrier_present", condition == "all_time"), condition == "shelter_present"):
                    # draw old two-sided barrier
                    bar_loc = [tracking["barrier_loc"][0][0], tracking["barrier_loc"][1][0]]

                if condition == "barrier_pre_flip":
                    # draw barrier from first point to the edge
                    if tracking["barrier_loc"][0][0] < 512:
                        bar_loc = [tracking["barrier_loc"][0][0], 512 + arena_radius]
                    else:
                        bar_loc = [512 - arena_radius, tracking["barrier_loc"][0][0]]

                if condition == "barrier_post_flip":
                    # draw barrier from second point to the edge
                    if tracking["barrier_loc"][1][0] < 512:
                        bar_loc = [tracking["barrier_loc"][1][0], 512 + arena_radius]
                    else:
                        bar_loc = [512 - arena_radius, tracking["barrier_loc"][1][0]]

                ax.plot([bar_loc[0], bar_loc[1]], [tracking["barrier_loc"][0][1], tracking["barrier_loc"][1][1]], color=[0, 0, 0])

        # draw arena edge
        a = 512 + (arena_radius * np.cos(np.linspace(0, 2 * np.pi, 150)))
        b = 512 + (arena_radius * np.sin(np.linspace(0, 2 * np.pi, 150)))

        ax.plot(a, b, color=[0, 0, 0])
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
