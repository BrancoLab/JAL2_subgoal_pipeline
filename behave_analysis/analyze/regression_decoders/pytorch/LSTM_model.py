# """
# Interpreting the loss: The more appropriate metric for regression problems is the loss itself as accuracy is a classification metric.
# """

# import keras
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from matplotlib import pyplot as plt
# from keras.metrics import MeanAbsoluteError
# import polars as pl
# import pandas as pd

# from behave_analysis.analyze.decoders.LSTM.helpers import *

# # TODO: Shape assertions
# # # TODO: Improve with cross validation
# # Check if the data is in the right format and move to preprocessing class
# # Check if the functions from the neural decoding script are valid
# # Check if preprocessing steps are valid


# class LSTMRegression(object):

#     """
#     Class for the long short term memory (LSTM) decoder

#     Parameters
#     ----------
#     units: integer, optional, default 400
#         Number of hidden units in each layer

#     dropout: decimal, optional, default 0
#         Proportion of units that get dropped out

#     num_epochs: integer, optional, default 10
#         Number of epochs used for training

#     verbose: binary, optional, default=0
#         Whether to show progress of the fit after each epoch
#     """

#     def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
#         self.units = units
#         self.dropout = dropout
#         self.num_epochs = num_epochs
#         self.verbose = verbose

#     def fit(self, X_train, y_train) -> None:
#         """
#         Train LSTM Decoder

#         Parameters
#         ----------
#         X_train: numpy 3d array of shape [n_samples, n_time_bins, n_neurons]
#             This is the neural data.
#             See example file for an example of how to format the neural data correctly

#         y_train: numpy 2d array of shape [n_samples, features]
#             This is the outputs that are being predicted
#         """

#         model = Sequential()  # Declare model

#         # Add recurrent layer
#         model.add(
#             LSTM(
#                 self.units,
#                 input_shape=(X_train.shape[1], X_train.shape[2]),
#                 dropout=self.dropout,
#                 recurrent_dropout=self.dropout,
#             )
#         )  # Within recurrent layer, include dropout
#         if self.dropout != 0:
#             model.add(
#                 Dropout(self.dropout)
#             )  # Dropout some units (recurrent layer output units)

#         # Add dense connections to output layer
#         model.add(Dense(y_train.shape[1]))

#         # Fit model (and set fitting parameters)
#         model.compile(
#             loss="mse", optimizer="rmsprop", metrics=[MeanAbsoluteError()]
#         )  # Set loss function and optimizer
#         model.fit(
#             X_train, y_train, epochs=self.num_epochs, verbose=self.verbose
#         )  # Fit the model
#         self.model = model

#     def predict(self, X_test) -> np.ndarray:
#         """
#         Predict outcomes using trained LSTM Decoder

#         Parameters
#         ----------
#         X_test: numpy 3d array of shape [n_samples, n_time_bins, n_neurons]
#             This is the neural data being used to predict outputs.

#         Returns
#         -------
#         y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
#             The predicted outputs
#         """

#         y_test_predicted = self.model.predict(X_test)  # Make predictions
#         return y_test_predicted

# def get_R2(y_test, y_test_pred):
#     """
#     Function to get R2

#     Parameters
#     ----------
#     y_test - the true outputs (a matrix of size number of examples x number of outputs)
#     y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

#     Returns
#     -------
#     R2_array: An array of R2s for each output
#     """

#     R2_list = []  # Initialize a list that will contain the R2s for all the outputs
#     y_mean = np.mean(y_test)
#     R2 = 1 - np.sum((y_test_pred - y_test) ** 2) / np.sum((y_test - y_mean) ** 2)
#     R2_list.append(R2)  # Append R2 of this output to the list
#     R2_array = np.array(R2_list)
#     return R2_array  # Return an array of R2s

# def main_new(frame_by_cluster_matrix, y):
#     # How many bins of neural data prior to the output are used for decoding
#     bins_before = 6
#     # Whether to use concurrent time bin of neural data
#     bins_current = 1
#     # How many bins of neural data after the output are used for decoding
#     bins_after = (6)

#     # Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
#     # Function to get the covariate matrix that includes spike history from previous bins
#     X = get_spikes_with_history(frame_by_cluster_matrix, bins_before, bins_after, bins_current)
    
#     #Set what part of data should be part of the training/testing/validation sets
#     training_range=[0, 0.7]
#     testing_range=[0.7, 0.85]
#     valid_range=[0.85,1]
    
#     num_examples=X.shape[0]

#     #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
#     #This makes it so that the different sets don't include overlapping neural data
#     training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
#     testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
#     valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

#     # Transform y to sine and cosine components
#     y_sin = np.sin(np.radians(y))
#     y_cos = np.cos(np.radians(y))
#     y_transformed = np.column_stack((np.sin(np.radians(y)), np.cos(np.radians(y))))


#     #Get training data
#     X_train=X[training_set,:,:]
#     y_train=y_transformed[training_set,:]

#     #Get testing data
#     X_test=X[testing_set,:,:]
#     y_test=y_transformed[testing_set,:]

#     #Get validation data
#     X_valid=X[valid_set,:,:]
#     y_valid=y_transformed[valid_set,:]
    
#     #Z-score "X" inputs. 
#     X_train_mean=np.nanmean(X_train,axis=0)
#     X_train_std=np.nanstd(X_train,axis=0)
#     X_train=(X_train-X_train_mean)/X_train_std
#     X_test=(X_test-X_train_mean)/X_train_std
#     X_valid=(X_valid-X_train_mean)/X_train_std

#     #Zero-center outputs
#     y_train_mean=np.mean(y_train,axis=0)
#     y_train=y_train-y_train_mean
#     y_test=y_test-y_train_mean
#     y_valid=y_valid-y_train_mean

    
#     # Declare model
#     model_lstm = LSTMRegression(units=400, dropout=0, num_epochs=5)

#     # Fit model
#     model_lstm.fit(X_train, y_train)
    
#     #Get predictions
#     y_valid_predicted_lstm=model_lstm.predict(X_valid)
    
#     # Now convert back to angles
#     predicted_angles = np.degrees(np.arctan2(y_valid_predicted_lstm[:, 0], y_valid_predicted_lstm[:, 1]))

#     #Get metric of fit
#     R2s_lstm=get_R2(np.degrees(np.arctan2(y_valid[:, 0], y_valid[:, 1])), predicted_angles)
#     print('R2s:', R2s_lstm)
    
#     # Plotting
#     fig, axs = plt.subplots(1, 1, figsize=(12, 18))
#     axs.plot(np.degrees(np.arctan2(y_test[:, 0], y_test[:, 1])), label="Actual values")
#     axs.plot(predicted_angles, label="Predicted values", linestyle="--")
#     axs.legend()
#     axs.set_xlabel("Sample")
#     axs.set_ylabel("Value")
#     plt.tight_layout()
#     plt.show()
    
# if __name__ == "__main__":
#     runToyExample = (False,)
#     runMultiFeatureToyExample = True

#     ################################################ MULTI-FEATURE TOY EXAMPLE ################################################

#     if runMultiFeatureToyExample:
#         # Generate synthetic data with a pattern
#         n_samples = 1000
#         n_time_bins = 10
#         n_neurons = 8
#         n_outputs = 3  # We now have 3 output features

#         t = np.arange(n_samples * n_time_bins).reshape(n_samples, n_time_bins)
#         phase_shifts = np.random.rand(n_neurons).reshape(1, 1, n_neurons)

#         # Generate a sine wave with different phase shifts for each neuron
#         X = np.sin(0.01 * t[:, :, np.newaxis] + phase_shifts)
#         X += 0.1 * np.random.randn(
#             n_samples, n_time_bins, n_neurons
#         )  # Adding Gaussian noise

#         # y1: predictable from neural activity
#         y1 = X.mean(axis=2).mean(axis=1).reshape(-1, 1) + 0.05 * np.random.randn(
#             n_samples, 1
#         )

#         # y2: somewhat predictable, but with more noise
#         y2 = X.mean(axis=2).mean(axis=1).reshape(-1, 1) + 0.5 * np.random.randn(
#             n_samples, 1
#         )

#         # y3: completely random, not predictable from neural data
#         y3 = np.random.randn(n_samples, 1)

#         # Concatenate the y values to form a matrix with n_samples rows and features columns
#         y = np.concatenate((y1, y2, y3), axis=1)

#         # Instantiate the LSTMRegression class
#         lstm_model = LSTMRegression(units=50, dropout=0.2, num_epochs=50, verbose=1)

#         # Split the data into training and test datasets
#         train_samples = int(0.8 * n_samples)
#         X_train, y_train = X[:train_samples], y[:train_samples]
#         X_test, y_test = X[train_samples:], y[train_samples:]

#         # Fit the training data
#         lstm_model.fit(X_train, y_train)

#         # Predict using the test data
#         predictions = lstm_model.predict(
#             X_test
#         )  # Same shape as y_test [n_samples, features]

#         # Plotting
#         fig, axs = plt.subplots(3, 1, figsize=(12, 18))
#         titles = ["y1: Predictable", "y2: Somewhat Predictable", "y3: Random"]

#         for i in range(n_outputs):
#             axs[i].plot(y_test[:, i], label="Actual values")
#             axs[i].plot(predictions[:, i], label="Predicted values", linestyle="--")
#             axs[i].legend()
#             axs[i].set_title(titles[i])
#             axs[i].set_xlabel("Sample")
#             axs[i].set_ylabel("Value")

#         plt.tight_layout()
#         plt.show()

#     ################################################ TOY EXAMPLE ################################################

#     if runToyExample:
#         # Generate synthetic data with a pattern
#         n_samples = 1000
#         n_time_bins = 10
#         n_neurons = 8
#         n_outputs = 1

#         # Generate a sine wave with added Gaussian noise
#         t = np.arange(n_samples * n_time_bins).reshape(n_samples, n_time_bins)
#         phase_shifts = np.random.rand(n_neurons).reshape(1, 1, n_neurons)

#         # Generate a sine wave with different phase shifts for each neuron
#         X = np.sin(0.01 * t[:, :, np.newaxis] + phase_shifts)

#         X += 0.1 * np.random.randn(
#             n_samples, n_time_bins, n_neurons
#         )  # Adding Gaussian noise

#         # For simplicity, let's make y the mean of the neural activity across all neurons and time bins
#         y = X.mean(axis=2).mean(axis=1).reshape(-1, 1) + 0.05 * np.random.randn(
#             n_samples, 1
#         )  # Adding Gaussian noise

#         # Instantiate the LSTMRegression class
#         lstm_model = LSTMRegression(units=50, dropout=0.2, num_epochs=50, verbose=1)

#         # Split the data into training and test datasets
#         train_samples = int(0.8 * n_samples)
#         X_train, y_train = X[:train_samples], y[:train_samples]
#         X_test, y_test = X[train_samples:], y[train_samples:]

#         # Fit the training data
#         lstm_model.fit(X_train, y_train)

#         # Predict using the test data
#         predictions = lstm_model.predict(X_test)

#         # Plotting
#         plt.figure(figsize=(12, 6))
#         plt.plot(y_test, label="Actual values")
#         plt.plot(predictions, label="Predicted values", linestyle="--")
#         plt.legend()
#         plt.title("Actual vs. Predicted Values")
#         plt.xlabel("Sample")
#         plt.ylabel("Value")
#         plt.show()
