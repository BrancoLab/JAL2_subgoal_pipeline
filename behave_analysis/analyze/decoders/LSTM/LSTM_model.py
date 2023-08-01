"""
Interpreting the loss: The more appropriate metric for regression problems is the loss itself as accuracy is a classification metric.
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from matplotlib import pyplot as plt
from keras.metrics import MeanAbsoluteError
import polars as pl
import pandas as pd

# TODO: Shape assertions 
# # TODO: Improve with cross validation

class LSTMRegression(object):

    """
    Class for the long short term memory (LSTM) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10 , verbose=0):
        self.units=units
        self.dropout=dropout
        self.num_epochs=num_epochs
        self.verbose=verbose

    def fit(self, X_train, y_train) -> None:

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples, n_time_bins, n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, features]
            This is the outputs that are being predicted
        """

        model = Sequential() #Declare model
        
        #Add recurrent layer
        model.add(LSTM(self.units, input_shape = (X_train.shape[1] , X_train.shape[2]), dropout = self.dropout, recurrent_dropout = self.dropout)) # Within recurrent layer, include dropout
        if self.dropout!=0: 
            model.add(Dropout(self.dropout)) # Dropout some units (recurrent layer output units)

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='rmsprop', metrics=[MeanAbsoluteError()]) # Set loss function and optimizer
        model.fit(X_train, y_train, epochs = self.num_epochs, verbose=self.verbose) # Fit the model
        self.model=model

    def predict(self,X_test) -> np.ndarray:

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples, n_time_bins, n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) # Make predictions
        return y_test_predicted

def get_spikes_with_history(neural_data, bins_before, bins_after, bins_current = 1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.NaN
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X

def get_R2(y_test, y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

def bin_polars_dataframes(video_data, spike_data):
    
    print("Binning dataframes")
    
    # Define the downsample factor
    sample = 1000000

    # Downsample the dataframe to a set number of rows
    spike_data = spike_data.sample(sample, with_replacement = False, seed = 0)
    
    # Generate the bins
    bins = np.arange(0, spike_data["aligned_spike_times"].max(), 0.05) # 0.1 is 100ms bins
    
    # Create bin labels
    bin_labels = np.arange(len(bins) - 1)
    
    # Pivot polars spike dataframe to have the shape [spike_times, clusters]
    pivoted_spike_df = spike_data.pivot(index = "aligned_spike_times", 
                                        columns = "spike_clusters",
                                        values= "aligned_spike_times")
    
    # Convert to pandas as can't figure out cut in polars
    pivoted_spike_df = pivoted_spike_df.to_pandas()
    pivoted_spike_df["bin_indexs"] = pd.cut(pivoted_spike_df["aligned_spike_times"], bins = bins, labels = bin_labels)
    
    # Remove nans from bins column # TODO: Fix this in a better way ensure no nans potential bug
    pivoted_spike_df_cleaned  = pivoted_spike_df.dropna(subset=["bin_indexs"])
    pivoted_spike_df["bin_indexs"] = pivoted_spike_df_cleaned["bin_indexs"].astype(int)
    
    # Turn back to polars for speed
    polars_cut_pivoted_spike_df = pl.from_pandas(pivoted_spike_df)
    
    # Grouby by bin_indes and count number of non null values
    excluded_columns = {"bin_indexs", "aligned_spike_times"}
    agg_exprs = [pl.col(column).count().alias(column) for column in polars_cut_pivoted_spike_df.columns if column not in excluded_columns]
    grouped_df = polars_cut_pivoted_spike_df.groupby("bin_indexs").agg(agg_exprs).sort("bin_indexs")
    
    # Remove rows where bin_indexs is NaN
    filtered_df = grouped_df.filter(grouped_df["bin_indexs"].is_not_null())

    # Now bin the video data to produce the Y values
    # Add the spike aligned_to_frame to the full polars_cut_pivoted_spike_df
    polars_cut_pivoted_spike_df = polars_cut_pivoted_spike_df.with_columns(spike_data["spike_aligned_to_frame"])
    
    # Join the behavioural data to the neural data
    uber_df = polars_cut_pivoted_spike_df.join(video_data, left_on="spike_aligned_to_frame", right_on="frames")
    
    # Group by bin_indexs and take the mean of each column
    columns_to_bin = ["h_bar_north_a", "h_bar_south_a", "mouse_x_position", "mouse_y_position", "hdir", "hsa"]
    agg_exprs_video = [pl.col(column).mean().alias(f"binned_{column}") for column in columns_to_bin]
    video_binned_df = uber_df.groupby("bin_indexs").agg(agg_exprs_video).sort("bin_indexs")
    
    # Outer join on bin_indexs to align the rows and fill nulls
    aligned_df = filtered_df.join(video_binned_df, on="bin_indexs", how="outer").fill_null(0)
    
    # Split the data back to X and y
    X_columns = [col for col in aligned_df.columns if not col.startswith('binned_')]
    X = aligned_df.select(X_columns)

    y_columns = ['bin_indexs'] + [col for col in aligned_df.columns if col.startswith('binned_')]
    y = aligned_df.select(y_columns)
    y = y.drop("bin_indexs")

    # Ensure X and y have the same number of rows
    assert X.shape[0] == y.shape[0], "Mismatch in row count between X and y!"
    
    print("Finished binning dataframes")

    return X.to_numpy(), y.to_numpy()

def preprocess_data_and_set_up(neural_data: np.ndarray, y: np.ndarray):
    
    print("Preprocessing data...")

    bins_before = 6 # How many bins of neural data prior to the output are used for decoding
    bins_current = 1 # Whether to use concurrent time bin of neural data
    bins_after = 6 # How many bins of neural data after the output are used for decoding
    
    # Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
    # Function to get the covariate matrix that includes spike history from previous bins
    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current) # Where neural data is in the format of ["number of time bins", "number of neurons"]
    
    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.7]
    testing_range=[0.7, 0.85]
    valid_range=[0.85,1]
    
    num_examples=X.shape[0] # As the shape of X is [num_samples, num_bins, num_neurons]
    
    #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
    testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

    #Get training data
    X_train=X[training_set,:,:]
    y_train=y[training_set,:]

    #Get testing data
    X_test=X[testing_set,:,:]
    y_test=y[testing_set,:]

    #Get validation data
    X_valid=X[valid_set,:,:]
    y_valid=y[valid_set,:]

    # We normalize (z_score) the inputs and zero-center the outputs. Parameters for z-scoring (mean/std.) should be determined on the training set only, 
    # and then these z-scoring parameters are also used on the testing and validation sets.
    
    #Z-score "X" inputs. 
    X_train_mean=np.nanmean(X_train,axis=0)
    X_train_std=np.nanstd(X_train,axis=0)
    X_train=(X_train-X_train_mean)/X_train_std
    X_test=(X_test-X_train_mean)/X_train_std
    X_valid=(X_valid-X_train_mean)/X_train_std

    #Zero-center outputs
    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean
    y_valid=y_valid-y_train_mean
    
    print("Finished preprocessing data")
    
    return X_valid, y_valid, X_train, y_train, y_test

def main(X_valid, y_valid, X_train, y_train, y_test):
    
    print("Running main...")
        
    #Declare model
    model_lstm = LSTMRegression(units=400, dropout=0, num_epochs=5)

    #Fit model
    model_lstm.fit(X_train, y_train)

    #Get predictions
    y_valid_predicted_lstm = model_lstm.predict(X_valid) # Shape [bins, features]

    #Get metric of fit
    R2s_lstm = get_R2(y_valid, y_valid_predicted_lstm)
    
    print('R2s:', R2s_lstm)
    
    # Plotting
    n_outputs = 6
    fig, axs = plt.subplots(n_outputs, 1, figsize=(12, 18))
    titles = ['binned_h_bar_north_a', 'binned_h_bar_south_a', 'binned_mouse_x_position', 'binned_mouse_y_position', 'binned_hdir', 'binned_hsa']
    
    for i in range(n_outputs):
        axs[i].plot(y_test[:, i], label="Actual values")
        axs[i].plot(y_valid_predicted_lstm[:, i], label="Predicted values", linestyle='--')
        axs[i].legend()
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Sample")
        axs[i].set_ylabel("Value")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    runToyExample = False,
    runMultiFeatureToyExample = True

    ################################################ MULTI-FEATURE TOY EXAMPLE ################################################
    
    if runMultiFeatureToyExample:
        
        # Generate synthetic data with a pattern
        n_samples = 1000
        n_time_bins = 10
        n_neurons = 8
        n_outputs = 3  # We now have 3 output features

        t = np.arange(n_samples * n_time_bins).reshape(n_samples, n_time_bins)
        phase_shifts = np.random.rand(n_neurons).reshape(1, 1, n_neurons)

        # Generate a sine wave with different phase shifts for each neuron
        X = np.sin(0.01 * t[:,:, np.newaxis] + phase_shifts)
        X += 0.1 * np.random.randn(n_samples, n_time_bins, n_neurons)  # Adding Gaussian noise

        # y1: predictable from neural activity
        y1 = X.mean(axis=2).mean(axis=1).reshape(-1, 1) + 0.05 * np.random.randn(n_samples, 1)
        
        # y2: somewhat predictable, but with more noise
        y2 = X.mean(axis=2).mean(axis=1).reshape(-1, 1) + 0.5 * np.random.randn(n_samples, 1)

        # y3: completely random, not predictable from neural data
        y3 = np.random.randn(n_samples, 1)
        
        # Concatenate the y values to form a matrix with n_samples rows and features columns
        y = np.concatenate((y1, y2, y3), axis=1)

        # Instantiate the LSTMRegression class
        lstm_model = LSTMRegression(units=50, dropout=0.2, num_epochs=50, verbose=1)

        # Split the data into training and test datasets
        train_samples = int(0.8 * n_samples)
        X_train, y_train = X[:train_samples], y[:train_samples]
        X_test, y_test = X[train_samples:], y[train_samples:]

        # Fit the training data
        lstm_model.fit(X_train, y_train)

        # Predict using the test data
        predictions = lstm_model.predict(X_test) # Same shape as y_test [n_samples, features]

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
        titles = ["y1: Predictable", "y2: Somewhat Predictable", "y3: Random"]
        
        for i in range(n_outputs):
            axs[i].plot(y_test[:, i], label="Actual values")
            axs[i].plot(predictions[:, i], label="Predicted values", linestyle='--')
            axs[i].legend()
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("Sample")
            axs[i].set_ylabel("Value")
        
        plt.tight_layout()
        plt.show()

    ################################################ TOY EXAMPLE ################################################
    
    if runToyExample:
        
        # Generate synthetic data with a pattern
        n_samples = 1000
        n_time_bins = 10
        n_neurons = 8
        n_outputs = 1

        # Generate a sine wave with added Gaussian noise
        t = np.arange(n_samples * n_time_bins).reshape(n_samples, n_time_bins)
        phase_shifts = np.random.rand(n_neurons).reshape(1, 1, n_neurons)

        # Generate a sine wave with different phase shifts for each neuron
        X = np.sin(0.01 * t[:,:, np.newaxis] + phase_shifts)

        X += 0.1 * np.random.randn(n_samples, n_time_bins, n_neurons)  # Adding Gaussian noise

        # For simplicity, let's make y the mean of the neural activity across all neurons and time bins
        y = X.mean(axis=2).mean(axis=1).reshape(-1, 1) + 0.05 * np.random.randn(n_samples, 1)  # Adding Gaussian noise

        # Instantiate the LSTMRegression class
        lstm_model = LSTMRegression(units=50, dropout=0.2, num_epochs=50, verbose=1)

        # Split the data into training and test datasets
        train_samples = int(0.8 * n_samples)
        X_train, y_train = X[:train_samples], y[:train_samples]
        X_test, y_test = X[train_samples:], y[train_samples:]

        # Fit the training data
        lstm_model.fit(X_train, y_train)

        # Predict using the test data
        predictions = lstm_model.predict(X_test)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label="Actual values")
        plt.plot(predictions, label="Predicted values", linestyle='--')
        plt.legend()
        plt.title("Actual vs. Predicted Values")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.show()
        