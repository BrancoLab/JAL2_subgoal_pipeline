"""
Interpreting the loss: The more appropriate metric for regression problems is the loss itself as accuracy is a classification metric.
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from matplotlib import pyplot as plt
from keras.metrics import MeanAbsoluteError

keras_v1=int(keras.__version__[0])<=1

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

    def fit(self, X_train, y_train):

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

        model=Sequential() #Declare model
        
        #Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units, input_shape = (X_train.shape[1], X_train.shape[2]), dropout_W = self.dropout, dropout_U = self.dropout)) #Within recurrent layer, include dropout
        else:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='rmsprop', metrics=[MeanAbsoluteError()]) #Set loss function and optimizer
        if keras_v1:
            model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        else:
            model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

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
        
        # Concatenate the y values to form a matrix with n_samples rows and n_outputs columns
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
        predictions = lstm_model.predict(X_test)

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
        