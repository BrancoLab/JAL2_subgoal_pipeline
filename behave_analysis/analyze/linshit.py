"""
Author chain of code: Brandon Benson & Guido Meijer.

Uses a conservative Linear Shift technique (Harris, Kenneth Arxiv 2021, https://arxiv.org/ftp/arxiv/papers/2012/2012.06862.pdf)
to estimate significance level of a statistical measure. stat_computation_func computes a scalar statistical measure (e.g. R^2)
from the data matrix, X, and the variable, y. A central window of X and y of size, D, is linearly shifted to generate a null distribution
of statistical measures. Significance level is reported relative to this null distribution. The theory is that if two time series data are
truely related, then they should be more so when recorded simulatenously rather than at random points.
"""

import numpy as np
from scipy.stats import pearsonr

class LinearShift:
    def __init__(self, X, y, stat_computation_func, size_of_central_chunk=300):
        """
        X : 2-d array
            Data of size (elements, timetrials)
        y : 1-d array
            predicted variable of size (timetrials)
        stat_computation_metric : function
            takes arguments (X, y) and returns a scalar statistical measure of how well X decodes y
            this is the "user defined function" referred to in Harris 2021. It is assumed that for
            this statistic a higher scalar value is "better".
        D : int
            the window length along the center of y used to compute the statistical measure.
            must have room to shift both right and left: len(y) >= D+2 - I believe this is the
            number of samples for the middle chunk.
        """
        # print("starting shift")
        self.D = size_of_central_chunk
        self.X = X
        self.y = y
        self.user_defined_function = stat_computation_func
        self.__check_inputs()
        self.T, self.N, self.shifts = self.init_params()
        self.real_stat = self.compute_V0_statistic()
        self.pseudo_stats = self.compute_shifted_statistics()
        self.reject_null, self.alpha, self.M, self.sig_level = self.compute_significance()
        # print("done with shift")

    def __check_inputs(self):
        """
        Check the user defined arguments are valid
        """
        assert len(self.y) >= self.D + 2, f"The combination of data size {len(self.y)} with central chunk size {self.D} is incompatible"
        assert len(self.y) != 0, "The data provided has no length"

    def init_params(self):
        """
        Set up the parameters.
        + T: Total length of data
        + N: The size of the segments on either side of the central chunk
        + shifts: An nd.array of how much to shift the central chunk to compare from the start to the end of the array. Number of shifts
        is len(shifts)
        """
        T = len(self.y)
        N = int((T - self.D) / 2)
        shifts = np.arange(-N, N - 400, 400) # 10 second shifts as camera is 40 Hz
        # shifts = np.arange(-N, N - 400, 40) # 1 second shifts as camera is 40 Hz

        return T, N, shifts

    def compute_V0_statistic(self):
        """
        Compute the real statistic for the simulatenously recorded central chunk
        """
        # return self.user_defined_function(self.X[self.N : self.T - self.N], self.y[self.N:self.T - self.N])[0]
        # Filtering rows in Polars
        # X_filtered = self.X.slice(self.N, self.T - 2*self.N)  # starts from self.N and takes (self.T - 2*self.N) rows
        # y_filtered = self.y.slice(self.N, self.T - 2*self.N)  # same for y
        
        X_filtered = self.X[self.N : self.T - self.N] # Take out central chunk
        y_filtered = self.y[self.N : self.T - self.N] # Take out central chunk
        
        # X_filtered = self.X[: self.N] # Take out last chunk
        # y_filtered = self.y[ : self.N] # Take out last chunk

        return self.user_defined_function(X_filtered, y_filtered)

    def compute_shifted_statistics(self):
        """
        Shift the central chunk and compute the user defined statistic on non simulatenously recorded segments of X and y.
        Hold X stationary.
        """
        
        pseudoStats = np.zeros(len(self.shifts)) # How many pseudo statistics to compute
        for shift_idx in range(len(self.shifts)):
            s = self.shifts[shift_idx] # How much to shift the central chunk by
            # print(f"shift {shift_idx} of {len(self.shifts)}")
            
            # Remove central chunk
            if s == 0:
                continue
            
            xFiltered = self.X.slice(self.N, self.T - 2*self.N)
            yFiltered = self.y.slice(s + self.N, self.T - 2*self.N)
            pseudoStats[shift_idx] = self.user_defined_function(xFiltered, yFiltered)

        return pseudoStats

    def compute_significance(self):
        """
        How often is the shifted chance statistic greater than the real statistic. If a stronger statistic means
        higher or lower, does that effect the sign for acceptance, question? As long as "better" means higher stat fine for now.
        + reject_null (bool): Can the null hypothesis be rejected. If True, rejoice
        + alpha (float): what is the signifcance level
        + M (int): how often is the shifted pseudo statistic greater than the real
        """
        M = np.sum(self.pseudo_stats >= self.real_stat) # m = sum I(V_S >_ V_0)
        alpha = M / len(self.pseudo_stats)
        reject_null = False
        if M <= alpha*(self.N + 1): # If true, reject the Null hypothesis. Your data is 'probably' significant. Rejoice.
            reject_null = True
            
        signifcance_level = alpha*(self.N + 1)    
        return reject_null, alpha, M, signifcance_level

if __name__ == "__main__":

    def compute_pearson_correlation(X, y):
        """
        Compute the Pearson correlation coefficient between X and y.
        Returns the correlation coefficient and the p-value.
        """
        return pearsonr(X.ravel(), y)

    # Set random seed for reproducibility
    #np.random.seed(0)
    
    real_correlation = True # Set to False to see the effect of uncorrelated data

    # Generate two synthetic time series datasets
    num_samples = 1000
    noise_level = 0.5

    # First time series: Random noise + trend
    x = np.linspace(0, 10, num_samples)
    X = 0.5 * x + noise_level * np.random.randn(num_samples)

    if real_correlation:
        y2 = X * 0.5 + noise_level * np.random.rand(num_samples) # Use this to see the effect of correlated data
    
    elif not real_correlation:
        y2 = noise_level * np.random.randn(num_samples) #Use this to see the effect of uncorrelated data
    
    # Create a LinearShift instance using the Pearson correlation coefficient as the user-defined function
    ls = LinearShift(X = X.reshape(-1, 1), 
                     y = y2, 
                     stat_computation_func = compute_pearson_correlation, 
                     size_of_central_chunk = 300)

    # Display results
    print(f"Real Statistic: {ls.real_stat:.4f}")
    print(f"Pseudo Statistics: {ls.pseudo_stats[:10]} ...")  # Display only the first 10 for brevity
    print(f"Reject Null: {ls.reject_null}")
    print(f"Alpha: {ls.alpha:.4f}")
    print(f"M: {ls.M}")
    print(f"Significance Level: {ls.sig_level:.4f}")
