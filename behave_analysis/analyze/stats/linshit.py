"""
Uses a conservative Linear Shift technique (Harris, Kenneth Arxiv 2021, https://arxiv.org/ftp/arxiv/papers/2012/2012.06862.pdf)
to estimate significance level of a statistical measure. stat_computation_func computes a scalar statistical measure (e.g. R^2)
from the data matrix, X, and the variable, y. A central window of X and y of size, D, is linearly shifted to generate a null distribution
of statistical measures. Significance level is reported relative to this null distribution. The theory is that if two time series data are
truely related, then they should be more so when recorded simulatenously rather than at random points.

Author chain of code: Brandon Benson (IBL) & Guido Meijer (crypto paper).
Benson function: https://int-brain-lab.github.io/iblenv/_modules/brainbox/population/decode.html#sigtest_linshift
Meijer function: https://github.com/guidomeijer/crypto-correlations/blob/master/correlate_crypto_controls.py
Harris code: https://github.com/kdharris101/nonsense-correlations/blob/main/nonsense.ipynb

crypto paper: https://peercommunityjournal.org/item/10.24072/pcjournal.30.pdf
"""

import numpy as np
from scipy.stats import pearsonr
from multiprocessing.pool import Pool
import multiprocessing

class LinearShift:
    def __init__(self, X, y, stat_computation_func, PPool = None, step = 400, size_of_central_chunk=300, step_n = 100):
        """
        X : 2-d array
            Data of size (time,clusters) [NB: @LF I don't think it's = (elements, timetrials)]
        y : 1-d array
            predicted variable of size (timetrials)
            it can be 2-d but then it is assumed that it's of size (elements,time) [it will therefore be transposed for part of this]
        stat_computation_metric : function
            takes arguments (X, y) and returns a scalar statistical measure of how well X decodes y
            this is the "user defined function" referred to in Harris 2021. It is assumed that for
            this statistic a higher scalar value is "better".
        size_of_central_chunk (D) : int
            the window length along the center of y used to compute the statistical measure.
            must have room to shift both right and left: len(y) >= D+2 - I believe this is the
            number of samples for the middle chunk.
        """
        self.D = size_of_central_chunk
        self.step_n = step_n # number of steps you want to take
        self.alpha_thresh = 0.01  # threshold for determining a significant p-value
        self.step = step  # this should be at least 40 (fps)
        self.user_defined_function = stat_computation_func
        self.__check_inputs(X)
        self.T, self.N, self.shifts = self.init_params(X)
        self.real_stat = self.compute_V0_statistic(X, y.T) # the transposed matrix is necessary for LDA!
        self.pseudo_stats = self.parallel_compute_shifted_statistics(X, y.T, self.shifts, PPool)
        self.reject_null, self.alpha, self.M, self.sig_level = self.compute_significance()

    def __check_inputs(self, X):
        """
        Check the user defined arguments are valid
        """
        assert (
            len(X) >= self.D + 2
        ), f"The combination of data size {len(X)} with central chunk size {self.D} is incompatible"
        assert len(X) != 0, "The data provided has no length"

    def init_params(self, X):
        """
        Set up the parameters.
        + T: Total length of data
        + N: The size of the segments on either side of the central chunk
        + shifts: An nd.array of how much to shift the central chunk to compare from the start to the end of the array. Number of shifts
        is len(shifts)
        """
        T = len(X)
        N = int((T - self.D) / 2)
        
        # ensure 0 step is not included 
        shifts = np.arange(-(((self.step_n/2)*self.step)), (((self.step_n/2)*self.step))+1, self.step)   
        shifts = np.delete(shifts,shifts == 0)    

        return T, N, shifts

    def compute_V0_statistic(self, X, y):
        """
        Compute the real statistic for the simulatenously recorded central chunk
        """
        if type(X) == np.ndarray:
            X_filtered = X[self.N : self.T - self.N]
            y_filtered = y[self.N : self.T - self.N]
        else:
            # Filtering rows in Polars
            X_filtered = X.slice(self.N, self.T - 2 * self.N)  # starts from self.N and takes (self.T - 2*self.N) rows
            y_filtered = y.slice(self.N, self.T - 2 * self.N)  # same for y

        return self.user_defined_function(X_filtered, y_filtered.T)

    def parallel_compute_shifted_statistics(self, X, y, shifts, pool):
        """
        Shift the central chunk and compute the user defined statistic on non simulatenously recorded segments of X and y.
        Hold X stationary.
        """

        # prep X matrix
        if type(X) == np.ndarray:
            xFiltered = X[self.N : self.T - self.N]
        else:
            xFiltered = X.slice(self.N, self.T - 2 * self.N)

        # prep Y matrix of shifts
        y_matrix = []
        for i, this_shift in enumerate(shifts):
            if type(X) == np.ndarray:
                # y could be a vector or a matrix
                y_matrix.append(y[int(this_shift + self.N) : int(this_shift + self.T - self.N)])  # transpose y so it is in the shape n x time (where n is the number of variables in y)
            else:
                assert type(X) == np.ndarray, "Your data is in polars - I'm not sure parallel processing can currently handle that"
                # y_matrix[i,:] = y.slice(this_shift + self.N, self.T - 2 * self.N)

        # zip the vars
        args_list = [(xFiltered, y) for y in y_matrix]

        # parallel process
        # Define the number of processes to use
        if pool == None:
            num_processes = multiprocessing.cpu_count()-1  # Adjust as needed
            with Pool(num_processes) as pool:
                pseudo_stats = pool.map(self.parallel_function, args_list)
        else:
            pseudo_stats = pool.mp_pool.map(self.parallel_function, args_list)

        return pseudo_stats

    def parallel_function(self,args):
        X,y = args
        out = self.user_defined_function(X, y.T) # np.array([np.shape(X),np.shape(y.T)])
        return out

    def compute_shifted_statistics(self, X, y, shifts):
        """
        Shift the central chunk and compute the user defined statistic on non simulatenously recorded segments of X and y.
        Hold X stationary.
        """

        pseudo_stats = np.zeros(len(shifts)) # How many pseudo statistics to compute

        for shift_idx in range(len(shifts)):
            s = shifts[shift_idx]  # How much to shift the central chunk by

            if type(X) == np.ndarray:
                xFiltered = X[self.N : self.T - self.N]
                yFiltered = y[s + self.N : s + self.T - self.N]
            else:
                xFiltered = X.slice(self.N, self.T - 2 * self.N)
                yFiltered = y.slice(s + self.N, self.T - 2 * self.N)
            
            pseudo_stats[shift_idx] = self.user_defined_function(xFiltered, yFiltered.T)

        return pseudo_stats

    def compute_significance(self):
        """
        How often is the shifted chance statistic greater than the real statistic. If a stronger statistic means
        higher or lower, does that effect the sign for acceptance, question? As long as "better" means higher stat fine for now.
        + reject_null (bool): Can the null hypothesis be rejected. If True, rejoice
        + alpha (float): p-value, what is the signifcance level
        + M (int): how often is the shifted pseudo statistic greater than the real
        """
        M = np.sum(self.pseudo_stats >= self.real_stat)  # m = sum I(V_S >_ V_0)
        
        # TODO: is this an alternative approach to significance?
        # alpha = M / len(self.pseudo_stats)
        # reject_null = False
        # if M <= alpha*(self.N + 1): # If true, reject the Null hypothesis. Your data is 'probably' significant. Rejoice.
        #     reject_null = True
        
        p_val = M / ((len(self.pseudo_stats) / 2) + 1)  # alpha = M / (N+1)
        reject_null = False
        if p_val < self.alpha_thresh:
            # If true, reject the Null hypothesis. Your data is 'probably' significant. Rejoice.
            # originally: if M <= alpha*((len(self.pseudo_stats)/2) + 1): m <= alpha*(N+1)
            reject_null = True

        signifcance_level = p_val * (self.N + 1)
        return reject_null, p_val, M, signifcance_level


if __name__ == "__main__":

    def compute_pearson_correlation(X, y):
        """
        Compute the Pearson correlation coefficient between X and y.
        Returns the correlation coefficient and the p-value.
        """
        return pearsonr(X.ravel(), y)

    # Set random seed for reproducibility
    # np.random.seed(0)

    real_correlation = True  # Set to False to see the effect of uncorrelated data

    # Generate two synthetic time series datasets
    num_samples = 1000
    noise_level = 0.5

    # First time series: Random noise + trend
    x = np.linspace(0, 10, num_samples)
    X = 0.5 * x + noise_level * np.random.randn(num_samples)

    if real_correlation:
        y2 = X * 0.5 + noise_level * np.random.rand(num_samples)  # Use this to see the effect of correlated data

    elif not real_correlation:
        y2 = noise_level * np.random.randn(num_samples)  # Use this to see the effect of uncorrelated data

    # Create a LinearShift instance using the Pearson correlation coefficient as the user-defined function
    ls = LinearShift(
        X=X.reshape(-1, 1), y=y2, stat_computation_func=compute_pearson_correlation, size_of_central_chunk=300
    )

    # Display results
    print(f"Real Statistic: {ls.real_stat:.4f}")
    print(f"Pseudo Statistics: {ls.pseudo_stats[:10]} ...")  # Display only the first 10 for brevity
    print(f"Reject Null: {ls.reject_null}")
    print(f"Alpha: {ls.alpha:.4f}")
    print(f"M: {ls.M}")
    print(f"Significance Level: {ls.sig_level:.4f}")
