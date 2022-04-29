#OS libaries
import numpy as np

def get_onset_offset(signal, th, clean=True):
    """
        Get onset/offset times when a signal goes below>above and
        above>below a given threshold
        Arguments:
            signal: 1d numpy array
            th: float, threshold
            clean: bool. If true ends before the first start and 
                starts after the last end are removed

        Returns:
            Starts: Indexes of pulse onsets
            Ends: Indexes of pulse offsets
    """
    above = np.zeros_like(signal) # Creates an array of zeros of length signal
    above[signal >= th] = 1 #If the signal is above threshold set to 1
    der = derivative(above) #Create an array of differences 
    starts = np.where(der > 0)[0] #Where does the signal switch from 0 to 1
    ends = np.where(der < 0)[0] #Where does the signal switch from 1 to 0

    #If the signal starts with a pulse add a zero to the start
    if above[0] > 0:
        starts = np.concatenate([[0], starts])

    #If the signal ends at the top of the pulse add the length of the signal
    if above[-1] > 0:
        ends = np.concatenate([ends, [len(signal)]])

    if clean:
        ends = np.array([e for e in ends if e > starts[0]])

        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    if not np.any(starts):
        starts = np.array([0])
    if not np.any(ends):
        ends = np.array([len(signal)])

    return starts, ends

def derivative(X, axis=0, order=1):
    """"
        Takes the derivative of an array X along a given axis
        Arguments:
            X: np.array with data
            axis: int. Axis along which the derivative is to be computed
            order: int. Derivative order
    """
    #Prepend 0 so the index is realigned to prevent off by 1 error
    return np.diff(X, n=order, axis=axis, prepend=0)