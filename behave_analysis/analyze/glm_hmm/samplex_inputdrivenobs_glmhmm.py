"""A function to predict observations based on expected states and fitted weights
This is modified from Lindrman lab ssm package
inputdriven observations"""

import numba
import numpy as np
import autograd.numpy.random as npr

def sample_x(inputs, z, Wk, C):
    """
    z is the states, its shape is
    input is the matrix of inputs, its shape
    Wk is the matrix of fitted weights
    C is the number of categories
    """
    input = inputs[0]
    time_dependent_logits = calculate_logits(input, Wk)  # size TxKxC
    ps = np.exp(time_dependent_logits)
    T = time_dependent_logits.shape[0]
    if T == 1:
        sample = np.array([npr.choice(C, p=ps[t, z]) for t in range(T)])
    elif T > 1:
        sample = np.array([npr.choice(C, p=ps[t, z[t]]) for t in range(T)])
    return sample


def calculate_logits(input, Wk):
    """
    Return array of size TxKxC containing log(pr(yt=C|zt=k))
    :param input: input array of covariates of size TxM
    :return: array of size TxKxC containing log(pr(yt=c|zt=k, ut)) for all c in {1, ..., C} and k in {1, ..., K}
    """
    # Transpose array dimensions, so that array is now of shape ((C-1)xKx(M+1))
    Wk_tranpose = np.transpose(Wk, (1, 0, 2))
    # Stack column of zeros to transform array from size ((C-1)xKx(M+1)) to ((C)xKx(M+1)) and then transform shape back to (KxCx(M+1))
    Wk = np.transpose(np.vstack([Wk_tranpose, np.zeros((1, Wk_tranpose.shape[1], Wk_tranpose.shape[2]))]),
                        (1, 0, 2))
    # Input effect; transpose so that output has dims TxKxC
    time_dependent_logits = np.transpose(np.dot(Wk, input.T), (2, 0, 1)) #Note: this has an unexpected effect when both input (and thus Wk) are empty arrays and returns an array of zeros
    time_dependent_logits = time_dependent_logits - logsumexp(time_dependent_logits, axis=2, keepdims=True)
    return time_dependent_logits


@numba.jit(nopython=True, cache=True)
def logsumexp(x):
    N = x.shape[0]
    # find the max
    m = -np.inf
    for i in range(N):
        m = max(m, x[i])
    # sum the exponentials
    out = 0
    for i in range(N):
        out += np.exp(x[i] - m)
    return m + np.log(out)
