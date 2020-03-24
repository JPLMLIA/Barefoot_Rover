""" Utitlity containg functions to operate on and/or return numpy arrays.
"""
import logging

import numpy as np
from numpy.linalg import lstsq
from scipy.stats import linregress

from bf_logging import bf_log

bf_log.setup_logger()
logger = logging.getLogger(__name__)

def _running_stats(x, N0, N1, stats_function):
    if N0 < N1:
        error_message = f"N0 must be greater than or equal to N1. N0: {N0}, N1: {N1}"
        logger.error(error_message)
        raise ValueError(error_message)

    moving = np.zeros((len(x), ))
    for i in range(len(x)):
        start = max(0, i-N0)
        end = min(i+N1+1, len(x))
        moving[i] = stats_function(x[start:end])

    return moving

def running_mean(x, N0, N1):
    """ Returns a running MEAN on a time series of values for a
    sliding window of size N0+N1.

    Parameters
    ----------
        x : array
            A vector, time series of data points
        N0 : int
            The number of time steps before
        N1 : int
            The number of time steps after

    Returns
    -------
        moving : ndarry
            The mean-smoothed time series vector of the same length as x

    """
    return _running_stats(x,N0,N1,lambda y : np.nanmean(y))

#--------------
def running_median(x, N0, N1):
    """ Returns a running MEDIAN on a time series of values for a
    sliding window of size N0+N1.

    Parameters
    ----------
        x : array
            A vector, time series of data points
        N0 : int
            The number of time steps before
        N1 : int
            The number of time steps after

    Returns
    -------
        moving : ndarry
            The median-smoothed time series vector of the same length as x
    """

    return _running_stats(x,N0,N1,lambda y : np.nanmedian(y))

#--------------
def running_std(x, N0, N1):
    """ Returns a running STANDARD DEVIATION on a time series of values for a
    sliding window of size N0+N1.

    Parameters
    ----------
        x : array
            A vector of size N, time series of data points
        N0 : int
            The number of time steps before
        N1 : int
            The number of time steps after

    Returns
    -------
        moving : ndarry
            A vector of size N of standard deviations of time series x
    """

    return _running_stats(x,N0,N1,lambda y : np.nanstd(y))




#--------------
def numpyify(indict) -> dict:
    """ Converts dictionary values into numpy arrays.

    Parameters
    ----------
    indict: dictionary
        The value for each key must be an array-like data structure (list, set, numpy array etc.)
        of numbers (int, float, etc.)

    Returns
    -------
    outdict:
        Dictionary with all in which all values are numpy arrays

    """
    return dict((key, np.array(val)) for key,val in indict.items())




#--------------
def outliers_iqr(ys, coef = 10.) -> bool:
    """ Removes outliers from a time series using
    interquartile range and scaling factor

    Parameters
    ----------
    ys : arrya_like
        Vector, time series of data points

    Returns
    -------
        A boolean mask of the same length as the input
        time series with True indicating where the outliers are

    """
    q1, q3 = np.nanpercentile(ys, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * coef)
    upper_bound = q3 + (iqr * coef)
    return (ys < upper_bound) & (ys > lower_bound)
