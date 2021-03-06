""" Utility module to help read and write HDF files
"""
from contextlib import closing
import h5py
import numpy as np
import logging
from bf_logging import bf_log


bf_log.setup_logger()
logger = logging.getLogger(__name__)

def save_h5(outfile, dictionary):
    """ Saves passed dictionary to an h5 file

    Parameters
    ----------
    outfile : string
        Name of output h5 file
    dictionary : dictionary
        Dictionary that will be saved

    """
    def save_layer(f, seed, dictionary):

        for key, value in dictionary.items():
            fullKey = f"{seed}/{key}"
            if type(dictionary[key]) == dict:
                f = save_layer(f, fullKey, value)
            else:
                try:
                    f[fullKey] = dictionary[key]
                except:
                    f[fullKey] = dictionary[key].astype(np.string_)

        return f
    logger.info(f'Saving file {outfile}')
    with closing(h5py.File(outfile, 'w')) as f:

        for key, value in dictionary.items():
            if type(dictionary[key]) == dict:
                f = save_layer(f, key, value)
            else:
                try:
                    f[key] = dictionary[key]
                except:
                    f[key] = dictionary[key].astype(np.string_)


def load_h5(feature_file):
    """ Loads h5 contents to dictionary.  Single level dictionary with keys being full h5 paths.

    Parameters
    ----------
    feature_file : string
        Name of input h5 file

    Returns
    -------
    dictionary : dictionary
        Dictionary of h5 contents

    """
    def load_layer(f, seed, dictionary):

        for key in f[seed].keys():
            fullKey = f"{seed}/{key}"
            if isinstance(f[fullKey], h5py.Dataset):
                dictionary[fullKey] = np.asarray(f[fullKey])
            else:
                dictionary = load_layer(f, fullKey, dictionary)

        return dictionary
    logger.info(f"Loading file {feature_file}")
    with h5py.File(feature_file, 'r') as f:

        dictionary = {}
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                dictionary[key] = np.asarray(f[key])
            else:
                dictionary = load_layer(f, key, dictionary)

    return dictionary
