#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:39:21 2019

@author: jackal
"""
import argparse
import glob
import os
import pickle
import sys
from contextlib import closing
from typing import Dict, Union

import numpy as np

from bf_config import bf_globals
from bf_tools import tools_pg
from bf_util import h5_util


def predict_slip(model_file: str,
                 model_features: Union[str, Dict[str, np.ndarray]],
                 features: Dict):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    model_file : str
        [description]
    model_features : Union[str, Dict[str, np.ndarray]]
        [description]
    features : Dict
        [description]

    Returns
    -------
    [type]
        [description]
    """
    with open(model_file, 'rb') as f:
        d = pickle.load(f)

    f = h5_util.load_h5(model_features) \
        if isinstance(model_features,str) else model_features
    featureList = list(f["feature_names"][:])
    featureList = [x.decode('utf-8') for x in featureList]

    fullFeatureList = list(features["feature_names"][:])
    fullFeatureList = [x.decode('utf-8') if not isinstance(x,np.str)
                                         else x
                        for x in fullFeatureList]

    data_list = []
    for feature in featureList:
        idx = fullFeatureList.index(feature)
        data_list.append(features["X"][:,idx])

    data = np.column_stack(data_list)

    return d.predict(data)


def predict_composition(model_file: str,
                        model_features: Union[str, Dict[str, np.ndarray]],
                        features: Dict):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    model_file : str
        [description]
    model_features : Union[str, Dict[str, np.ndarray]]
        [description]
    features : Dict
        [description]

    Returns
    -------
    [type]
        [description]
    """
    with open(model_file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')

    f = h5_util.load_h5(model_features) \
        if isinstance(model_features, str) else model_features
    featureList = list(f["feature_names"][:])
    featureList = [x.decode('utf-8') for x in featureList]

    fullFeatureList = list(features["feature_names"][:])
    fullFeatureList = [x.decode('utf-8') if not isinstance(x, np.str)
                       else x
                       for x in fullFeatureList]

    data_list = []
    for feature in featureList:
        idx = fullFeatureList.index(feature)
        data_list.append(features["X"][:,idx])

    data = np.column_stack(data_list)

    lkhd = d.predict_proba(data)
    idx = np.argmax(lkhd, axis = 1)

    class_dict = dict(tools_pg.composition_dict)

    lkhd_max = np.zeros((len(lkhd, )))
    class_max = np.zeros((len(lkhd, )), dtype = '<U8')
    for i in range(len(lkhd)):
        lkhd_max[i] = lkhd[i, idx[i]]
        class_max[i] = list(class_dict)[idx[i]]

    return np.column_stack((lkhd_max, class_max))


def predict_hydration(model_file: str,
                      model_features: Union[str, Dict[str, np.ndarray]],
                      features: Dict):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    model_file : str
        [description]
    model_features : Union[str, Dict[str, np.ndarray]]
        [description]
    features : Dict
        [description]

    Returns
    -------
    [type]
        [description]
    """
    with open(model_file, 'rb') as f:
        d = pickle.load(f)

    f = h5_util.load_h5(model_features) if isinstance(model_features,str) \
        else model_features
    featureList = list(f["feature_names_hyd"][:])
    featureList = [x.decode('utf-8') for x in featureList]

    fullFeatureList = list(features["feature_names_hyd"][:])
    fullFeatureList = [x.decode('utf-8') if not isinstance(x, np.str)
                       else x
                       for x in fullFeatureList]

    data_list = []
    for feature in featureList:
        idx = fullFeatureList.index(feature)
        data_list.append(features['X_hyd'][:,idx])

    data = np.column_stack(data_list)

    #do not predict around 270 rotational angle, no EIS sensor coverage
    rot_mask = (features['X_hyd'][:, -1] < 270+20) & (features['X_hyd'][:, -1] >  270-20)

    mask_nan = np.isnan(data.sum(axis=1))

    lkhd_max = np.zeros((len(data, )))
    class_max = np.zeros((len(data, )))

    if mask_nan.sum() == len(data.sum(axis=1)):
        lkhd_max[:] = np.nan
        class_max[:] = np.nan
        lkhd_dry = np.zeros((len(data, )))
        lkhd_dry[:] = np.nan

    else:
        lkhd_nonan = d.predict_proba(data[~mask_nan, :])
        lkhd = np.zeros((len(data), lkhd_nonan.shape[1]))
        lkhd[:] = np.nan
        lkhd[~mask_nan] = lkhd_nonan
        idx = np.argmax(lkhd, axis = 1)

        class_dict = {'0.0': 0, '1.0': 1, '3.0': 2, '5.0': 3,
                      '10.0': 4, '15.0': 5}

        lkhd_max = np.zeros((len(lkhd, )))
        class_max = np.zeros((len(lkhd, )))
        for i in range(len(lkhd)):
            lkhd_max[i] = lkhd[i, idx[i]]
            class_max[i] = list(class_dict)[idx[i]]
        lkhd_dry = lkhd[:, 0]

        lkhd_max[rot_mask] = np.nan
        lkhd_dry[rot_mask] = np.nan
        class_max[rot_mask] = np.nan

    return np.column_stack([lkhd_dry, lkhd_max, class_max])


def predict_rock(model_file: str,
                 model_features: Union[str, Dict[str, np.ndarray]],
                 features: Dict):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    model_file : str
        [description]
    model_features : Union[str, Dict[str, np.ndarray]]
        [description]
    features : Dict
        [description]

    Returns
    -------
    [type]
        [description]
    """
    with open(model_file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')

    f = h5_util.load_h5(model_features) \
        if isinstance(model_features, str) else model_features
    featureList = list(f["feature_names"][:])
    featureList = [x.decode('utf-8') for x in featureList]

    fullFeatureList = list(features["feature_names"][:])
    fullFeatureList = [x.decode('utf-8') if not isinstance(x, np.str)
                       else x
                       for x in fullFeatureList]

    data_list = []
    for feature in featureList:
        idx = fullFeatureList.index(feature)
        data_list.append(features["X"][:,idx])

    data = np.column_stack(data_list)

    return d.predict_proba(data)[:, 1]


def predict_surface_patterns(length):
# TODO Is this code going to be used?
#def predict_surface_patterns(datadir, features):
    """ Short description

    Longer description

    Parameters
    ----------

    Returns
    -------

    """

    '''
    infile = datadir + bf_globals.PATTERN_MODEL
    with open(infile, 'rb') as f:
        d = pickle.load(f)

    f = h5py.File(datadir + bf_globals.PATTERN_FEATURE_FILE, 'r')
    featureList = list(f["feature_names"][:])
    featureList = [x.decode('utf-8') for x in featureList]
    f.close()

    fullFeatureList = list(features["feature_names"][:])
    fullFeatureList = [x.decode('utf-8') for x in fullFeatureList]

    data_list = []
    for feature in featureList:
        idx = fullFeatureList.index(feature)
        data_list.append(features["X"][:,idx])

    data = np.column_stack(data_list)
    '''

    '''
    lkhd = d.predict_proba(data)
    idx = np.argmax(lkhd, axis = 1)

    class_dict = tools_pg.terrain_dict

    lkhd_max = np.zeros((len(lkhd, )))
    class_max = np.zeros((len(lkhd, )), dtype = '<U8')
    for i in range(len(lkhd)):
        lkhd_max[i] = lkhd[i, idx[i]]
        class_max[i] = list(class_dict)[idx[i]]
    '''


    '''
    class_dict = dict(tools_pg.terrain_dict)
    vals = np.unique(idx)
    keys = list(class_dict)
    label = idx.copy()
    for i in range(len(vals)):
        label[idx == vals[i]] = keys[i]
    '''
    a = np.zeros(length)
    a[:] = np.nan

    #return np.column_stack([lkhd, idx])
    return a
