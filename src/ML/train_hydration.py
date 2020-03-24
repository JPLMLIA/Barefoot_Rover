#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:22:54 2019

@author: marchett
"""
import argparse
import glob
import itertools
import logging
import os
import pickle
import sys
from contextlib import closing

import h5py

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as colormap
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, SVR

from typing import List

from bf_config import bf_globals
from bf_logging import bf_log
from bf_plot import plot_EIS
from bf_tools import tools_eis
from bf_util import numpy_util, trig_util, h5_util

bf_log.setup_logger()
logger = logging.getLogger(__name__)

def train_hydration_model(datadir: str, regex: str, subfolders: List[str],
                          version: str, date: str):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    datadir : str
        [description]
    regex : str
        [description]
    subfolders : List[str]
        [description]
    version : str
        [description]
    date : str
        [description]
    """
    classdir = f"{datadir}/models/hydration/{version}/"

    if not os.path.exists(classdir):
        os.makedirs(f'{classdir}plots/')

    # TODO Not neede.  Set as defauly
    # regex = ['*_Steel*/*_Y_[0-1]*']

    files = [glob.glob(f'{datadir}/{a}/{b}/') for a, b in zip(subfolders, regex)]
    files = np.sort(np.hstack(files))
    if not len(files):
        error_msg = f"No files found for training in {datadir}."
        logger.error(error_msg)
        sys.exit(error_msg)

    EISdata = {}
    for i in range(0, len(files)):
        path = files[i]
        fname = path.split('/')[-2].split('_')
        ftype = '_'.join(fname)
        logger.debug(f"ftype: {ftype}")

        logger.debug(f'Reading files in {path}')
        files_eis = glob.glob(f'{path}/**/*.idf', recursive=True)
        logger.info(f"Number of EIS readings: {len(files_eis)}")

        EISdata[ftype] = {'RE':[],'IM':[],'freq':[], 'hyd':[], 'mat':[], 'rot': []}
        time_eis = []
        for idffile in files_eis:
            logger.debug(f"Processing {idffile}")
            data = tools_eis.parse_idf(idffile)
            if data['timesteps'] == '0':
                continue
            time_eis.append(data['exp_time'])

            EISdata[ftype]['RE'  ].append(data[     'real_primary'])
            EISdata[ftype]['IM'  ].append(data['imaginary_primary'])
            EISdata[ftype]['freq'].append(data['frequency_primary'])

            hyd = idffile.split('/')[-2].split('_')[1]
            mat = idffile.split('/')[-2].split('_')[0]
            rot = idffile.split('/')[-2].split('_')[4]
            EISdata[ftype]['hyd'].append(float(hyd))
            EISdata[ftype]['mat'].append(mat)
            EISdata[ftype]['rot'].append(int(rot))

        if len(time_eis) == 0:
            del EISdata[ftype]
            logger.info('No plots created, problem with EIS readings')
            continue

        EISdata[ftype] = trig_util.add_polar(EISdata[ftype])
        EISdata[ftype] = numpy_util.numpyify(EISdata[ftype])


    features = {}
    for ftype in EISdata.keys():
        for x in EISdata[ftype].keys():
            features.setdefault(x, []).append(EISdata[ftype][x][:])
    features = {k: np.concatenate(features[k]) for k in features.keys()}

    #save feature file
    outputfile = f'data_hydration_{version}_{date}.h5'
    h5_util.save_h5(f'{classdir}/{outputfile}',features)


    #multiclass classifier on a reduced set of amps and phases
    #hard-coded frequencies here
    sub_freqs = [10, 100, 1000, 10000]
    sub_freqs = [48, 554.863, 4725.18]

    n_seeds = 20
    np.random.seed(63468)
    seeds = np.random.choice(np.arange(0, 100000), n_seeds)

    FREQ_LIMIT_amp = 1
    FREQ_LIMIT_phase = 1
    mask_sane_phase = features['freq'][0, :] >= FREQ_LIMIT_phase
    mask_sane_amp = features['freq'][0, :] >= FREQ_LIMIT_amp
    mask_rot = np.in1d(features['rot'], [270])
    mask_hyd = np.in1d(features['hyd'], [])
    mask_bad = mask_rot | mask_hyd

    #only 4 frequencies
    mask_sane_amp = [(np.abs(features['freq'][0, :] - d)).argmin() for d in sub_freqs]
    mask_sane_phase = [(np.abs(features['freq'][0, :] - d)).argmin() for d in sub_freqs]

    X = np.column_stack([features['amp'][~mask_bad][:, mask_sane_amp],
                         features['phase'][~mask_bad, :][:, mask_sane_phase],
                         features['rot'][~mask_bad]])
    y = features['hyd'][~mask_bad]

    nF = X.shape[1]
    nS = X.shape[0]
    nC = len(np.unique(y))
    amp_names = ['amplitude_' + str(x) for x in features['freq'][0][mask_sane_amp]]
    phase_names = ['phase_' + str(x) for x in features['freq'][0][mask_sane_phase]]
    feature_names = amp_names + phase_names + ['rot_angle']

    #save final feature file
    outputfile = f'features_hydration_{version}_{date}.h5'
    hyd_dict = {'feature_names_hyd': np.hstack(feature_names).astype(np.string_),
                 'X': X,
                 'y': y}
    h5_util.save_h5(f'{classdir}/{outputfile}', hyd_dict)

    yhat = np.empty((nS, n_seeds))
    yhat[:] = np.nan
    probs = [None] * n_seeds
    imp = np.empty((nF, n_seeds))
    for s in range(n_seeds):
        probs[s] = np.zeros((len(X), len(np.unique(y))))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state = seeds[s])
        indices = skf.split(X, y)
        imp_fold = np.zeros((nF, 0))
        for idx_train, idx_test in indices:
            y_train = y[idx_train]
            X_train = X[idx_train, :]

            y_test = y[idx_test]
            X_test = X[idx_test, :]

            rf = RandomForestClassifier(n_estimators = 20, min_samples_leaf = 1,
                                        max_features = 0.33, random_state = seeds[s])

            rf.fit(X_train, y_train)
            yhat[idx_test, s] = rf.predict(X_test)
            probs[s][idx_test, :] = rf.predict_proba(X_test)
            imp_fold = np.column_stack([imp_fold, rf.feature_importances_])

        imp[:, s] = imp_fold.mean(axis = 1)

        #PICKLE THE TRAINED MODEL
        model_file = f"{classdir}{'_'.join(['trained_RF', 'hydration', version, date])}"
        with open(model_file, 'ab') as model:
            _ = pickle.dump(rf, model)

    #results
    hyd_levels = np.unique(y)
    prob_mean = np.array(probs).mean(axis = 0)
    y_pred_idx = np.argmax(prob_mean, axis = 1)
    y_pred = np.empty(y_pred_idx.shape)
    y_pred[:] = np.nan
    for i in range(len(hyd_levels)):
        y_pred[y_pred_idx == i] = hyd_levels[i]
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)


    fig = plot_EIS.plot_confusion_matrix(cm, hyd_levels, normalize=True)
    plt.savefig(f'{classdir}plots/cm_{version}.png')

    mean_imp = imp.mean(axis = 1)
    std_imp = imp.std(axis = 1)
    colors = colormap.Greens(mean_imp / mean_imp.max())
    fig = plt.figure(figsize = (12, 6.5))
    plt.bar(np.arange(1, nF+1), mean_imp / mean_imp.max(), yerr = std_imp, align = 'center',
                 color = colors, alpha=0.5,label = feature_names, error_kw=dict(ecolor='gray'))

    plt.xticks(np.arange(1, nF+1), np.hstack(feature_names), rotation = 90)
    plt.title(f'RF regression relative feature importances w/{str(n_seeds)} runs')
    plt.tight_layout()
    plt.savefig(f'{classdir}plots/var_imp_{version}.png')
    plt.close()


    #boxplots of likelihoods per class
    hyd = np.unique(features['hyd'][~mask_bad])
    hyd_col_dict = plot_EIS.set_colors(hyd)
    plot_data = []
    prob_level= []
    x_color = []
    logger.debug(f"hyd_col_dict: {hyd_col_dict}")
    for h in range(len(hyd)):
        mask_hyd = np.in1d(features['hyd'][~mask_bad], [hyd[h]])
        prob_pred = prob_mean[mask_hyd, h]
        plot_data.append(prob_pred)
        prob_level.append(prob_pred.mean())
        x_color.append(hyd_col_dict[str(hyd_levels[h])])

    plt.figure()
    bp = plt.boxplot(plot_data, labels = [f'{str(x)}\n(+{str(y)})'
                    for x,y in zip(hyd_levels, np.round(prob_level, 2))],
                                     patch_artist = True)

    for b, c in zip(bp['boxes'], x_color):
        b.set_edgecolor(c)
        b.set_facecolor(c)
        b.set_alpha(0.2)

    plt.axhline(y=0.5, alpha = 0.5, ls = '--')
    plt.xlabel('hydration\n (mean lkhd)')
    plt.ylabel('likelihood of hydration level')
    plt.minorticks_on()
    plt.grid(True, alpha = 0.25)
    plt.title('likelihoods per hydration levels')
    plt.tight_layout()
    plt.savefig(f'{classdir}plots/boxplots_lkhd_{version}.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir',        default="/Volumes/MLIA_active_data/data_barefoot/",
                                            help="Path to data repository. Defaults to /Volumes/MLIA_active_data/data_barefoot/")

    parser.add_argument('--regex',          default="*_Steel*/*_Y_[0-1]*",
                                            nargs='+',
                                            help="list of regex strings. Ex: --regex '*'")

    parser.add_argument('--subfolders',     default="/models/Hydration/EIS_Trials/",
                                            nargs='+',
                                            help="Options include: rock_detection, composition, data_andrew")

    parser.add_argument('--version',        help="Version identifier for output model. Ex: v1")

    parser.add_argument('--date',           help="Date to tag hydration model with. Ex: 04162019")

    args = parser.parse_args()

    train_hydration_model(args.datadir, args.regex, args.subfolders, args.version, args.date)
