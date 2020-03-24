#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:07:02 2019

@author: marchett
"""


import glob
import importlib
import logging
import os
import sys
from collections import namedtuple
from typing import List

import h5py
import numpy as np

# logger.info(os.getcwd())
# sys.path.append('/home/marchett/code/Barefoot_Rover/')
import compute_data
import compute_features
import train_model
from bf_config import bf_globals
from bf_logging import bf_log

bf_log.setup_logger()
logger = logging.getLogger(__name__)

# sys.path.append('/Users/marchett/Documents/BAREFOOT/Barefoot_Rover/Barefoot_Rover/src/ML/config')
#import rock_config as config
# os.chdir('/Users/marchett/Documents/BAREFOOT/Barefoot_Rover/Barefoot_Rover/src/ML')


def main(datadir, *,
         version: str,
         date: str,
         module: str,
         subfolders: str,
         regex: str,
         input_burnin: List[int],
         model_type: str,
         featuresFile: str):

    logger.info('Running training... ' + module + '.... ' + version)
    # --------
    classdir = datadir + 'models/' + module + '/' + version + '/'
    if not os.path.exists(classdir):
        os.makedirs(classdir)

    #outdir = '/Users/marchett/Documents/BAREFOOT/classifier/slip/TEST/'
    data_file = classdir + '_'.join(['data', module, version, date]) + '.h5'
    outfile_name = '_'.join(['features', module, version, date])

    # run the data
    #Point = namedtuple('Point', 'datadir subfolders outfile regex')
    #args = Point(datadir, subfolders, unified_file, regex)
    # compute_data.main(args)

    logger.info("Computing experiment data")
    compute_data.unify_experiment_data(datadir, subfolders, regex,
                                              data_file, fileList=None)

    # run the features
    # bf_globals.bf_log_reset('compute_features_log.txt')
    #featuresFile = 'full_features.txt'
    logger.info("Generating features")
    compute_features.generate_features(datadir, classdir, data_file, outfile_name, input_burnin,
                                            False, module, True, featuresFile)

    # run classifier/regressor
    logger.info("Training model")
    train_model.train_model(outfile_name, classdir,
                            model_type, module, version, date)


if __name__ == '__main__':
    logger.info(sys.argv[1])
    logger.info(os.getcwd())
    configfile = sys.argv[1]
    sys.path.insert(0, os.getcwd() + '/config')
    config = importlib.import_module(configfile)
    datadir = config.datadir
    if len(sys.argv[2:]) > 0:
        import argparse
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        sys.argv.pop(1)

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--datadir', default='/data/MLIA_active_data/data_barefoot/',
                            help='Data directory.')
        #parser.add_argument('--doctest', help='Doctest trigger', action='store_true')
        args = parser.parse_args()
        main(**vars(args))
    else:
        main(datadir, version=config.version,
             date=config.date,
             module=config.module,
             subfolders=config.subfolders,
             regex=config.regex,
             input_burnin=config.burnin,
             model_type=config.model_type,
             featuresFile=config.featuresFile)
