# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:38:37 2018

@author: marchett, jackal
"""
from bf_logging import bf_log
from bf_config import bf_globals
from collections import defaultdict
import timeit
import os
import glob
import sys
import h5py
import pywt
import copy
import argparse
import multiprocessing
import logging
from tqdm import tqdm

import numpy as np

from bf_tools import tools_pg, tools_eis
from bf_plot import plots_pg
from bf_util import numpy_util, h5_util

from skimage.feature import peak_local_max
from sklearn.neighbors import KDTree
from contextlib import closing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy import ndimage
from scipy.signal import argrelextrema, find_peaks_cwt
from sklearn.neighbors.kde import KernelDensity
from typing import List

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib as mp
from matplotlib import pyplot as plt
plt.set_cmap('afmhot')


bf_log.setup_logger()
logger = logging.getLogger(__name__)

# TODO Put these in config files
computed_dense_feature_names = ['X', 'X_hyd', 'ambiguous', 'feature_names',
                                'feature_names_hyd', 'files', 'hydration',
                                'imu_contact_bin', 'material', 'nF',
                                'patterns', 'rock', 'rock_depth', 'sink',
                                'slip', 'time_to_compute', 'y']
computed_sparse_feature_names = ['ambiguous', 'features',
                                'features_hyd', 'files', 'hydration',
                                'imu_contact_bin', 'material', 'names',
                                'patterns', 'rock', 'rock_depth', 'sink',
                                'slip', 'time_to_compute', 'exp_path']

def make_features(args):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    args : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Decompose input arguments dict into easier to handle local variables
    data_binned = args["data_binned"]
    ftype = args["ftype"]
    i = args["i"]
    files = args["files"]
    exp_path = args["exp_path"]
    datadir = args['datadir']
    feature_list = args['features']
    burn_in = [args["low_burn"], args["high_burn"]]


    logger.debug(f"datadir: {datadir}")
    logger.debug(f'exp_path: {exp_path}')

    pglag = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    stride = 1
    tlag = 200

    contact_data = tools_pg.contact_area_run(
        data_binned['pg'], data_binned['imu_bin'])

    nt = data_binned['pg'].shape[0]
    lower_t = nt-1 if burn_in[0] >= nt else burn_in[0]

    t_seq = np.arange(lower_t, nt, stride)

    # account for flat areas of rock runs for surface patterns
    pattern = data_binned['pattern'][t_seq]
    rock = data_binned['rock'][t_seq]
    if np.in1d(pattern, ['rock-below', 'bedrock', 'rock-above']).sum() > 1:
        mask_rock = rock == 0
        pattern[mask_rock] = 'flatlvl'

    file_names = np.repeat(files[i], nt)[t_seq]
    imu_contact_bin = data_binned['imu_bin'][t_seq]
    mask_ambiguous = np.in1d(
        data_binned['imu_bin'], tools_pg.contact_area_indexes['ambiguous'])[t_seq]

    logging.info(f"{i} of {len(files)} {ftype}")
    logging.info(f"Data points read: {len(data_binned['slip'])}")

    features = defaultdict(list)
    features_hydration = defaultdict(list)
    contactAreaEndTime = 0
    contactAreaLaggedEndTime = 0
    IMUunwrapEndTime = 0
    IMUunwrapImageGeoEndTime = 0
    sinkEndTime = 0
    contactAreaTsfreshEndTime = 0
    hydrationEndTime = 0
    logger.info(f"Creating features for {len(t_seq)} time steps")
    for t in tqdm(t_seq):
        pg_binned = data_binned['pg']
        imu_binned = data_binned['imu_bin']
        time_binned = data_binned['time']
        # -----
        # stats for contact, scaled, ratio, max dist
        contactAreaStartTime = timeit.default_timer()
        features_subset_contact = tools_pg.contactAreaFeatures(
            t, pg_binned, imu_binned, contact_data, feature_list)
        contactAreaEndTime += timeit.default_timer() - contactAreaStartTime
        for k in features_subset_contact.keys():
            features[k].append(features_subset_contact[k])
        # ratio of grouser vs non-grouser contact for a certain time interval
        # ambiguous rows are excluded
        contactAreaLaggedStartTime = timeit.default_timer()
        features_subset_contact_lagged = tools_pg.contactAreaLaggedFeatures(
            t, imu_binned, contact_data, feature_list, tlag)
        contactAreaLaggedEndTime += timeit.default_timer() - contactAreaLaggedStartTime
        for k in features_subset_contact_lagged.keys():
            features[k].append(features_subset_contact_lagged[k])
        # ------
        # area for IMU unwrapped data, normalized by max of all lags
        # ambiguous time steps are excluded
        IMUunwrapStartTime = timeit.default_timer()
        features_subset_unwrap = tools_pg.IMUunwrapFeatures(t, pg_binned,
                                                            imu_binned,
                                                            time_binned,
                                                            contact_data,
                                                            pglag,
                                                            feature_list,
                                                            tlag)
        IMUunwrapEndTime += timeit.default_timer() - IMUunwrapStartTime
        for k in features_subset_unwrap.keys():
            features[k].append(features_subset_unwrap[k])
        # ------
        # geometric features of IMU unwrap objects
        # ambiguous time steps are excluded
        IMUunwrapImageGeoStartTime = timeit.default_timer()
        features_subset_unwrap_geo = tools_pg.IMUunwrapImageGeoFeatures(
            t, pg_binned, imu_binned, time_binned, data_binned['imu'],
             contact_data, pglag, feature_list, tlag)
        IMUunwrapImageGeoEndTime += timeit.default_timer() - IMUunwrapImageGeoStartTime
        for k in features_subset_unwrap_geo.keys():
            features[k].append(features_subset_unwrap_geo[k])
        # ------
        # add sink angle derived value
        sinkStartTime = timeit.default_timer()
        features_subset_sink = tools_pg.sinkFeatures(
            t, pg_binned, imu_binned, contact_data, feature_list, tlag)
        sinkEndTime += timeit.default_timer() - sinkStartTime
        for k in features_subset_sink.keys():
            features[k].append(features_subset_sink[k])
        # ------
        # add collected hydration features for all non-hydration models
        hydrationStartTime = timeit.default_timer()
        features_subset_hyd = tools_eis.collect_hyd_features(
            t, data_binned['freq'],data_binned['amp'],data_binned['phase'],
            data_binned['imu'], contact_data, feature_list, tlag)
        hydrationEndTime += timeit.default_timer() - hydrationStartTime
        for k in features_subset_hyd.keys():
            features[k].append(features_subset_hyd[k])

        # PLEASE DO NOT REMOVE THESE FEATURES, WILL BE USED IN THE FUTURE
        '''
        #------
        #contact features from the tsfresh package
        contactAreaTsfreshStartTime = timeit.default_timer()
        features_subset_tsfresh = tools_pg.contact_area_tsfresh_features(t, data_binned, contact_data, feature_list)
        contactAreaTsfreshEndTime += timeit.default_timer() - contactAreaTsfreshStartTime
        for k in features_subset_tsfresh.keys():
            features[k].append(features_subset_tsfresh[k])
        '''

        # ------
        # add collected hydration features for the hydration model only
        for k in features_subset_hyd.keys():
            features_hydration[k].append(features_subset_hyd[k])

    if args['plot']:
        # Determine plot directory, check if it exists, make it if it doesn't.
        plotdir = f"{datadir}{exp_path.decode('utf-8')}plots/features/"
        logging.info(f"Placing plots in: {plotdir}")
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

        # contact area time series with smoothed
        fig = plots_pg.contactAreaSmoothPlot(ftype, data_binned, contact_data)
        plt.savefig(f"{plotdir}/run_mean_contact_{ftype}.png")
        plt.close()

        # kernel density of histograms
        fig = plots_pg.contactAreaHistPlot(ftype, data_binned, contact_data)
        plt.savefig(f"{plotdir}/histogram_contact_{ftype}.png")
        plt.close()

        # imu wraps
        fig = plots_pg.IMUunwrapRowMaxPlot(data_binned, pglag, ftype)
        plt.savefig(f"{plotdir}/imurows_{ftype}.png")
        plt.close()

        # low pass filter
        fig = plots_pg.contactAreaLowFiltPlot(ftype, data_binned, contact_data)
        plt.savefig(f"{plotdir}/lowpass_{ftype}.png")
        plt.close()

    total_feature_compute_time = contactAreaEndTime + contactAreaLaggedEndTime + \
        IMUunwrapEndTime + IMUunwrapImageGeoEndTime + sinkEndTime + hydrationEndTime

    logging.info("Time to compute {i} contact area features: {time:.1f}".format(
        i=len(features_subset_contact), time=contactAreaEndTime))
    logging.info("Time to compute {i} contact area lagged features: {time:.1f}".format(
        i=len(features_subset_contact_lagged), time=contactAreaLaggedEndTime))
    logging.info("Time to compute {i} IMU unwrapped features: {time:.1f}".format(
        i=len(features_subset_unwrap), time=IMUunwrapEndTime))
    logging.info("Time to compute {i} IMU unwrapped geometric features: {time:.1f}".format(
        i=len(features_subset_unwrap_geo), time=IMUunwrapImageGeoEndTime))
    logging.info("Time to compute {i} sinkage features: {time:.1f}".format(
        i=len(features_subset_sink), time=sinkEndTime))
    logging.info("Time to compute {i} hydration features: {time:.1f}".format(
        i=len(features_subset_hyd), time=hydrationEndTime))
    logging.info("Time to compute {i} features: {time:.1f}".format(
        i=len(features), time=total_feature_compute_time))
    #logging.info("Time to compute tsfresh features: {time:.1f}".format(time=contactAreaTsfreshEndTime))
    logging.info(f"Features done... {ftype}")

    slip = data_binned['slip'][t_seq]
    sink = data_binned['sink'][t_seq]
    rock_depth = data_binned['rock_depth'][t_seq]
    labels = {'slip': slip, "sink": sink, "rock": rock, "rock_depth": rock_depth,
              "imu_contact_bin": imu_contact_bin, "mask_ambiguous": mask_ambiguous,
              'material': data_binned['material'][t_seq],
              'patterns': data_binned['pattern'][t_seq],
              'hydration': data_binned['hyd'][t_seq]}

    other = {'file_names': file_names, "ftype": ftype, "exp_path": exp_path}
    logging.info(f"Number of features extracted: {len(features)}")

    return {"features": features, 'features_hyd': features_hydration,
            "labels": labels, "other": other,
            "time_to_compute": total_feature_compute_time}


def generate_features(datadir: str, output_folder: str, data_file: str,
                      outfile_name: str, input_burnin: List[str],
                      plot: bool, module: str, multithreading: bool,
                      featuresFile: str):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    datadir : str
        Path to data repository\n
    output_folder : str
        Full path, and name, of the resulting sparse feature data file\n
    data_file : str
        Full path to unified data H5 created using compute_data.py\n
    outfile_name : str
        Full path, and name, of the resulting sparse feature data file\n
    input_burnin : list
        Two values indicating low-end burn in and high-end burn in.\n
    plot : bool
        Generate feature plots.\n
    module : str
        Barefoot classifier type.  Ex: rock, patterns, hydrations\n
    multithreading : bool
        Use multithreading or not
    featuresFile : str
        Full path to file containing list of features to compute.  One per line.

    Returns
    -------
    [type]
        [description]
    """

    if data_file is None or not os.path.exists(data_file):
        error_msg = "Data file does not exist.  Point to valid unified H5 data file to continue"
        logging.error(error_msg)
        sys.exit(error_msg)

    try:
        low_burn = int(input_burnin[0])
        logging.info(f"Leading burn in set to {low_burn}")
    except:
        low_burn = 0
        logging.warning(
            f"Leading burn value of {input_burnin[0]} couldn't be parsed.  Set to default of 0")

    try:
        high_burn = int(input_burnin[1])
        logging.info(f"Trailing burn in set to {high_burn}")
    except:
        high_burn = None
        logging.warning(
            f"Leading burn value of {input_burnin[1]} couldn't be parsed.  Set to default of None")

    if not os.path.exists(output_folder):
        logging.info(f"Output directory does not exist. Creating {output_folder}")
        os.makedirs(output_folder)

    dat = h5py.File(data_file,'r')
    files = dat.keys()
    files = list(files)
    logger.debug(f'{len(files)} data files found')

    logger.debug(f'Reading features from {featuresFile}')
    with open(featuresFile) as f:
        features = f.read().splitlines()

    # Set up list of argument dicts to be passed into make_features.
    #   Needed since input arguments >1
    experiments = []
    for i, ftype in enumerate(files):
        data_binned = {k: dat[ftype][k][:] for k in dat[ftype].keys()}
        exp_path = dat[ftype]["exp_path"][:][0]
        experiments.append({"i": i,
                            "ftype": ftype,
                            "files": files,
                            "data_binned": data_binned,
                            "low_burn": low_burn,
                            "high_burn": high_burn,
                            "exp_path": exp_path,
                            "datadir": datadir,
                            "plot": plot,
                            'module': module,
                            'features': features})
    dat.close()
    if multithreading:
        logging.info("Performing multithreaded featurization")

        # Multithreaded launch of experiment featurization, returns list of result dicts
        with multiprocessing.Pool(processes=bf_globals.THREADS) as pool:
            results = pool.map(make_features, experiments)

    else:
        logging.info("Performing single threaded featurization")
        # single threaded implementation.
        results = [make_features(exp) for exp in experiments]

    # Generates sparse dictionary and sparse H5 of name convention:
    sparse_dictionary = {}
    for result in results:

        features = result["features"]
        features_hydration = result["features_hyd"]
        labels = result["labels"]
        other = result["other"]
        ftype = other["ftype"]
        time_to_compute = result["time_to_compute"]

        sparse_dictionary[ftype] = {}
        fnames = np.asarray(list(features.keys())).astype(np.string_)
        sparse_dictionary[ftype]['names'] = fnames

        sparse_dictionary[ftype]['ambiguous'] = labels['mask_ambiguous']
        sparse_dictionary[ftype]['slip'] = labels['slip']
        sparse_dictionary[ftype]['sink'] = labels['sink']
        sparse_dictionary[ftype]['rock'] = labels['rock']
        sparse_dictionary[ftype]['rock_depth'] = labels['rock_depth']
        sparse_dictionary[ftype]['material'] = labels['material'].astype(
            np.string_)
        sparse_dictionary[ftype]['patterns'] = labels['patterns'].astype(
            np.string_)
        sparse_dictionary[ftype]['exp_path'] = other['exp_path'].astype(
            np.string_)
        sparse_dictionary[ftype]['time_to_compute'] = [time_to_compute]

        file_names = np.asarray(
            list(other['file_names'].astype(str))).astype(np.string_)
        sparse_dictionary[ftype]['files'] = file_names
        sparse_dictionary[ftype]['imu_contact_bin'] = labels['imu_contact_bin']

        sparse_dictionary[ftype]['features'] = {}
        for k in features.keys():
            sparse_dictionary[ftype]['features'][k] = np.array(features[k])

        sparse_dictionary[ftype]['hydration'] = labels['hydration']

        sparse_dictionary[ftype]['features_hyd'] = {}
        for k in features_hydration.keys():
            sparse_dictionary[ftype]['features_hyd'][k] = np.array(
                features_hydration[k])

    sparse_file = os.path.join(output_folder, outfile_name + "_sparse.h5")
    if os.path.isfile(sparse_file):
        os.remove(sparse_file)
    h5_util.save_h5(sparse_file, sparse_dictionary)

    feature_stacked_sparse = tools_pg.read_all_sparse_features(
        sparse_dictionary, module)
    logging.info(
        f"Dimension of sparse feature matrix: {feature_stacked_sparse['X'].shape}")

    feature_selected_sparse = tools_pg.selectFeatures(feature_stacked_sparse)
    logging.info(
        f"Dimension of feature selected matrix: {feature_selected_sparse['X'].shape}")

    features_dense, nan_mask = tools_pg.remove_nan(
        feature_selected_sparse, module)
    logging.info(
        f"Dimension of NaN cleansed matrix: {features_dense['X'].shape}")

    features_dense["feature_names"] = features_dense["feature_names"].astype(
        np.string_)
    features_dense['material'] = features_dense['material'].astype(np.string_)
    features_dense['patterns'] = features_dense['patterns'].astype(np.string_)
    features_dense["feature_names_hyd"] = features_dense["feature_names_hyd"].astype(
        np.string_)
    features_dense['files'] = np.asarray(
        list(features_dense['files'].astype(str))).astype(np.string_)

    dense_file = os.path.join(output_folder, outfile_name + "_dense.h5")
    if os.path.isfile(dense_file):
        os.remove(dense_file)
    h5_util.save_h5(dense_file, features_dense)

    return sparse_dictionary, features_dense, nan_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--multithreading', help='multithreading trigger',
                                            action='store_true')

    parser.add_argument('--datadir',        default="/Volumes/MLIA_active_data/data_barefoot/",
                                            help="Path to data repository. Defaults to /Volumes/MLIA_active_data/data_barefoot/")

    parser.add_argument('--features',       default="full_features.txt",
                                            help="List of features to compute.  One per line.")

    parser.add_argument('--datafile',       help="Full path to unified data H5 created using compute_data.py")

    parser.add_argument('--outfolder',      help="Full path, and name, of the resulting sparse feature data file")

    parser.add_argument('--outfile_name',   help="Full path, and name, of the resulting sparse feature data file")

    parser.add_argument('--burnin',         nargs='+',
                                            help="Two values indicating low-end burn in and high-end burn in.")

    parser.add_argument('--plot',           help="Generate feature plots.  Set to True if required, defaults to False")

    parser.add_argument('--module',         default="rock",
                                            help="Barefoot classifier type.  Ex: rock, patterns, hydrations")

    args = parser.parse_args()

    logging.basicConfig(filename='compute_features.log', level=0)

    sparse_features, dense_features, nan_mask = generate_features(args.datadir,
                                                                  args.outfolder,
                                                                  args.datafile,
                                                                  args.outfile_name,
                                                                  args.burnin,
                                                                  args.plot,
                                                                  args.module,
                                                                  args.multithreading, args.features)
