# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:14:34 2018

@author: marchett
"""
import argparse
import copy
import glob
import itertools
import logging
import os
import re
import sys
import timeit
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import h5py
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib as mp
import numpy as np
import pandas as pd
import pywt
from idlelib.run import exit
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import generic_filter
from scipy.signal import argrelextrema, find_peaks_cwt
from scipy.stats import kurtosis, skew
from skimage import measure, morphology, segmentation
# ------------
# TEST FEATURES
from skimage.feature import local_binary_pattern, peak_local_max
from skimage.filters import rank, threshold_local, threshold_otsu
from sklearn.neighbors import KDTree
from sklearn.tree import DecisionTreeRegressor
from tsfresh.feature_extraction import feature_calculators as feat_cal

from bf_config import bf_globals
from bf_logging import bf_log
from bf_util import numpy_util, utc_util

mp.use('Agg')

np.seterr(divide='ignore', invalid='ignore')

bf_log.setup_logger()
logger = logging.getLogger(__name__)

# --------------
grouser_even_idx = [0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                    34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
                    70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]
nongrouser_odd_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
                      35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65,
                      67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]

grouser_idx = [6, 8, 10, 12, 14, 16, 18, 20, 22, 28, 30, 32,
               34, 36, 38, 40, 42, 44, 52, 54, 56, 58, 60, 62, 64, 66, 68, 76, 78, 80, 82,
               84, 86, 88, 90, 92]
nongrouser_idx = [5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 29, 31, 33, 35, 37, 39, 41,
                  43, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 75, 77, 79, 81, 83, 85, 87,
                  89, 91, 93]
ambiguous_idx = [0, 1, 2, 3, 4, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50,
                 70, 71, 72, 73, 74, 94, 95]
all_idx = range(0, 96)

# can change this dictionary to different index
contact_area_keys = ['all', 'grouser', 'nongrouser', 'ambiguous']
contact_area_indexes = {'all': all_idx, 'grouser': grouser_even_idx, 'nongrouser': nongrouser_odd_idx, 'ambiguous':
                        ambiguous_idx, 'grouser_noamb': grouser_idx, 'nongrouser_noamb': nongrouser_idx}


grain_size_dict = {'redgar': '1.55', 'wed730': '0.14', 'mmdust': '0.025', 'mm.2mm': '0.04',
                   'mmcrse': '1.0', 'bst110': '0.08', 'grc-01': '0.39', 'mins30': '0.19',
                   'mmintr': '0.22'}

# types of terrain and integer for class label, 'gradbed' is bedrock
terrain_dict = {'flatlvl': 0, 'gullies': 1, 'pebbles': 2, 'sharpdunes': 3, 'smoodunes': 4,
                'rock-above': 5, 'bedrock': 6, 'JPL': 7, 'footprints': 8, 'freeslip': 9,
                'rock-below': 10}

# types of compositions and integer for class label
composition_dict = {'mmcrse': 0, 'grc-01': 1, 'mmintr': 2, 'mins30': 3,
                    'wed730': 4, 'mm.2mm': 5, 'bst110': 6}


@dataclass(init=True)
class ExperimentMetadata:
    """Class for data extracted from the experiment directory

    Attributes
    ----------
    label: str\n
    terrain: str\n
    loading: str\n
    material: str\n
    hydration: str\n
    date: str\n
    rep: str
    """
    label: str
    terrain: str
    loading: str
    material: str
    hydration: str
    date: str
    rep: str

    def __repr__(self):
        return (f"{self.label}_terrain_{self.terrain}_{self.loading}_"
                f"{self.material}_{self.hydration}_{self.date}_{self.rep}")
    def __str__(self):
        return (f"{self.label}_terrain_{self.terrain}_{self.loading}_"
                f"{self.material}_{self.hydration}_{self.date}_{self.rep}")

def extract_exp_metadata(experiment_path: str) -> ExperimentMetadata:
    """ Extracts experiment meta data

    Parmeters
    ---------
    experiment_path: str
        Path to experiment directory.  Does not have to be a path.
        It could be the experiment directly

    Returns
    --------
    metadata: ExperimentMetadata
        Experiment metadata or 'ftype' as it is called in the code


    Raises
    ------
    ValueError
        If the experiment name is malformed
    """
    validate = re.compile(
        r"""[^_]+_([^_]+)_ # group 1
        terrain_([^_]+)_ # group 2
        vel_([^_]+)_ # group 3
        EISfreqsweep_([^_]+)_ # group 4
        grousers_([^_]+)_ # group 5
        loading_([^_]+)_ # group 6
        material_([^_]+)_ # group 7
        hydra_([^_]+)_ # group 8
        pretreat_([^_]+)_ # group 9
        date_([^_]+)_ # group 10
        rep_([^_]+) # group 11
        """, re.VERBOSE)
    exp = experiment_path.rstrip("/").split('/')[-1]
    match = validate.fullmatch(exp)
    if not match:
        error_msg = (f"Expriment {exp} in path {experiment_path} is not "
                     f"of the correct format")
        logger.error(error_msg)
        raise ValueError(exp)
    return ExperimentMetadata(label=match.group(1),
                              terrain=match.group(2),
                              loading=match.group(6),
                              material=match.group(7),
                              hydration=match.group(8),
                              date=match.group(10),
                              rep=match.group(11))


def get_calibration_data(cal_dir: str,
                experiment_date: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Retrieves mean and standard deviation date from the passed
        calibration directory.  The calibration files used depends on
        the date of the experiment.

    Parameters
    ----------
    cal_dir : str
        Calibration directory
    experiment_date : str
        Date of experiment

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two numpy arrays
    """
    for d in bf_globals.CALIBRATION_FILE_DATES:
        if experiment_date >= d:
            cal_dir = f"{cal_dir}/cal_{d}"
            break
    logger.info(f"Using calibration directory {cal_dir}")
    mean_file = f"{cal_dir}/mean_offset.npy"
    std_file = f"{cal_dir}/std_offset.npy"
    if not (os.path.isfile(mean_file) and os.path.isfile(std_file)):
        logger.debug(f"Looking for mean/std files in {cal_dir}")
        files = glob.glob("/".join([cal_dir, '*.dat']))
        # TODO: Add check of files
        mean_offset, std_offset = extract_mean_std(files)
        np.save(mean_file, mean_offset)
        np.save(std_file, std_offset)
    else:
        mean_offset = np.load(mean_file)
        std_offset = np.load(std_file)

    return mean_offset, std_offset

# --------------
def extract_mean_std(pressure_grid_files):
    """Calculate the mean and standard deviation arrays of the pressure grid data
       at each point in time in each *.dat file in the passed directory.

    Parameters
    ----------
    datadir : str
        Main data directory
    pressure_grid_files : list
        List if pressure grid files (.dat) that will be used to create the mean and std arrays

    Returns
    -------
    mean_offset: array containing the pressure sensor dark means
    std_offset: array containing the pressure sensor dark standard deviations

    """

    if not pressure_grid_files:
        error_msg = 'No calibration files found.'
        logger.warning(error_msg)
        sys.exit(error_msg)

    data_all = np.vstack([
        convert_xiroku(f)[2] for f in pressure_grid_files
    ])
    mean_offset = np.rot90(data_all.mean(axis=0))
    std_offset = np.rot90(data_all.std(axis=0))

    return mean_offset, std_offset


# --------------
def convert_xiroku(dat_filename, num_cols=96, num_rows=20):
    """Read .dat file that contains the pressure sensor data and convert them to numpy arrays.

    An example of the file format is as follows:

    Capture started at Fri Mar 09 17:39:10 2018
    frameno=16233
    timestamp=2018/03/09 17:39:10.978
    frequency=937.500000
    gain=0
    number=64
    xcoilmin=1
    xcoilmax=96
    ycoilmin=1
    ycoilmax=20
    1	  1
    96	 20

    Parameters
    ----------
    dat_filename: the name of the .dat file
    num_cols: number of columns of the pressure sensor
    num_rows: number of row of the pressure sensor

    Returns
    -------
    data_flattened : array
        N x (num_cols*num_rows + 1) array where N is the number of frames. The first column is the time stamp\n
    data_timestamp : array
        N element vector of UTC time stamps where N is the number of frames\n
    data_xy : array
        N x num_cols x num_rows array which contains each of the N pressure frames\n

    Notes
    ------
    For now num_rows = 20 and num_cols = 96
    """

    # parse through initially to find total size
    num_frames = 0
    CN = 0
    with open(dat_filename) as f:
        for l in f.readlines():
            if l.startswith('timestamp'):
                num_frames += 1
            elif l.startswith('Capture started'):
                CN += 1

    logger.info(f"Xiroku datapoints: {num_frames}")
    logger.info(f"Xiroku capture count: {CN}")

    data_flattened = np.zeros([num_frames, 1+num_cols*num_rows])
    data_xy = np.zeros([num_frames, num_cols, num_rows])
    data_timestamp = np.zeros([num_frames, 1])
    ni = -1
    with open(dat_filename) as f:
        logger.debug(f"Processing file {dat_filename}")
        lines = f.readlines()
        for l in lines:
            if l.startswith('timestamp'):
                tstr = repr(l).split('\\')[0][11:]
                dt_s = utc_util.utc_time_from_xiroku(tstr)
                # yi=0   #reset y-index
                ni += 1
                data_flattened[ni][0] = dt_s
                data_timestamp[ni] = dt_s
            else:
                l_arr = l.split('\t')
                if len(l_arr) == num_cols+1:  # either data or header string
                    if l_arr[0] == 'amp':
                        continue
                    else:
                        xi = int(l_arr[0])
                        data_flattened[ni][1+(xi-1)*num_cols:1+xi *
                                           num_cols] = [float(a) for a in l_arr[1:]]
                        data_xy[ni, :, xi-1] = [float(a)
                                                for a in l_arr[1:]]  # 96 data row

    return data_flattened, data_timestamp, data_xy


# -------------
def bad_impute_pixel_map(bad_col=23, rows=20, cols=96, k=9):
    """Compute neighbors for pressure sensor bad pixels given a column number.

    Parameters
    ----------
    bad_col : int
        The index of the bad column
    rows : int
        Number of rows of pressure sensor
    cols : int
        Number of columns of pressure sensor
    k : int
        Number of nearest neighbors to find

    Returns
    -------
    impute_masks: list
    idx_mask : array
        A (rows X 2) array where each row is [row_number, bad_col]

    """
    X = np.repeat(np.arange(0, rows), cols)
    Y = np.tile(np.arange(0, cols), rows)
    XY = np.column_stack([X, Y])
    XY_grid = np.arange(0, cols*rows).reshape((rows, cols))

    tree = KDTree(XY)
    impute_masks = []
    row_count_vector = np.arange(rows)
    bad_columns = np.repeat(bad_col, rows)
    for j in range(rows):
        X0 = XY[XY_grid[j, bad_col]]
        X0 = X0.reshape((1, -1))
        ndist, ind = tree.query(X0, k=k)
        ind = ind[ndist < 2]
        a1 = ~np.in1d(XY[ind][:, 0], row_count_vector)
        a2 = ~np.in1d(XY[ind][:, 1], bad_columns)
        mask = a1 + a2
        impute_masks.append(XY[np.hstack(ind)[mask]])

    idx_mask = np.array([(a, bad_col) for a in range(rows)])
    return impute_masks, idx_mask


# -------------
def impute_pixels(pg, idx_mask, impute_masks):
    """Replace the bad pixels by the mean of the neighboring pixels.

    Parameters
    ----------
    pg : array
        Array of pressure values for each time step/readout
    idx_mask : arrary
        A (rows X 2) array where each row is [row_number, bad_col]

    impute_mask :

    Returns
    -------
    pg : array
        Array of pressure values with imputed bad pixels

    """
    for j in range(len(idx_mask)):
        knn_values = pg[impute_masks[j][:, 0], impute_masks[j][:, 1]]
        pg[idx_mask[j, 0], idx_mask[j, 1]] = np.mean(knn_values)

    return pg


def scale_pg_data(pg_data: np.ndarray,
                 mean_array: np.ndarray,
                 std_array: np.ndarray,
                 experiment_date: str) -> Tuple[np.ndarray, np.ndarray]:
    """Scales pressure gride data based on mean/std data passed

    Parameters
    ----------
    pg_data : np.ndarray
        Pressure grid data\n
    mean_array : np.ndarray
        Mean offset\n
    std_array : np.ndarray
        Standard deviation offset\n
    experiment_date : str
        Date of experiment\n

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Degraded pressure grid data, scaled pressure grid date
    """

    # scale/zero-offset each pressure grid in time
    N = pg_data.shape[0]

    stacked_mean_offset = np.stack([np.rot90(mean_array) for _ in range(N)])
    stacked_std_offset = np.stack([np.rot90(std_array) for _ in range(N)])

    pg_scaled = -(pg_data - stacked_mean_offset) / stacked_std_offset
    # shape = (number of frames, rows=20, columns=96)
    pg_scaled = np.rot90(pg_scaled, axes=(1, 2))

    if experiment_date > '20181230':
        # TODO - Refactor bad_impute_pixel_map() and impute_pixesl() so that no for loop is needed
        impute_masks, idx_mask = bad_impute_pixel_map(rows=20, cols=96)
        for t in range(N):
            pg_norm = pg_scaled[t, :, :]
            pg_norm = impute_pixels(pg_norm, idx_mask, impute_masks)
            pg_scaled[t, :, :] = pg_norm

    pg_degrad = np.median(pg_scaled, axis=0)
    # apply degradation
    pg_scaled = pg_scaled - pg_degrad[np.newaxis, :, :]

    return pg_degrad, pg_scaled

# -------------
def read_all_pg_data(experiment_path: str, no_contact_dir: str, plot=True):
    """Read and calibrate pressure sensor data

    Read other related data values,
    including IMU rotational angle, slip ratio, motor current, force/torque and
    zstage data, rock labels, pattern name, material name.

    Parameters
    ----------
    experiment_path  : string
        The name of a particular run/experiment directory.
    no_contact_dir : string
        Full path to directory that contains mean_offset.npy, std_offset.npy and a default pressure
        grid data file in case these files are not present.  The default pressure grid data file will
        then be used to calculate the mean and standard deviation.
    plot : bool
        whether to create and save the pressure sensor degradation plot. Default: True.

    Returns
    -------
    ftype : str
        Path name split and joined to form a shortened run/experiment name\n
    data  : dict
        A dictionaty containing the data\n
    time : dict
        A dictionary containing time stamp corresponding to each data point in
        the data dictionary.

    Notes
    -----
    The data dictionay will have the following key value pairs where K is some integer:\n
    pg: N x 20 x 96 array\n
    slip: (N,) or (K,) vector/array (last column in slip/slip_sp file)\n
    imu: (K,) vector/array (last column in imu file)\n
    sink: (K,) vector/array (third column in cart file)\n
    current: (K,) vector/array (fourth column in cart file)\n
    ft_xyz: (K,6) vector/array (second column in ft_file file)\n
    rock: (K, ) vector/array (sum of 2nd and third columns in rock file)\n
    rock_depth: (K, ) vector/array (5th column in rock file)
    pattern:
    material:
    slip_fiducials: (K, ) vector/array (last column in slip file)\n\n

    The time dictionary will have the following key-value pairs:
    pg: N x 20 x 96 array\n
    slip: (N,) or (K,) vector/array (first column in slip/slip_sp file)\n
    imu: (K,) vector/array (first column in imu file)\n
    sink: (K,) vector/array (first column in cart file)\n
    current: (K,) vector/array (first column in cart file)\n
    ft_xyz: (K,) vector/array (first column in ft_file file)\n
    rock: (K, ) vector/array (first of 2nd and third columns in rock file)\n
    rock_depth: (K, ) vector/array (first column in rock file)
    pattern:
    material:
    slip_fiducials: (K, ) vector/array (first column in slip file\n

    """
    files_dat = glob.glob(f"{experiment_path}/*.dat")
    if len(files_dat) == 0:
        error_msg = f"No PG data in {experiment_path}"
        logger.error(error_msg)
        sys.exit(error_msg)

    # Cannot continue without IMU data
    imu_file = glob.glob(f"{experiment_path}/*wheel_v3.npy")
    imu_file = glob.glob(
        f"{experiment_path}/*wheel.npy") if len(imu_file) == 0 else imu_file
    if len(imu_file) == 0:
        error_msg = "No IMU file found. (wheel_v3.npy or wheel.npy)"
        logger.error(error_msg)
        sys.exit(error_msg)

    # TODO Use regex since there could be more than one terrain type
    exp = experiment_path.rstrip("/").split('/')[-1]
    experiment_data = extract_exp_metadata(exp)

    data_file = files_dat[0]
    data_flat, time_pg, load_pg = convert_xiroku(data_file)
    if data_flat.shape[0] == 0:
        error_msg = f'No PG data recorded in {data_file}'
        logger.error(error_msg)
        sys.exit(error_msg)

    mean_offset, std_offset = get_calibration_data(no_contact_dir, experiment_data.date)

    pg_degrad, pg_scaled = scale_pg_data(load_pg, mean_offset,
                                         std_offset, experiment_data.date)

    # plot degradation
    ftype = str(experiment_data)
    if plot:
        if not os.path.exists(f"{experiment_path}/plots/"):
            os.makedirs(f"{experiment_path}/plots/")
        im = plt.matshow(pg_degrad, aspect='auto')
        im.set_clim((0, 120))
        plt.gca().xaxis.tick_bottom()
        plt.title('median dark, ' + ftype)
        plt.colorbar()
        plt.savefig(f"{experiment_path}/plots/dark_degrad_corr_{ftype}.png")
        plt.close()

    data = {'pg': pg_scaled}
    time = {'pg': np.ravel(time_pg)}
    crossbow_data_dict, crossbow_time_dict = load_crossbow_data(experiment_path, len(pg_scaled))
    data.update(crossbow_data_dict)
    time.update(crossbow_time_dict)

    # pattern
    ftype_surf = ftype.split('_')
    stypes = list(terrain_dict)
    idx = np.where(np.in1d(stypes, ftype_surf))[0]
    data['pattern'] = np.ravel([stypes[x]
                                for x in idx]) if len(idx) > 0 else 'unknown'
    if data['pattern'] == 'unknown':
        logger.info(f"Unknown surface type in {ftype}")
    if len(idx) > 1:
        logger.info('More than 1 pattern category found')

    # material
    mtypes = ['redgar', 'bst110', 'wed730', 'grc-01',
              'mins30', 'mm.2mm', 'mmintr', 'mmcrse']
    idx = np.where(np.in1d(mtypes, ftype_surf))[0]
    data['material'] = np.ravel([mtypes[x] for x in idx])

    return ftype, data, time


def load_crossbow_data(experiment_path: str, N: int) \
    -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Loads data gathered from CROSSBOW. Only meant to be called
       from read_all_pg_data

    Parameters
    ----------
    experiment_path : str
        Experiment directory\n
    time_pg: np.ndarray
        Array of timestamps for each pressure grid frame\n
    N : int
        Number of pressure grid frames in the *.dat fiel from the
        experiment path passed.

    Returns
    -------
    crossbow_dict, time_dict: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        Dictionary of crossbow data and a dictionary of its timestamps.
        Each dictionary will have the following keys: imu, slip,
        slip_fudicials, sink, current, ft_xyz and rock.  The
        crossbow dictionary will also have the key 'rock_depth'.
    """

    # rotational angle from IMU
    imu_file = glob.glob(f"{experiment_path}/*wheel_v3.npy")
    imu_file = glob.glob(f"{experiment_path}/*wheel.npy") \
               if len(imu_file) == 0 else imu_file
    logger.info(f"Using wheel file {imu_file} for IMU data")
    load_imu = np.load(imu_file[0])

    crossbow_dict = {'imu': load_imu[:, -1],
                     'slip': np.repeat(np.nan, N),
                     'slip_fiducials': np.repeat(np.nan, N),
                     'sink': np.repeat(np.nan, N),
                     'current': np.repeat(np.nan, N),
                     'ft_xyz': np.repeat(np.nan, N * 6).reshape((N, 6)),
                     'rock': np.repeat(0, N),
                     'rock_depth': np.repeat(0, N)
                    }
    time_dict = {'imu': load_imu[:, 0],
                 'slip': np.repeat(np.nan, N),
                 'slip_fiducials': np.repeat(np.nan, N),
                 'sink': np.repeat(np.nan, N),
                 'current': np.repeat(np.nan, N),
                 'ft_xyz': np.repeat(np.nan, N),
                 'rock': np.repeat(0, N)
                }

    # slip
    slip_file = glob.glob(f"{experiment_path}/*slip_sp.npy")
    slip_file = glob.glob(f"{experiment_path}/*slip.npy") \
                if len(slip_file) == 0 else slip_file
    if len(slip_file):
        logger.info(f"Found slip file {slip_file}")
        load_slip = np.load(slip_file[0])
        crossbow_dict['slip'] = load_slip[:, -1]
        time_dict['slip'] = load_slip[:, 0]

    # fiducials slip
    slip_file = glob.glob(f"{experiment_path}/*slip.npy")
    if len(slip_file):
        logger.info(f"Using file {slip_file} for slip fuducial data")
        load_slip = np.load(slip_file[0])
        crossbow_dict['slip_fiducials'] = load_slip[:, -1]
        time_dict['slip_fiducials'] = load_slip[:, 0]

    #read in force/torque
    ft_file = glob.glob(f"{experiment_path}/*ati.npy")
    if len(ft_file):
        logger.info(f"Found ati file {ft_file}.  Loading force/torque data.")
        load_ft = np.load(ft_file[0])
        crossbow_dict['ft_xyz'] = load_ft[:, 1:]
        time_dict['ft_xyz'] = load_ft[:, 0]

    # sink and ESCON motor current value
    cart_file = glob.glob(f"{experiment_path}/*cart.npy")
    if len(cart_file):
        logger.info((f"Found cart file {cart_file}.  "
                     f"Loading sink and ESCON motor current"))
        load_sink = np.load(cart_file[0])
        crossbow_dict['sink'] = load_sink[:, 2]
        crossbow_dict['current'] = load_sink[:, 3]
        time_dict['sink'] = load_sink[:, 0]
        time_dict['current'] = load_sink[:, 0]

    # rock
    # Rock file format is as follows: time_time str(int) str(int) int float
    # Column 1 is time
    # Columns 2 and 3 are the rock data
    # Column 5 is rock_depth
    # Column 4?
    rock_file = glob.glob(f"{experiment_path}/rock_positions*.txt")
    rock_file = glob.glob(f"{experiment_path}/*rocks.txt") \
                if len(rock_file) == 0 else rock_file
    if len(rock_file):
        logger.info(f"Using file {rock_file} for rock data.")
        load_rock = np.loadtxt(rock_file[0], dtype=str)
        rock_split = [x.split('_') for x in load_rock[:, 0]]
        time_dict['rock'] = np.hstack(['.'.join(x)
                                  for x in rock_split]).astype(float)
        mask_nan = load_rock[:, 1:-1] == 'Nan'
        load_rock[:, 1:-1][mask_nan] = '0'
        crossbow_dict['rock'] = load_rock[:, 1:-2].astype(int).sum(axis=1)
        load_rock[load_rock[:, -1] == '.'] = 0.0
        crossbow_dict['rock_depth'] = load_rock[:, -1].astype(float)

    return crossbow_dict, time_dict



# -------------
def align_pg_to_imu(data, time, T, R):
    """Align pressure sensor and other related variables in time by averaging each
    variable on a regular grid of time bins.

    Parameters
    ----------
    data: dictionary
        A dictionary of all the variables, including the pressure sensor data,
        to align, output of read_all_pg_data
    time: dictionary
        A dictionary with corresponding time stamps for each variable in the
        data dictionary, output of read_all_pg_data
    T: float
        The size of the time step for the regular time grid

    R: int
        window over which to average IMU rotational angle such that the average is
        mean(x[t-R:t+R+1])

    Returns
    -------
    data_binned: dictionary
        A dictioary of all the variables on the same time grid
    time_binned: array
        A vector of time values for the regular time grid

    """
    # breakpoint()
    t_min = min(time['pg'].min(), time['imu'].min())
    t_max = max(time['pg'].max(), time['imu'].max())
    bins = np.arange(t_min, t_max, T)

    rho_x = np.cos(np.radians(data['imu']))
    rho_y = np.sin(np.radians(data['imu']))
    rho_counts = np.histogram(time['imu'], bins=bins)[0]
    rho_binned_x = np.histogram(time['imu'], weights=rho_x, bins=bins)[
        0] / rho_counts
    rho_binned_y = np.histogram(time['imu'], weights=rho_y, bins=bins)[
        0] / rho_counts
    rho_binned = np.rad2deg(np.arctan2(rho_binned_y, rho_binned_x)) % 360

    bin_mean = np.histogram(bins[:-1], weights=bins[:-1], bins=bins)[0]
    pg_counts = np.histogram(time['pg'], bins=bins)[0]
    mask = (rho_counts > 0) & (pg_counts > 0)
    time_binned = bin_mean[mask]

    ynew_rho = rho_binned[mask]
    rho_x = np.cos(np.radians(ynew_rho))
    rho_y = np.sin(np.radians(ynew_rho))
    running_rho_x = numpy_util.running_mean(rho_x, R, R)
    running_rho_y = numpy_util.running_mean(rho_y, R, R)

    data_binned = {k: [] for k in data.keys()}
    data_binned['imu'] = np.rad2deg(
        np.arctan2(running_rho_y, running_rho_x)) % 360
    pg_binned = np.apply_along_axis(lambda a:
                                    np.histogram(time['pg'], weights=a, bins=bins)[0], 0, data['pg'])
    data_binned['pg'] = pg_binned[mask, :, :] / \
        pg_counts[mask][:, np.newaxis, np.newaxis]

    for k in ['slip', 'slip_fiducials', 'sink', 'current']:
        count = np.histogram(time[k], bins=bins)[0]
        binned_data = np.histogram(time[k], weights=data[k], bins=bins)[0]
        data_binned[k] = binned_data[mask] / count[mask]

    rock_counts = np.histogram(time['rock'], bins=bins)[0]
    rock_counts[rock_counts == 0] = 1
    rock_depth = np.histogram(
        time['rock'], weights=data['rock_depth'], bins=bins)[0]
    data_binned['rock_depth'] = rock_depth[mask] / rock_counts[mask]

    rock_binned = np.histogram(
        time['rock'], weights=data['rock'], bins=bins)[0]
    data_binned['rock'] = rock_binned[mask] / rock_counts[mask]
    data_binned['rock'][data_binned['rock'] > 0] = 1
    vd = np.unique(data['rock_depth'][(data['rock'] == 1)])
    if len(vd) > 0:
        #rock_mask = numpy_util.running_mean(data_binned['rock'], 0, 1) == 0.5
        rock_mask = numpy_util.running_mean(data_binned['rock'], 1, 0) == 0.5
        data_binned['rock'][rock_mask] = 1
        vd = np.unique(data['rock_depth'][(data['rock'] == 1)])[0]
        data_binned['rock_depth'][rock_mask] = vd

    ft_counts = np.histogram(time['ft_xyz'], bins=bins)[0]
    ft_binned = []
    for j in range(data['ft_xyz'].shape[1]):
        hist_sums = np.histogram(
            time['ft_xyz'], weights=data['ft_xyz'][:, j], bins=bins)[0]
        ft_binned.append(hist_sums[mask] / ft_counts[mask])

    data_binned['ft_xyz'] = np.column_stack(ft_binned)
    data_binned['material'] = np.repeat(data['material'], len(time_binned))
    data_binned['pattern'] = np.repeat(data['pattern'], len(time_binned))

    return data_binned, time_binned


# --------------
def locate_on_pg(idx_imu, end_val=96):
    """Account for the circular nature of the pressure sensor and adjust column index if
    the column index is greater than the total number of columns or less than 0. This
    situation can arise when computing the unwrap image, e.g. when idx_imu = 95 and lag = 3.

    Parameters
    ----------
    idx_imu: int
        Column index of the pressure grid, where IMU is showing gravity down position
        end_val: int
                Total number of columns of the pressure sensor

    Returns
    -------
    idx_imu: int
        Adjusted column of the pressure sensor


    """
    new_idx_imu = idx_imu - end_val if idx_imu > end_val-1 else idx_imu

    if new_idx_imu < 0:
        new_idx_imu = new_idx_imu + end_val
    if new_idx_imu > end_val - 1:
        new_idx_imu = new_idx_imu - end_val

    return new_idx_imu


# --------------
def contact_imu(rho_binned):
    """Obtain a column index of the pressure sensor corresponding to the rotational
    angle of the IMU.

    Parameters
    ----------
    rho_binned: narray
                Vector of rotational values from matched IMU

    Returns
    -------
    idx_imu: narray
        Column index of the pressure grid corresponding to the IMU angle

    """

    rot_to_pg = np.linspace(0, 360, 96)
    idx_adj = np.searchsorted(rot_to_pg, 63)
    sort_pg = np.hstack([np.arange(idx_adj, 96), np.arange(0, idx_adj)])

    idx_imu = np.zeros((len(rho_binned), ))
    for t in range(len(rho_binned)):
        idx_contact = np.searchsorted(rot_to_pg, rho_binned[t])
        # unflip
        idx_contact = np.where(
            rot_to_pg[sort_pg][::-1] == rot_to_pg[idx_contact])[0][0]
        idx_imu[t] = locate_on_pg(idx_contact, 96)

    return idx_imu


# --------------
def unwrap_pg_to_imu(pg_binned, rho_binned, x_binned, lag, extraction='nongrouser'):
    """ Create an unwrap image based on selecting the column of the pressure sensor
    that corresponds to IMU contact column and the corresponding columns before
    (front of the wheel) or after (back of the wheel) the IMU column. The x-axis of
    the unwrap image corresponds to time and y-axis corresponds to the columns of
    the pressure sensor across 20 rows.

    Parameters
    ----------
    pg_binned : narray\n\t
        An array with pressure sensor data, binned to the common time grid\n
    rho_binned : narray
        An array with IMU rotational angle, binned to the common time grid\n
    x_binned : narray
        A vector of time values for the matched data\n
    lag : list
        A list of integers representing the difference of the pressure sensor column
        to the IMU contact column index, e.g. [-2, -1, 0, 1, 2] corresponds to an
        unwrap image with 0 as the IMU contact column and two columns before and after it.\n
    extraction : string
        Whether to unwrap for 'grouser','grouser_noamb', 'non-grouser', 'nongrouser_noamb' or both 'all'.
        Default 'nongrouser'\n

    Returns
    -------
    pg_imu: list
        A list of pressure sensor unwrap arrays with length corresponding to the length of
        the lag list. Each element of the list correspond to the unwrap for the lag specified
        in the lag list. Each list element is an array of pressure sensor rows (20) by
        the number of time steps.\n
    rho_imu: list
                A list of IMU rotational angles per lag, with length corresponding to the number of
                time steps\n
    x: list
        A list of matched time values per lag

    """
    extraction_options = ['grouser', 'grouser_noamb',
                          'nongrouser', 'nongrouser_noamb', 'all']
    if extraction not in extraction_options:
        error_msg = f"Extraction type {extraction} is not one of {','.join(extraction_options)}"
        logger.error(error_msg)
        sys.exit(error_msg)

    idx_contact_binned = contact_imu(rho_binned)
    pg_imu = []
    x = []
    for l in range(len(lag)):
        x.append([])
        pg_imu.append(np.zeros((pg_binned.shape[1], len(rho_binned))))
        for t in range(len(x_binned)):
            idx_contact = idx_contact_binned[t].astype(int)
            is_nongrouser_off = np.in1d(
                contact_area_indexes['nongrouser'], idx_contact).sum() != 1
            is_grouser_off = np.in1d(
                contact_area_indexes['grouser'], idx_contact).sum() != 1
            is_nongrouser = extraction == 'nongrouser' or extraction == 'nongrouser_noamb'
            is_grouser = extraction == 'grouser' or extraction == 'grouser_noamb'
            if (is_nongrouser and is_nongrouser_off) or (is_grouser and is_grouser_off):
                pg_imu[l][:, t] = np.nan
            else:
                # if 'all' then lag[l] b/c wheel moves in the direction of smaller bin number per IMU
                # otherwise lag[l]*2 b/c wheel moves in the direction of smaller bin number per IMU
                is_extract_all = extraction == 'all'
                idx_next = np.atleast_1d(idx_contact - lag[l])[0] if is_extract_all \
                    else idx_contact - lag[l]*2

                if idx_next >= len(contact_area_indexes['all']):
                    idx_imu = locate_on_pg(
                        idx_next, len(contact_area_indexes['all']))
                else:
                    idx_imu = idx_next if is_extract_all else contact_area_indexes['all'][idx_next]
                pg_imu[l][:, t] = pg_binned[t, :, idx_imu]
                x[l].append(t)
        t_sum = pg_imu[l].sum(axis=0)
        mask = np.isnan(t_sum)
        pg_imu[l] = pg_imu[l][:, ~mask]
        if len(x[l]) > 0:
            x[l] = np.hstack(x[l])
    rho_imu = rho_binned[~mask]

    return pg_imu, rho_imu, x


# --------------
def waveletDegrouse(pg_imu: List[np.ndarray], wtype='haar', level=3):
    """ Apply wavelet smoothing to the unwrap images.

    Parameters
    ----------
    pg_imu: list
        A list of unwrap images, output of unwrap_pg_to_imu.
        wtype: string
                Wavelet basis type, see types for pywt package function wavedec2
        level: int
                Wavelet decomposition level, see types for pywt package function wavedec2

    Returns
    -------
    pg_recon: list
        A list of the same length as pg_imu with the wavelet
        reconstructed smooth images.

    """

    pg_recon = []
    for imu_data in pg_imu:
        coeffs = pywt.wavedec2(imu_data, wtype, level=level)
        coeffs_H = copy.deepcopy(coeffs)
        for z in range(1, len(coeffs_H)):
            list(coeffs_H[z])[1] *= 0
        recon = pywt.waverec2(coeffs_H, wtype)
        if recon.shape[1] > imu_data.shape[1]:
            recon = recon[:, :-1]
        pg_recon.append(recon)

    return pg_recon


# -------------
def number_rot(rho_y, d=200.):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    rho_y : [type]
        [description]
    d : [type], optional
        [description], by default 200.

    Returns
    -------
    [type]
        [description]
    """
    rho_int = np.ceil(rho_y)
    upper = np.where(rho_int == 360)[0]
    mask_upper = np.diff(upper) > 100
    n_full_rot = mask_upper.sum()

    part_rot1 = 360. - rho_int[0]
    part_rot2 = rho_int[-1] - 0.

    n_part_rot = (part_rot1 + part_rot2) / 360.
    n_rot = n_full_rot + n_part_rot

    wc = 165.
    mean_slip = 1 - np.min([d / (n_rot * wc), 1.])

    return n_rot, mean_slip

# -------------


def rot_start_stop_motor(current_binned: np.ndarray, imu: np.ndarray,
                         graphs=False):
    """ Creates a mask that removes stationary data, i.e. data when the wheel is
    not rotating based on the motor current data.

    Parameters
    ----------
    current_binned: ndarray
        Motor current data binned to the common time grid. Output from
        align_pg_to_imu.
    imu:
        IMU data
    graphs : bool
        Output plots, True or False.

    Returns
    -------
    stationary_mask: bool
        A mask of the same size as data_binned that removes unmoving parts of
        a run/experiment.

    """
    current = current_binned.copy()
    current[np.isnan(current)] = 0
    stationary_mask = current == 0

    rho_y = np.sin(np.radians(imu))
    if stationary_mask.sum() == len(current):
        logger.info('Motor data zero... switch to IMU for stationary mask')
        idx_start, idx_stop = rot_start_stop(rho_y, graphs=False)

        stationary_mask = np.repeat(True, len(current))
        stationary_mask[idx_start[0]:idx_stop[0]] = False

    if graphs:
        fig, _ = plt.subplots(1, 2, figsize=(14, 5))
        plt.subplot(121)
        plt.plot(current_binned, '-')

        plt.subplot(122)
        plt.plot(rho_y, '-', color='0.8')
        x = np.arange(len(rho_y))
        plt.plot(x[stationary_mask], rho_y[stationary_mask], 'ro')
        return stationary_mask, fig
    else:
        return stationary_mask


# -------------
def rot_start_stop(rho_y, graphs=False):
    """ Creates a mask that removes stationary data, i.e. data when the wheel is
    not rotating based on IMU rotational angle, and is used only when motor current
    data is not available.

    Parameters
    ----------
    rho_y: narray
        IMU rotational angle values for a given experiment

    Returns
    -------
    idx_start: int
        The index of the first time step when the wheel is rotating
    idx_stop: int
        The index of the last time step when the wheel is rotating

    """
    test = generic_filter(np.diff(rho_y), np.mean, size=5)

    d4 = pywt.Wavelet('haar')
    coeffs = pywt.wavedec(test, d4, mode='per')
    filteredCoeffs = copy.deepcopy(coeffs)
    keep_coefs = np.floor(len(filteredCoeffs) * 0.25)
    levels = -np.arange(1, keep_coefs + 1, dtype=int)
    for level in levels:
        filteredCoeffs[level] = np.zeros(len(filteredCoeffs[level]))
    wave1 = pywt.waverec(filteredCoeffs, d4, mode='per')
    wave_der = np.diff(wave1)

    for p in [1.5, 5.5, 0.5]:
        thresh_l, thresh_u = np.percentile(wave_der, [p, 100.-p])
        idx1 = np.where((wave_der < thresh_l) | (wave_der > thresh_u))[0]
        if len(idx1) > 50:
            break

    rho_der = np.diff(rho_y)
    x = np.arange(len(rho_der)).reshape((-1, 1))
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_2.fit(x, rho_der)
    y_2 = regr_2.predict(x)
    y_2_der = np.diff(y_2)

    der_mask = (y_2 > -1e-4) & (y_2 <= 1e-4)
    np.where((y_2_der != 0) & (der_mask[:-1]))
    idx1 = np.where(der_mask)[0]
    idx2 = np.where(y_2_der != 0)[0]

    # with the smallest derivatives prior or later
    idx_start = []
    idx_stop = []
    for k in range(len(idx2)):
        win_back = 15
        win_forw = 15
        if win_back > idx2[k]:
            continue
        if win_forw > len(y_2) - idx2[k]:
            continue
        n_back = np.sum(np.abs(y_2)[idx2[k]-win_back:idx2[k]] < 0.0001)
        n_forw = np.sum(np.abs(y_2)[idx2[k]+1:idx2[k]+win_forw+1] < 0.0001)

        if n_back == win_back and n_forw == 0:
            idx_start.append(idx2[k])
        if n_back == 0 and n_forw == win_forw:
            idx_stop.append(idx2[k])

    if len(idx_start) == 0:
        idx_start = [0]
    if len(idx_stop) == 0:
        idx_stop = [len(rho_y) - 1]

    if graphs:
        fig, _ = plt.subplots(1, 2, figsize=(14, 5))
        xnew = np.arange(len(rho_y))
        plt.subplot(121)
        plt.plot(np.hstack(test), '.')
        plt.plot(y_2, '--', color='k')
        plt.plot(y_2_der, '--', color='b')
        plt.plot(xnew[idx1], y_2[idx1], 'co')
        plt.plot(xnew[idx2], y_2[idx2], 'yo')

        plt.subplot(122)
        plt.plot(rho_y, '-', color='0.8')
        plt.plot(xnew[idx_start], rho_y[idx_start], 'ro')
        plt.plot(xnew[idx_stop], rho_y[idx_stop], 'ko')
        return idx_start, idx_stop, fig

    return idx_start, idx_stop


# --------------
contact_names = {'npix', 'contact_coords', 'contact_value', 'noncontact_value',
                 'mask_ambiguous'}


def contactAreaPG(pg_t, idx_k, imu_contact_bin, remove_lag=5, multiple=3):
    """ Computes contact area and outputs the number of pixels in the contact area,
    contact area pixel coordinates and pressure values for each pressure sensor image.

    Parameters
    ----------
    pg_t: narray
        A pressure sensor array at each given point in time
    idx_k:
        An
    imu_contact_bin : int
        Column index corresponding to the IMU rotational angle
    remove_lag: int
        The number of columns right and left from the IMU column index after
        which any contact area coordinates will be removed. Ensures that noise
        in the non-contact area is not included in the contact area.
    multiple: float
        Scaling factor for interquartile range used to determine the thresholds
        for determining the contact area pixels.

    Returns
    -------
    output: dictionary
        A dictionary that contains contact and non-contact area data. The dictionary
        keys are: \n'npix' = number of pixels in the contact area,
        \n'contact_coords' = row and column indices of the contact area pixels,
        \n'contact_value' = corresponding normalized pressure value of each pixel in the contact area,
        \n'noncontact_value' = pressure values of the non-contact pixels,
        \n'mask_ambiguous' = True/False for the ambiguous pixels if in the contact area.
    """

    output = {k: [] for k in contact_names}
    row, col = pg_t.shape

    pg_flat = np.hstack(pg_t)
    l, u = np.percentile(pg_flat, [25, 75])
    q2 = u + multiple*(u-l)
    mask_iqr = pg_flat >= q2
    mask_contact = pg_t >= q2

    #contact_idx = grid_flat[mask_iqr]
    pg_dummy = np.repeat(np.nan, 20*96).reshape((20, 96))
    pg_temp = pg_dummy[:, idx_k]
    pg_temp[mask_contact.reshape((row, col))] = 1
    pg_dummy[:, idx_k] = pg_temp
    output['contact_coords'] = np.column_stack(np.where(~np.isnan(pg_dummy)))

    output['contact_value'] = pg_flat[mask_iqr]
    output['noncontact_value'] = pg_flat[~mask_iqr]

    # remove contact too far away
    rot_lag1 = np.arange(-96/2, 0)
    rot_lag2 = np.arange(0, 96/2)
    rot_lag = np.hstack([rot_lag1, rot_lag2])

    rot_pg = rot_lag + imu_contact_bin
    rot_pg[rot_pg >= 96] = rot_pg[rot_pg >= 96] - 96
    rot_pg[rot_pg < 0] = rot_pg[rot_pg < 0] + 96

    if len(output['contact_coords']) > 0:
        map_contact = np.hstack([np.where(np.in1d(rot_pg, a))[0]
                                 for a in output['contact_coords'][:, 1]])
        lag = rot_lag[map_contact]
        contact_mask = np.abs(lag) <= remove_lag
        output['contact_coords'] = output['contact_coords'][contact_mask]
        output['noncontact_value'] = np.hstack([output['noncontact_value'],
                                                output['contact_value'][contact_mask == False]])
        output['contact_value'] = output['contact_value'][contact_mask]

    output['npix'] = len(output['contact_coords'])

    check_amb = np.in1d(output['contact_coords'][:, 1],
                        contact_area_indexes['ambiguous']).sum()
    includes_amb = check_amb != 0
    output['mask_ambiguous'].append(includes_amb)

    return output


# --------------
def contact_area_run(pg_binned, imu_contact_bin, remove_lag=5,
                     multiple=3) -> Dict[str, Dict[str, List[Union[int, np.ndarray]]]]:
    """Computes contact area for the full experiment/run.

    Parameters
    ----------
    pg_binned: narray
        Pressure sensor arrays for the full run, from alignPGtoIMu, e.g. data_binned['pg']
    imu_contact_bin: narray
        A vector of column indices corresponding to the IMU rotational angle
    remove_lag: int
        The number of columns right and left from the IMU column index after
        which any contact area coordinates will be removed. Ensures that noise
        in the non-contact area is not included in the contact area.
    multiple: float
        Scaling factor for interquartile range used to determine the thresholds
        of the contact area pixel pressure values.

    Returns
    -------
    contact_data: dictionary
        A nested dictionary with contact area data. Contact area data is separated as follows:\n
        'all' = all pixels in the contact area\n
        'grouser' = only those pixels corresponding to the grousers\n
        'nongrouser' = only those pixels corresponding to non-grousers. Within each of these, the sub-keys correspond to the
        output of contactAreaPG.
    """

    contact_data = {k: {c: [] for c in contact_names}
                    for k in contact_area_keys}
    nt = pg_binned.shape[0]

    for t in range(0, nt):
        # 'grouser', 'nongrouser', 'ambiguous'
        for k in contact_area_keys[1:]:
            pg_t = pg_binned[t, :, contact_area_indexes[k]].T
            output = contactAreaPG(
                pg_t, contact_area_indexes[k], imu_contact_bin[t], remove_lag=5, multiple=3)

            for c in contact_data[k].keys():
                contact_data[k][c].append(output[c])

        contact_data['all']['npix'].append(
            np.sum(contact_data[key]['npix'][-1] for key in contact_area_keys[1:-1]))
        contact_data['all']['mask_ambiguous'].append(np.sum([contact_data[key]['mask_ambiguous'][-1]
                                                             for key in contact_area_keys[1:-1]]) != 0)
        contact_data['all']['contact_coords'].append(np.vstack(contact_data[key]['contact_coords'][-1]
                                                               for key in contact_area_keys[1:-1]))
        contact_data['all']['contact_value'].append(np.hstack(contact_data[key]['contact_value'][-1]
                                                              for key in contact_area_keys[1:-1]))
        contact_data['all']['noncontact_value'].append(np.hstack(contact_data[key]['noncontact_value'][-1]
                                                                 for key in contact_area_keys[1:-1]))

        if contact_data['all']['contact_coords'][-1].shape[0] < 2:
            continue

    return contact_data


# -------------
def sharpPG(pg_binned: np.ndarray,
            contact_data: Dict[str, Dict[str, List[Union[int, np.ndarray]]]],
            npix_lim=15, sharp_lim=4, pg_lim=80.
            ) -> np.ndarray:
    """ Returns a vector of True/False indicating whether there is a sharp area
    underneath the pressure sensor based on the distribution of the pressure values. All three
    criteria is to be met to detemine sharpness: the contact area is smaller than
    a certain number of pixels, the number of extreme values greater than 95th percentile
    of the distribution of the contact pressure values is less or equal than a specified number
    and the mean of the pressure values is greater than a specified number.

    Parameters
    ----------
    pg_binned: narray
        Pressure sensor arrays for the full run, from alighPGtoIMU, e.g. data_binned['pg']
    contact_data: dictionary
        Dictionary of contact data from contact_area_run
    npix_lim : int, optional (default is 15)
        The smallest number of pixels in the contact area
    sharp_lim : int, optional (defaults is 4)
        Number of extreme pressure pixels in the contact area, with pressure over 95th
        percentile
    pg_lim : float, optional (default is 80)
        Pressure value maximum (normalized)

    Reuturns
    --------
    sharp: narray
        Vector of the same length as pg_binned with True corresponding to a sharp area under
        the pressure grid at a certain time step and False for otherwise

    """
    contact_idx = contact_data['all']['contact_coords']
    nt = pg_binned.shape[0]
    sharp = np.zeros((nt, ))
    for t in range(nt):
        pg_t = pg_binned[t, :, :]
        npix_sharp = 0
        high_pg = 0
        if len(contact_idx[t]) > 0:
            pg_contact = np.hstack(
                pg_t[contact_idx[t][:, 0], contact_idx[t][:, 1]])
            l, u = np.percentile(pg_contact, [5, 95])
            mask_sharp = pg_contact > u
            high_pg = pg_contact[mask_sharp]
            npix_sharp = mask_sharp.sum()

        cond1 = len(contact_idx[t]) < npix_lim
        cond2 = npix_sharp <= sharp_lim
        cond3 = np.mean(high_pg) > pg_lim
        sharp[t] = cond1 and cond2 and cond3

    return sharp


# -------------
def leanPG(pg_binned: np.ndarray,
           contact_data: Dict[str, Dict[str, List[Union[int, np.ndarray]]]],
           q2_perc=0.05) -> np.ndarray:
    """ Returns a vector of values in [0,1,-1] indicating whether there might
    be leaning left or right from the center of the pressure sensor. The leaning indicator
    is computed based on the observed contact area size to the total number of pixels
    in rows 1-5, 6-10, 11-15, 16-20 over all the columns of the contact area.

    Parameters
    ----------
    pg_binned: narray
        Pressure sensor arrays for the full run, from align_pg_to_imu(), e.g. data_binned['pg']\n
    contact_data: dictionary
        Dictionary of contact data from contact_area_run\n
    q2_perc: float

    Returns
    -------
    lean: ndarray

    """
    nt = pg_binned.shape[0]

    # detect leaning
    quads = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [
        10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    lean = np.zeros((nt, ))
    for t in range(nt):
        pg_t = pg_binned[t, :, :]

        # contact area
        cols = contact_data['all']['contact_coords'][t][:, 1]
        if len(cols) == 0:
            lean[t] = 0
            continue

        width = np.ptp(contact_data['all']['contact_coords'][t][:, 1]) + 1
        n_obs = [np.in1d(contact_data['all']['contact_coords']
                         [t][:, 0], a).sum() for a in quads]
        n_exp = width * 5.
        ratio = n_obs / n_exp
        small = ratio < 2 / width

        ratio = n_obs / n_exp
        small = ratio < 2 / width

        # contact pressure
        light = np.zeros((len(contact_area_keys[1:-1]), len(quads)))
        for r, k in enumerate(contact_area_keys[1:-1]):
            contact_idx = contact_data[k]['contact_coords'][t]
            if len(contact_idx) > 0:
                pg_contact = np.hstack(
                    pg_t[contact_idx[:, 0], contact_idx[:, 1]])
                l, u = np.percentile(pg_contact, [25, 75])
                q2 = u + 3*(u-l)
                ucol = np.unique(contact_idx[:, 1])
                if len(ucol) > 1:
                    pg_max = [np.max(pg_t[:, ucol][a, :]) for a in quads]
                    light[r] = pg_max < q2 + q2 * q2_perc
                else:
                    light[r] = 1.
            else:
                light[r] = 1.

        light = light.sum(axis=0)
        light = light == len(contact_area_keys[1:-1])

        idx_small = np.where(small)
        idx_light = np.where(light)

        if idx_small == 0 or idx_light == 0:
            # lean left
            lean[t] = 1
        elif idx_small == len(quads)-1 or idx_light == len(quads)-1:
            # lean right
            lean[t] = -1
        else:
            lean[t] = 0

    return lean


contact_area_feature_names = ['area_ratio', 'maxcol_imu_diff', 'row_wheel_ratio',
                              'contact_mean', 'contact_max',
                              'contact_min', 'contact_std', 'contact_skew',
                              'contact_kurtosis']
# -------------


def contactAreaFeatures(t, pg_binned, imu_binned, contact_data, featureList):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    t : [type]
        [description]
    pg_binned : [type]
        [description]
    imu_binned : [type]
        [description]
    contact_data : [type]
        [description]
    featureList : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    features = {}

    pg_t = pg_binned[t, :, :]
    colmax = pg_binned.max(axis=1)
    pg_z = pg_t / pg_t.max()
    for k in contact_area_keys[:-1]:

        # -----
        contact_idx_t = contact_data[k]['contact_coords'][t]

        if len(contact_idx_t) > 0:

            rimu = int(imu_binned[t])
            rmin, rmax = np.percentile(
                contact_idx_t[:, 1], [0, 100]).astype(int)

            runi = np.unique(contact_idx_t[:, 1])

            # check if wraps around the wheel --> 95 to 0
            rdiff = np.diff(runi)
            rwrap = np.where(rdiff > 80)[0]
            if len(rwrap) > 0:
                rwrap = int(rwrap) + 1
                runi = np.hstack([runi[rwrap:], runi[0:rwrap]])

            # ratio of normalized intensity
            mask_col_back = runi > rimu
            mask_col_front = runi < rimu
            mask_back = np.in1d(contact_idx_t[:, 1], runi[mask_col_back])
            mask_front = np.in1d(contact_idx_t[:, 1], runi[mask_col_front])
            back_pixels = pg_z[contact_idx_t[mask_back, 0],
                               contact_idx_t[mask_back, 1]]
            front_pixels = pg_z[contact_idx_t[mask_front, 0],
                                contact_idx_t[mask_front, 1]]

            back_area = len(back_pixels)
            front_area = len(front_pixels)

            # to avoid setting to zero
            front_area = 0.01 if front_area == 0 else front_area
            back_area = 0.01 if back_area == 0 else back_area

            # ratio of area size
            featureName = f"area_ratio_{k}"
            if featureName in featureList:
                features[featureName] = front_area / back_area

            # TODO Unused variables. Delete?
            # sep_back = rimu - runi[mask_col_back]
            # sep_front = rimu - runi[mask_col_front]
            # separation = np.nanmin(np.abs(runi - rimu))

            # -----
            max_bin = np.argmax(colmax[t, :])

            # remove contact too far away
            rot_lag1 = np.arange(-48, 0)
            rot_lag2 = np.arange(0, 48)
            rot_lag = np.hstack([rot_lag1, rot_lag2])

            rot_pg = rot_lag + rimu
            rot_pg[rot_pg >= 96] = rot_pg[rot_pg >= 96] - 96
            rot_pg[rot_pg < 0] = rot_pg[rot_pg < 0] + 96
            map_contact = np.where(np.in1d(rot_pg, max_bin))[0]

            featureName = f"maxcol_imu_diff_{k}"
            if(featureName in featureList):
                features[featureName] = int(-rot_lag[map_contact])

            # -----
            # ratio of contact rows vs the whole wheel
            featureName = f"row_wheel_ratio_{k}"
            if featureName in featureList:
                width = np.unique(contact_idx_t[:, 1]).shape[0]
                features[featureName] = width / 96.

            # -----
            contact_z = np.hstack([pg_z[a[0], a[1]] for a in contact_idx_t])
            is_contact_z = contact_z.shape[0] > 0
            feature_dict = {f"contact_std_{k}": lambda x: np.nanstd(x),
                            f"contact_max_{k}": lambda x: np.nanmax(x),
                            f"contact_min_{k}": lambda x: np.nanmin(x),
                            f"contact_mean_{k}": lambda x: np.nanmean(x),
                            f"contact_skew_{k}": lambda x: skew(x),
                            f"contact_kurtosis_{k}": lambda x: kurtosis(x)}
            for f, y in feature_dict.items():
                if f in featureList:
                    features[f] = y(contact_z) if is_contact_z else np.nan

        else:
            for c in contact_area_feature_names:
                featureName = f"{c}_{k}"
                if featureName in featureList:
                    features[featureName] = np.nan

    return features


# -------------
sink_feature_names = ['sink_mean', 'sink_std', 'sink_slope',
                      'sink_diff_angle_mean', 'sink_theta_m',
                      'sink_diff_angle_std']


def sinkFeatures(t, pg_binned, imu_binned, contact_data, featureList, tlag=200):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    t : [type]
        [description]
    data_binned : [type]
        [description]
    contact_data : [type]
        [description]
    featureList : [type]
        [description]
    tlag : int, optional
        [description], by default 200

    Returns
    -------
    [type]
        [description]
    """

    features = {}

    interval = np.arange(t-tlag, t+1) if tlag < t else np.arange(0, t+1)

    sink_smoo, sink, theta_r, theta_f, theta_m = sinkagePG(pg_binned,
                                                           imu_binned,
                                                           contact_data,
                                                           s=100,
                                                           return_theta=True)

    featureName = 'sink_theta_m'
    if featureName in featureList:
        features[featureName] = np.nanmean(theta_m[interval])

    featureName = 'sink_mean'
    if featureName in featureList:
        features[featureName] = np.nanmean(sink[interval])

    featureName = 'sink_std'
    if featureName in featureList:
        features[featureName] = np.nanstd(sink[interval])

    featureName = 'sink_diff_angle_mean'
    if featureName in featureList:
        features[featureName] = np.nanmean(
            theta_f[interval] - theta_r[interval])

    featureName = 'sink_diff_angle_std'
    if featureName in featureList:
        features[featureName] = np.nanstd(
            theta_f[interval] - theta_r[interval])

    featureName = 'sink_slope'
    if featureName in featureList:
        slope = np.polyfit(np.arange(len(interval)), sink[interval], 1)[0]
        features[featureName] = slope

    return features


# -------------
contact_area_lag_featue_names = ['npix_max_diff', 'npix_min_diff',
                                 'peak_max_std', 'peak_min_std',
                                 'smness', 'peak_max_std_cwt',
                                 'wt_jump_max', 'wt_jump_mean',
                                 'wt_min', 'wt_max', 'len_max_flat',
                                 'n_flat']


def contactAreaLaggedFeatures(t, imu_binned, contact_data, featureList, tlag=200):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    t : [type]
        [description]
    imu_binned : [type]
        [description]
    contact_data : [type]
        [description]
    featureList : [type]
        [description]
    tlag : int, optional
        [description], by default 200

    Returns
    -------
    [type]
        [description]
    """

    features = {}

    interval = np.arange(t-tlag, t+1) if tlag < t else np.arange(0, t+1)

    # -----
    imu_tlag = imu_binned[interval].astype(int)
    mask_ambiguous = ~np.in1d(imu_tlag, contact_area_indexes['ambiguous'])
    if mask_ambiguous.sum() > 0:
        npix_grouser = np.hstack(contact_data['grouser']['npix'])
        npix_grouser = npix_grouser[interval][mask_ambiguous]

        npix_nongrouser = np.hstack(contact_data['nongrouser']['npix'])
        npix_nongrouser = npix_nongrouser[interval][mask_ambiguous]

        npix_diff = npix_grouser - npix_nongrouser

        featureName = 'npix_max_diff'
        if featureName in featureList:
            features[featureName] = np.nanmax(npix_diff)

        featureName = 'npix_min_diff'
        if featureName in featureList:
            features[featureName] = np.nanmin(npix_diff)

    else:
        for featureName in ['npix_max_diff', 'npix_min_diff']:
            if featureName in featureList:
                features[featureName] = np.nan

    # -----
    for k in contact_area_keys[:-1]:
        npix = np.hstack(contact_data[k]['npix'])[interval]
        run_mean = numpy_util.running_mean(npix, 2, 2)

        # consistency and periodicity of peaks
        idx_max = argrelextrema(np.exp(npix), np.greater)
        idx_min = argrelextrema(np.exp(npix), np.less)
        peak_max_std = np.diff(run_mean[idx_min]).std()
        peak_min_std = np.diff(run_mean[idx_max]).std()

        featureName = f'peak_max_std_{k}'
        if featureName in featureList:
            features[featureName] = peak_max_std

        featureName = f'peak_min_std_{k}'
        if featureName in featureList:
            features[featureName] = peak_min_std

        # smoothness i.e. coefficient of variation
        diff_run = np.diff(npix)
        diff_mean = np.mean(diff_run)
        diff_mean = 0.1 if diff_mean == 0 else diff_mean
        smness = diff_run.std()

        featureName = f'smness_{k}'
        if featureName in featureList:
            features[featureName] = smness

        # more peak difference std and periodicity?
        peakind = find_peaks_cwt(run_mean, np.arange(1, 5))
        peakind = np.array(peakind)

        peak_max_std = np.diff(run_mean[peakind]).std() if len(peakind) > 0 \
            else np.nan

        featureName = f'peak_max_std_cwt_{k}'
        if featureName in featureList:
            features[featureName] = peak_max_std

        # jumps in the signal (high pass filter)
        coeffs = pywt.wavedec(npix, 'haar')
        coeffs_H = copy.deepcopy(coeffs)
        for z in range(4, len(coeffs_H)):
            coeffs_H[z] *= 0
        recon = pywt.waverec(coeffs_H, 'haar')
        deriv = np.diff(recon)
        idx_0 = np.where(deriv != 0)[0]
        try:
            jump_max = np.abs(np.diff(recon)).max()
        except ValueError:
            jump_max = 0
        jump_mean = np.abs(deriv[idx_0]).mean()

        featureName = f'wt_jump_max_{k}'
        if featureName in featureList:
            features[featureName] = jump_max

        featureName = f'wt_jump_mean_{k}'
        if featureName in featureList:
            features[featureName] = jump_mean

        featureName = f'wt_min_{k}'
        if featureName in featureList:
            features[featureName] = np.nanmin(recon)

        featureName = f'wt_max_{k}'
        if featureName in featureList:
            features[featureName] = np.nanmax(recon)

        # little flat areas (low pass filter)
        coeffs_flat = copy.deepcopy(coeffs)
        for z in range(0, len(coeffs_flat)-1):
            coeffs_flat[z] *= 0
        recon = pywt.waverec(coeffs_flat, 'haar')
        diff_recon = np.diff(np.where(recon == 0)[0])
        diff_idx = np.diff(np.where(diff_recon > 1)[0])
        diff_idx_ = diff_idx[diff_idx > 4]

        featureName = f'len_max_flat_{k}'
        if featureName in featureList:
            len_max_flat = diff_idx_.max() if len(diff_idx_) > 0 else 0
            features[featureName] = len_max_flat

        featureName = f'n_flat_{k}'
        if featureName in featureList:
            features[featureName] = len(diff_idx_)

    return features


# -------------
def clean_slip(slip):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    slip : array
        [description]

    Returns
    -------
    array
        [description]
    """

    mask_std = np.repeat(False, len(slip))

    if len(slip) > 10:
        slip_std = numpy_util.running_std(slip, 10, 10)
        mask_std = numpy_util.outliers_iqr(slip_std, 5)

    # mask for slip outliers
    mask_outliers = numpy_util.outliers_iqr(slip, 3)

    return mask_std & mask_outliers


# -------------
def read_all_sparse_features(feature_object: Dict[str,Dict[str,Any]],
                            response_type) -> Dict[str,Union[int, np.ndarray]]:
    """[summary]

    [extended_summary]

    Parameters
    ----------
    feature_object : [type]
        [description]
    response_type : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    data = defaultdict(list)
    f = feature_object if isinstance(
        feature_object, dict) else h5py.File(feature_object, 'r')
    features = defaultdict(list)
    features_hydration = defaultdict(list)
    response = []
    files = []

    for ftype in list(f):
        for x in f[ftype]['features'].keys():
            features[x].append(f[ftype]['features'][x][:])

        for x in f[ftype]['features_hyd'].keys():
            features_hydration[x].append(f[ftype]['features_hyd'][x][:])

        response.append(f[ftype][response_type][:])
        files.append(np.repeat(ftype, len(f[ftype][response_type][:])))

        dict_keys = list(f[ftype].keys()) if isinstance(
            feature_object, dict) else list(f[ftype])

        mask = ~np.in1d(
            dict_keys, ['features', 'features_hyd', 'files', 'names', 'exp_path'])

        for k in np.asarray(dict_keys)[mask]:
            data[k].append(f[ftype][k][:])

    for k in data.keys():
        data[k] = np.hstack(data[k])

    features = {k: np.concatenate(features[k]) for k in features.keys()}
    features_hydration = {k: np.concatenate(
        features_hydration[k]) for k in features_hydration.keys()}

    dims = [features[k].shape for k in features.keys()]
    feature_names = []
    for j in range(len(dims)):
        feature_names.extend(np.repeat(list(features)[j], dims[j][1]
                                       if len(dims[j]) > 1 else 1))

    dims = [features_hydration[k].shape for k in features_hydration.keys()]
    feature_hyd_names = []
    for j in range(len(dims)):
        feature_hyd_names.extend(np.repeat(list(features_hydration)[j],
                                           dims[j][1] if len(dims[j]) > 1 else 1))

    if response_type == 'rock':
        response = np.hstack(response)
        response[response > 1] = 1
    elif response_type == 'patterns':
        response = np.hstack(response).astype(str)
        keys = np.unique(response).astype(str)
        for i in range(len(keys)):
            response[response == keys[i]] = terrain_dict[keys[i]]
        response = response.astype(int)
    elif response_type == 'material':
        response = np.hstack(response).astype(str)
        keys = np.unique(response).astype(str)
        for i in range(len(keys)):
            response[response == keys[i]] = composition_dict[keys[i]]
        response = response.astype(int)
    elif response_type == 'hydration':
        response = np.hstack(response).astype(np.string_)
    # TODO What if 'response_type' is not one of these options?

    data['y'] = np.hstack(response)
    data['files'] = np.hstack(files)
    # print(feature_names)
    data['feature_names'] = np.hstack(feature_names)
    data['nF'] = len(features.keys())
    data['X'] = np.column_stack(features.values())
    data['X_hyd'] = np.column_stack(features_hydration.values())
    data['feature_names_hyd'] = np.hstack(feature_hyd_names)

    if isinstance(feature_object, str):
        f.close()

    return data


# -------------
def selectFeatures(data, rfe_file=None, rfe_key=None, preset=True, names_to_exclude=None):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data : [type]
        [description]
    rfe_file : [type], optional
        [description], by default None
    rfe_key : [type], optional
        [description], by default None
    preset : bool, optional
        [description], by default True
    names_to_exclude : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """

    if rfe_file is not None:
        with closing(h5py.File(rfe_file, 'r')) as f:
            features_tochoose = f[rfe_key][:]
        name_mask = np.in1d(data['feature_names'], features_tochoose)

    else:
        if preset:
            features_toexclude = ['bbox_area',
                                  'moments_hu', 'moments_normalized', 'weighted_centroid',
                                  'weighted_moments_hu', 'weighted_moments_normalized']
        else:
            features_toexclude = ['None'] if not names_to_exclude \
                else names_to_exclude

        idx = [np.flatnonzero(np.core.defchararray.find(data['feature_names'], x) != -1)
               for x in features_toexclude]
        idx = np.unique(np.hstack(idx))
        name_mask = ~np.in1d(data['feature_names'], data['feature_names'][idx])

    data['X'] = data['X'][:, name_mask]
    data['feature_names'] = data['feature_names'][name_mask]
    data['nF'] = len(data['feature_names'])

    return data


# -------------
def remove_nan(data: Dict[str, np.ndarray], response_type: str) \
    -> Tuple[Dict[str,np.ndarray], np.ndarray]:
    """Filters out all data points in which X or Y is NaN or Inf.

    [extended_summary]

    Parameters
    ----------
    data : [type]
        [description]\n
    response_type : str
        Rock, patterns, material, slip or hydration

    Returns
    -------
    Tuple[Dict[str,np.ndarray], np.ndarray]
        Data with filtered according to X, y and slip data.
        Numpy array of valid indices.
    """

    if not data:
        logger.warning("No data provided")
        return None

    keysNan = ['X', 'X_hyd', 'y', 'rock_depth', 'sink', 'slip', 'files', 'imu_contact_bin',
               'material', 'patterns']

    # check predictors
    check_rows = np.sum(data['X'], axis=1)
    mask_X = ~np.isnan(check_rows) & ~np.isinf(check_rows)

    # check response
    is_not_hydration = response_type != 'hydration'
    mask_y = ~np.isnan(data['y']) & ~np.isinf(
        data['y']) if is_not_hydration else np.repeat(True, len(mask_X))

    # check response for outliers in continuous response
    # change if more than 10 classes
    is_more_than_10 = len(np.unique(data['y'])) > 10
    mask_outliers = clean_slip(
        data['y']) if is_more_than_10 else np.repeat(True, len(mask_X))

    mask = mask_X & mask_y & mask_outliers
    for k in keysNan:
        data[k] = data[k][mask]

    return data, mask


# -------------
def sinkagePG(pg_binned, imu_binned, contact_data, s=100, imu_contact_bin=None,
              return_theta=False):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    pg_binned : [type]
        [description]
    imu_binned : [type]
        [description]
    contact_data : [type]
        [description]
    s : int, optional
        [description], by default 100
    imu_contact_bin : [type], optional
        [description], by default None
    return_theta : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    imu_bins = imu_binned if not imu_contact_bin else imu_contact_bin

    nt = pg_binned.shape[0]

    # CHECK the correct params!
    wheel_radius = 509. / 2
    wheel_circum = 1598.
    wp = wheel_circum / 96.

    sinkage = np.zeros((nt, ))
    theta_rs = np.zeros((nt, ))
    theta_fs = np.zeros((nt, ))
    theta_m = np.zeros((nt, ))
    wfmm = np.zeros((nt, ))
    for t in range(nt):
        contact_cols = contact_data['all']['contact_coords'][t][:, 1]
        if len(contact_cols) == 0:
            continue
        width = np.unique(contact_cols).shape[0]

        ucols = np.unique(contact_cols)
        imu_bin = imu_bins[t]
        mcol = np.argmax(pg_binned[t, :, :][:, ucols].max(axis=0))
        max_bin = ucols[mcol]

        # angle corresponding to contact arc
        # between imu and the front of the wheel
        # the pressure grid moves from lower bin indices
        # to higher bin indices 96 --> 0
        idx_r = ucols < imu_bin
        w_r = ucols[idx_r].shape[0]

        # convert to mm
        w_r_mm = wp * w_r
        wfmm[t] = w_r_mm

        #angle in radians
        theta_r = w_r_mm / wheel_radius

        # between imu and the back of the wheel
        idx_f = ucols >= imu_bin
        w_f = ucols[idx_f].shape[0]
        w_f_mm = wp * w_f
        theta_f = w_f_mm / wheel_radius

        # forward (clockwise) coordinates (x,y)
        x_f = wheel_radius * np.sin(np.pi - theta_f)
        y_f = wheel_radius * np.cos(np.pi - theta_f)

        # backward (clockwise) coordinates (x, y)
        x_r = wheel_radius * np.sin(np.pi + theta_r)
        y_r = wheel_radius * np.cos(np.pi + theta_r)

        # sinkage
        sinkage[t] = -wheel_radius - y_f
        theta_rs[t] = theta_r
        theta_fs[t] = theta_f

        contact_value = contact_data['all']['contact_value'][t]
        idx_max_value = np.argmax(contact_value)
        max_bin = contact_cols[idx_max_value]

        # angle between max contact value bin and imu bin
        #idx_max = max_bin < imu_bin
        w_max = imu_bin - max_bin
        w_f_mm = wp * w_max
        theta_max = w_f_mm / wheel_radius
        theta_m[t] = theta_max

    # 100 data point smoothing for a quarter-wheel period
    sinkage_smoothed = numpy_util.running_mean(sinkage, s, 0)

    if return_theta:
        return sinkage_smoothed, sinkage, theta_rs, theta_fs, theta_m
    return sinkage_smoothed, sinkage


# -------------
def contact_area_tsfresh_features(t, data_binned, contact_data, featureList, tlag=200):
    """ Uses the fetaure functions that were extracted from tsfresh to compute features from the
        contact area time series data

    Parameters
    ----------
    Returns
    -------
    """

    features = {}

    interval = np.arange(t-tlag, t+1) if tlag < t else np.arange(0, t+1)

    for k in contact_area_keys[:-1]:
        npix = np.hstack(contact_data[k]['npix'])[interval]

        # ratio beyond sigma
        for ratio in [0.5, 1, 1.5, 2, 2.5, 3]:
            featureName = f"ratio_beyond_r_sigma_{ratio}_{k}"
            if featureName in featureList:
                features[featureName] = feat_cal.ratio_beyond_r_sigma(
                    npix, ratio)

        # ratio of reoccuring data points to all data points
        featureName = f"percentage_of_reoccurring_datapoints_to_all_{k}"
        if featureName in featureList:
            features[featureName] = feat_cal.percentage_of_reoccurring_datapoints_to_all_datapoints(
                npix)

        # quantiles
        q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(0, 9):
            featureName = f"quantile_{q[i]}_{k}"
            if featureName in featureList:
                # TODO These features all have the same value.
                # Is this the intended behavior?
                features[featureName] = np.quantile(npix, q)

        featureName = f"longest_strike_below_mean_{k}"
        if featureName in featureList:
            features[featureName] = feat_cal.longest_strike_below_mean(npix)

        featureName = f"longest_strike_above_mean_{k}"
        if featureName in featureList:
            features[featureName] = feat_cal.longest_strike_above_mean(npix)

        # abs sum of changes
        featureName = f'absolute_sum_of_changes_{k}'
        if featureName in featureList:
            features[featureName] = feat_cal.absolute_sum_of_changes(npix)

        # energy ratio by chunks
        num_segments = 10
        energy_rat_chunks = [("", 0.)]*num_segments if not np.any(npix) \
            else feat_cal.energy_ratio_by_chunks(npix,
                                                 [{'num_segments': num_segments, 'segment_focus': v} for v in range(num_segments)])
        for i, energy_chunk in enumerate(energy_rat_chunks):
            featureName = f"energy_ratio_by_chunks_{i}_{k}"
            if featureName in featureList:
                energy_ratio = energy_chunk[1]
                features[featureName] = energy_ratio

        # Symmetry Looking
        """
        Boolean variable denoting if the distribution of npix *looks symmetric*. This is the case if
        |mean(npix)-median(npix)| < r * (max(npix)-min(npix)) with r = 0.95
        """
        featureName = f"symmetry_looking_{k}"
        if featureName in featureList:
            s = feat_cal.symmetry_looking(npix, [{"r": 0.95}])[0][1]
            features[featureName] = 1 if s else 0

        # absolute change variance quantiles
        """
        Fixes a corridor between two quantiles (0-0.8) of the contact data time series. Then calculates the
        absolute and non-absolute variance of the consecutive changes of the time series inside this corridor.
        """
        featureName = f"change_variance_quantiles_abs_{k}"
        if featureName in featureList:
            features[featureName] = feat_cal.change_quantiles(npix, ql=0.0,
                                                              qh=0.8,
                                                              isabs=True,
                                                              f_agg="var")

        # non-absolute change variance quantiles
        """
        Fixes a corridor between two quantiles (0-0.8) of the contact data time series. Then calculates the
        absolute and non-absolute variance of the consecutive changes of the time series inside this corridor.
        """
        featureName = f"change_variance_quantiles_nonabs_{k}"
        if featureName in featureList:
            features[featureName] = feat_cal.change_quantiles(npix, ql=0.0,
                                                              qh=0.8,
                                                              isabs=False,
                                                              f_agg="var")
        # variance
        featureName = f"variance_{k}"
        if featureName in featureList:
            features[featureName] = np.nanvar(npix)

    return features


imu_unwrap_feature_names = ['unwrap_mean', 'unwrap_std']
def IMUunwrapFeatures(t, pg_binned, imu_binned, time_binned, contact_data,
                      pglag, featureList, tlag=200):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    t : [type]
        [description]
    pg_binned : [type]
        [description]
    imu_binned : [type]
        [description]
    time_binned : [type]
        [description]
    contact_data : [type]
        [description]
    pglag : [type]
        [description]
    featureList : [type]
        [description]
    tlag : int, optional
        [description], by default 200

    Returns
    -------
    [type]
        [description]
    """

    features = {}
    interval = np.arange(t-tlag, t+1) if tlag <= t \
         else np.arange(0, t+1)

    imu_tlag = imu_binned[interval].astype(int)
    mask_ambiguous = ~np.in1d(imu_tlag, contact_area_indexes['ambiguous'])
    for k in contact_area_keys[1:-1]:
        mask_ambiguous = ~np.in1d(imu_tlag, contact_area_indexes[f"{k}_noamb"])
        if mask_ambiguous.sum() > 0:
            pg_imu, _, _ = unwrap_pg_to_imu(pg_binned[interval, :, :],
                                            imu_binned[interval],
                                            time_binned[interval],
                                            pglag, extraction=f"{k}_noamb")

            npix = np.hstack(contact_data[k]['npix'])
            npix = npix[interval][mask_ambiguous]

            all_lag_max = np.max([a.max() for a in pg_imu])
            for l in range(len(pg_imu)):
                pg_norm = pg_imu[l] / all_lag_max
                m_lag = np.mean(pg_norm)
                std_lag = np.std(pg_norm)

                featureName = f"unwrap_mean{pglag[l]}_{k}"
                if featureName in featureList:
                    features[featureName] = m_lag

                featureName = f"unwrap_std{pglag[l]}_{k}"
                if featureName in featureList:
                    features[featureName] = std_lag

        else:
            for l in range(len(pglag)):
                featureName = f"unwrap_mean{pglag[l]}_{k}"
                if featureName in featureList:
                    features[featureName] = np.nan

                featureName = f"unwrap_std{pglag[l]}_{k}"
                if featureName in featureList:
                    features[featureName] = np.nan

    return features


imu_image_geo_feature_names = ['area', 'max_intensity', 'solidity',
                               'perimeter', 'convex_area',
                               'eccentricity', 'orientation',
                               'min_intensity', 'major_axis_length',
                               'minor_axis_length', 'extent',
                               'filled_area', 'equivalent_diameter',
                               'mean_intensity', 'euler_number']
def IMUunwrapImageGeoFeatures(t, pg_binned, imu_binned, time_binned,
                              imu, contact_data, pglag, featureList,
                              tlag=200):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    t : [type]
        [description]
    data_binned : [type]
        [description]
    contact_data : [type]
        [description]
    pglag : [type]
        [description]
    featureList : [type]
        [description]
    tlag : int, optional
        [description], by default 200

    Returns
    -------
    [type]
        [description]
    """

    # 'equivalent_diameter', 'euler_number' - to ADD (with dims 1 and 1)
    # shapes have to match the names above! if adding look up dimensions.
    shapes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    subnames = ['mean']

    features = {}
    interval = np.arange(t-tlag, t+1) if tlag <= t else np.arange(0, t+1)

    imu_tlag = imu_binned[interval].astype(int)
    mask_ambiguous = ~np.in1d(imu_tlag, contact_area_indexes['ambiguous'])
    for k in contact_area_keys[1:-1]:
        mask_ambiguous = ~np.in1d(imu_tlag, contact_area_indexes[k + '_noamb'])
        if mask_ambiguous.sum() > 0:
            pg_imu, _, _ = unwrap_pg_to_imu(pg_binned[interval, :, :],
                                            imu[interval],
                                            time_binned[interval],
                                            pglag, extraction=k + '_noamb')

            # normalize PG
            '''
             TODO Should maybe use amax(initial=) here in case of
                  empty array?
             '''
            all_lag_max = np.max([np.max(a) for a in pg_imu])
            pg_norm = [None] * len(pg_imu)
            for l in range(len(pg_imu)):
                pg_norm[l] = pg_imu[l] / all_lag_max

            dn = pg_imu[0].shape[1]
            w = pywt.dwt_max_level(dn, pywt.Wavelet('db2'))
            pg_recon = waveletDegrouse(
                pg_norm, wtype='db2', level=w if w < 2 else 2)

            area = []
            binary = [[]] * len(pg_recon)
            labels = [[]] * len(pg_recon)
            for l in range(len(pg_recon)):
                recon_image = pg_recon[l]
                binary[l] = recon_image > threshold_otsu(recon_image)
                labels[l] = measure.label(
                    binary[l], background=0, connectivity=1)
                props = measure.regionprops(
                    labels[l], intensity_image=recon_image)

                temp = np.hstack([r['area'] for r in props])
                area.append(len(temp))

                # by size
                max_area = np.argsort(temp)[::-1] + 1
                mask_area = np.sort(temp)[::-1] >= 10

                mask = np.in1d(labels[l], max_area[mask_area]).reshape(
                    labels[l].shape)
                labels[l][~mask] = 0
                binary[l][~mask] = 0

                props = measure.regionprops(
                    labels[l], intensity_image=recon_image)
                n = len(props)
                for m in range(len(imu_image_geo_feature_names)):
                    if n > 0:
                        # 'solidity1_lag3_std_grouser'
                        prop = [np.ravel(r[imu_image_geo_feature_names[m]])
                                for r in props]
                        prop = np.row_stack(prop)
                        prop[np.isnan(prop)] = 0

                        v1 = np.nanmean(prop, axis=0).astype(np.float32)
                        v3 = np.nanstd(prop, axis=0).astype(np.float32)

                        if np.isinf(v1).sum() > 1:
                            v1 = np.repeat(0, shapes[m])
                        if np.isinf(v3).sum() > 1:
                            v3 = np.repeat(0, shapes[m])
                    else:
                        v1 = np.repeat(0, shapes[m])
                        v3 = np.repeat(0, shapes[m])

                    for w in range(len(v1)):
                        featureName = (f"{imu_image_geo_feature_names[m]}"
                                       f"{w+1}_lag{pglag[l]}_"
                                       f"{subnames[0]}_{k}"
                                       )
                        if featureName in featureList:
                            features[featureName] = [v1[w]]

    return features
