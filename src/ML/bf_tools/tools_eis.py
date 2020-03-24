
'''
Author: Jack Lightholder
Date  : 12/12/17

Brief : IDF file management

Notes :
'''

import argparse
import io
import math
import os
import re
import shutil
import sys
from datetime import datetime
from os import listdir
from os.path import isdir, isfile, join
from typing import List, Dict

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt

import numpy as np

from bf_config import bf_globals
from bf_logging import bf_log
from bf_util import numpy_util, trig_util

bf_log.setup_logger()
logger = logging.getLogger(__name__)
# TODO What is the significance of these frequencies?
old_freqs = [10000, 1000, 100, 10]
freqs_new = [10550.1, 947.86, 111.304, 10.0]
hyd_features = [f'{key}_{s}' for key in ['amplitude', 'phase']
                for s in freqs_new] + ['rot_angle']


def parse_idf(filePath):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    filePath : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if not os.path.isfile(filePath):
        logger.info(f"{filePath} not found")
        return None

    data = []
    with io.open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    dictionary = {}
    times = []
    pretreat_real = []
    pretreat_imag = []
    pretreat_freq = []
    primary_real = []
    primary_imag = []
    primary_freq = []

    logging.info(f"Parsing {filePath}")
    for x in range(0,len(data)):

        line = data[x].rstrip().rstrip("\r\n")
        match = None

        regex = r"^Peaks=(true|false)$"
        match = re.match(regex, line)
        if match is not None:
            logging.debug(f"Peaks={match.group(1)}")
            dictionary["Peaks"] = match.group(1)

        regex = r"^timing=(\d+)$"
        match = re.match(regex, line)
        if match is not None:
            logging.debug(f"timesteps={match.group(1)}")
            dictionary["timesteps"] = match.group(1)

        regex = r"^starttime=(\d{1,2}/\d{1,2}/\d{4}) (\d{1,2}):(\d\d):(\d\d) (AM|PM)$"
        match = re.match(regex, line, flags=re.ASCII)
        if match is not None:
            logging.debug(f"starttime={line}")
            day = match.group(1)
            hour = int(match.group(2))
            minutes = match.group(3)
            seconds = match.group(4)
            if match.group(5) == "PM" and hour != 12:
                hour = hour + 12
            if match.group(5) == "AM" and hour == 12:
                hour = 0
            combined_time = f"{day} {hour}:{minutes}:{seconds}"
            dt = datetime.strptime(combined_time,'%m/%d/%Y %H:%M:%S')
            utc = (dt-datetime.utcfromtimestamp(0)).total_seconds()
            dictionary["exp_time"] = utc

        regex = r"^timing\.time\[\d+\]=(\d+\.\d+)$"
        match = re.match(regex, line, flags=re.ASCII)
        if(match is not None):
            logging.debug(f"timing.time={match.group(1)}")
            times.append(float(match.group(1)))

        regex = r"^(primary_data|pretreatmentdata)$"
        match = re.match(regex,line)
        if match is not None:
            logging.debug(f"Found {match.group(1)}")
            primary_start = x + 3
            numSamples = int(data[x+2].rstrip("\r\n"))
            primary = data[primary_start:primary_start+numSamples]
            data_type = match.group(1)
            for line in primary:
                split = line.split()
                if data_type == "primary_data":
                    dictionary['times'] = times
                    keys = ['real_primary', 'imaginary_primary', 'frequency_primary']
                else:
                    keys = ['real_pretreat', 'imaginary_pretreat', 'frequency_pretreat']
                for i in range(len(keys)):
                    dictionary.setdefault(keys[i],[]).append(float(split[i]))

    # Only 'timing' is allowed to be empty
    return dictionary


### ---------------------------------------
def read_eis(files_eis: List[str], time_binned: np.ndarray, imu_binned = None, tray = False) -> Dict:
    """[summary]

    [extended_summary]

    Parameters
    ----------
    files_eis : list
        [description]
    imu_binned : [type], optional
        [description], by default None
    time_binned : ndarray, optional
        [description], by default None
    tray : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    eis_data = {'RE':[],'IM':[],'freq':[], 'hyd':[], 'mat':[], 'rot': []}
    time_eis = []
    # Sorting should ensure time_eis is monotonically increasing
    for idffile in sorted(files_eis):
        data = parse_idf(idffile)
        if data['timesteps'] == '0':
            continue
        time_eis.append(data['exp_time'])

        if not tray:
            freqs = list(old_freqs) # TODO Why these frequencies?
            #check if frequencies are old or new
            check_freq = np.in1d(freqs, data['frequency_primary'])
            check_freq_new = np.in1d(freqs_new, data['frequency_primary'])
            num_freq_new = check_freq_new.sum()
            num_freq_old = check_freq.sum()
            if num_freq_new >= num_freq_old: # Use only new frequencies
                freqs = list(freqs_new)
                check_freq = check_freq_new

            nfreq = len(freqs)
            if check_freq.sum() < nfreq:
                for k in list(data)[3:]: # TODO Why start at the fourth element?
                    temp = np.array([np.nan] * nfreq)
                    logger.debug(f"len(data[{k}]) = {len(data[k])}")
                    temp[check_freq] = data[k]
                    data[k] = temp


        eis_data['RE'].append(data['real_primary'])
        eis_data['IM'].append(data['imaginary_primary'])
        eis_data['freq'].append(data['frequency_primary'])

        if tray:
            hyd = idffile.split('/')[-2].split('_')[1]
            mat = idffile.split('/')[-2].split('_')[0]
            rot = idffile.split('/')[-2].split('_')[4]
        else:
            hyd = idffile.split('/')[-3].split('_')[15]
            mat = idffile.split('/')[-3].split('_')[13]

            tdiff = data['exp_time'] - time_binned
            smin = np.argmin(np.abs(tdiff))
            rot = imu_binned[smin]

        eis_data['hyd'].append(float(hyd))
        eis_data['mat'].append(mat)
        eis_data['rot'].append(int(rot))


    eis_data_binned = {}
    if len(time_eis) == 0:
        logger.info('Problem with EIS readings')
        for k in list(eis_data) + ['amp'] + ['phase']:
            eis_data[k] = np.repeat(np.nan, len(files_eis))
            eis_data_binned[k] = np.repeat(np.nan, len(time_binned))

    else:
        eis_data = trig_util.add_polar(eis_data)
        eis_data = numpy_util.numpyify(eis_data)
        match_idx = np.digitize(time_binned, time_eis)

        for k in list(eis_data):
            dims = np.array(eis_data[k].shape)
            dims[0] = len(time_binned)
            eis_data_binned[k] = np.zeros((dims))

            if eis_data[k].dtype != float:
                eis_data_binned[k] = eis_data_binned[k].astype(np.string_)
                for b in np.unique(match_idx):
                    eis_data_binned[k][match_idx == b] = eis_data[k][b-1]
            else:
                for b in np.unique(match_idx):
                    eis_data_binned[k][match_idx == b] = eis_data[k][b-1]

        eis_data_binned['rot'] = imu_binned.astype(int)

    return eis_data, eis_data_binned


def collect_hyd_features(t, freq_binned, amp_binned, phase_binned, imu,
                        contact_data, featureList, tlag=200):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    t : [type]
        [description]
    freq_binned : [type]
        [description]
    amp_binned : [type]
        [description]
    phase_binned : [type]
        [description]
    imu : [type]
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

    subnames = freqs_new.copy()

    #shapes have to match the names above! if adding look up dimensions.
    mask_freq = np.isnan(freq_binned)
    features = {}

    if mask_freq.sum() == len(freq_binned):
        for s in subnames:
            featureName = 'amplitude_' + str(s)
            if featureName in featureList:
                features[featureName] = np.nan

            featureName = 'phase_' + str(s)
            if featureName in featureList:
                features[featureName] = np.nan

    else:
        ufreqs = np.unique(freq_binned[~mask_freq])
        interval = np.arange(t-tlag, t+1) if tlag <= t\
                                          else np.arange(0, t+1)
        for s in subnames:
            idx = np.abs(ufreqs - s).argmin()

            featureName = 'amplitude_' + str(s)
            if featureName in featureList:
                features[featureName] = np.nanmean(
                    amp_binned[interval, idx])

            featureName = 'phase_' + str(s)
            if featureName in featureList:
                features[featureName] = np.nanmean(
                    phase_binned[interval, idx])

    featureName = 'rot_angle'
    if featureName in featureList:
        features[featureName] = imu[t]

    return features
