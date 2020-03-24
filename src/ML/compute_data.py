# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:23:52 2018

@author: marchett, jackal
"""
import argparse
import glob
import logging
import multiprocessing
import os
import sys
from typing import List
from contextlib import closing

import h5py
import numpy as np


from bf_config import bf_globals
from bf_logging import bf_log
from bf_tools import tools_eis, tools_pg
from bf_util import h5_util, numpy_util


bf_log.setup_logger()
logger = logging.getLogger(__name__)

np.seterr(divide='ignore', invalid='ignore')

# TODO Should be in a config file
compute_data_feature_names = ['amp', 'current', 'exp_path', 'freq',
                              'ft_xyz', 'hyd', 'imu', 'imu_bin',
                              'material', 'pattern', 'pg', 'phase',
                              'rock', 'rock_depth', 'sink', 'slip',
                              'slip_fiducials', 'time']


def unify_experiment_single(args):
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
    datadir = args["datadir"]
    i = args["i"]
    files = args["files"]
    no_contact_dir = args["no_contact_dir"]
    logger.debug(f"Processing experiment {i} of {len(files)}")
    return unify_experiment_single_single_thread(files[i], datadir, no_contact_dir)


def unify_experiment_single_single_thread(experiment_path, datadir, no_contact_dir="cal/Xiroku_no_contact"):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    experiment_path : [type]
        [description]\n
    datadir : [type]
        [description]\n
    no_contact_dir : [type]
        [description]\n

    Returns
    -------
    [type]
        [description]
    """
    logger.debug(
        f"Processing experiment file {experiment_path.split('/')[-1]}")
    ftype, data, time = tools_pg.read_all_pg_data(
        experiment_path, f"{datadir}/{no_contact_dir}/")

    logger.debug(f"Experiment Metadata: {ftype}")
    data_binned, time_binned = tools_pg.align_pg_to_imu(
        data, time, bf_globals.T, bf_globals.R)

    rot_mask = tools_pg.rot_start_stop_motor(data_binned['current'], data_binned['imu'])
    time_binned=time_binned[~rot_mask]
    for k in data_binned.keys():
        data_binned[k]=data_binned[k][~rot_mask]
    imu_contact_bin=tools_pg.contact_imu(data_binned['imu'])

    logger.info(f"Data points after align/clean: {len(imu_contact_bin)}")

    files_eis=glob.glob(f"{experiment_path}/**/*.idf", recursive=True)
    _, eis_binned=tools_eis.read_eis(files_eis,
                                     time_binned,
                                     data_binned['imu'])

    # Generate return dictionary to be merged with results from other experiment threads
    exp_path=np.array([experiment_path.replace(datadir, "")])
    result={"time_binned": time_binned, "imu_contact_bin": imu_contact_bin,
              "data_binned": data_binned, "ftype": ftype, "exp_path": exp_path,
              'eis_binned': eis_binned}

    return result

def unify_experiment_data(datadir: str, subfolders: List[str], regex: str,
                  outputfile: str, fileList=None, multithreading=False):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    datadir : string
        Path to data repository.  Ex. /Volumes/MLIA_active_data/data_barefoot/.\n
    subfolders : list
        List of relative folder paths to datadir for regex to search in for relevant experiment folders\n
    regex : [type]
        [description]\n
    outputfile : [type]
        [description]\n
    fileList : [type], optional
        [description], by default None\n
    multithreading : bool, optional
        [description], by default False\n

    Returns
    -------
    dictionary
        [description]
    """
    # Confirm datadir exists.  Protects against unmounted MLIA_ACTIVE_DATA volume.
    if not os.path.exists(datadir):
        error_msg="Data directory does not exist.  Point to valid data directory to continue."
        logger.error(error_msg)
        sys.exit(error_msg)

    # Outfile argument is required for saving h5.
    if outputfile is None:
        error_msg="Outfile file path is a required argument. Specify using --outfile or visit --help"
        logger.error(error_msg)
        sys.exit(error_msg)

    if fileList:
        files=fileList
    else:
        logger.info(f"Data root: {datadir}")
        logger.info(f"Subdirs: {','.join(subfolders)}")
        logger.info(f"Regex: {str(regex)}")
        # Apply subfolder and regex combo to determine which experiments should be read.
        # NOTE A path name ending with '/' will return directory names, not files.
        files=[glob.glob(f"{datadir}/{a}/{b}/")
                         for a, b in zip(subfolders, regex)]
        if not files:
            error_msg="Glob resulted in no experiment files. Exiting."
            logger.warning(error_msg)
            sys.exit(error_msg)

        # logger.debug("Files to read: \n" + "\n".join(str(x) for x in files))
        for subfiles, subfolder in zip(files, subfolders):
            if not subfiles:
                logger.warning(
                    f'No files found for rock data merge in {str(subfolder)} subfolder')

    # Generate file list
    files=np.sort(np.hstack(files))
    logger.debug(f"Number of experiments to compute = {len(files)}")

    if multithreading:
    # Set up list of argument dicts to be passed into unify_experiment_single.
    #   Needed since input arguments >1
        experiments=[]
        for i in range(0, len(files)):
            experiments.append({"i": i, "datadir": datadir, "files": files,
                               "no_contact_dir": 'cal/Xiroku_no_contact'})

        # Multithreaded launch of experiment mergers, returns list of result dicts
        with multiprocessing.Pool(processes=bf_globals.THREADS) as pool:
            results=pool.map(unify_experiment_single, experiments)

    else:
        # Single thread implementation
        # Flatten file list
        results=[unify_experiment_single_single_thread(experiment_path, datadir)
                            for experiment_path in files]

    # Decompose results list and save into H5 and return dictionary
    return_dictionary={}
    for result in results:

        return_dictionary[result['ftype']]={}

        for k in result['data_binned'].keys():
            try:
                return_dictionary[result['ftype']][k]=result['data_binned'][k]
            except:
                return_dictionary[result['ftype']
                    ][k]=result['data_binned'][k].astype(np.string_)

        return_dictionary[result['ftype']]["time"]=result['time_binned']
        return_dictionary[result['ftype']]['imu_bin']=result['imu_contact_bin']
        return_dictionary[result['ftype']
            ]['exp_path']=result['exp_path'].astype(np.string_)
        return_dictionary[result['ftype']]['amp']=result['eis_binned']['amp']
        return_dictionary[result['ftype']
            ]['phase']=result['eis_binned']['phase']
        return_dictionary[result['ftype']]['hyd']=result['eis_binned']['hyd']
        return_dictionary[result['ftype']]['freq']=result['eis_binned']['freq']

    if os.path.isfile(outputfile):
        os.remove(outputfile)
    logger.debug(f'Saving data to {outputfile}')
    h5_util.save_h5(outputfile, return_dictionary)
    return return_dictionary

if __name__ == "__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument('--datadir',      default="/Volumes/MLIA_active_data/data_barefoot/",
                                          help="Path to data repository. Defaults to /Volumes/MLIA_active_data/data_barefoot/")

    parser.add_argument('--subfolders',   nargs='+',
                                          help="Options include: rock_detection, composition, data_andrew")

    parser.add_argument(
        '--outfile',      help="Full path, and name, of the resulting unified data file")

    parser.add_argument('--regex',        default="*",
                                          help="list of regex strings. Ex: --regex '*'")
    # TODO Should there be an option for multithreading?

    args=parser.parse_args()

    data=unify_experiment_data(
        args.datadir, args.subfolders, args.regex, args.outfile)
