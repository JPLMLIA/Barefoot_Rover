# -*- coding: utf-8 -*-
"""
Created on Thurs January 10 12:20:50 2019

@author: jackal
"""
import argparse
import logging
import os
import sys
from contextlib import closing

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import predict_model
import compute_data
import compute_features
from bf_config import bf_globals
from bf_logging import bf_log
from bf_tools import tools_pg

bf_log.setup_logger()
logger = logging.getLogger(__name__)

def streaming_system():

    sparse_, features, nan_mask = compute_features.generate_features(datadir, outputdir, data_file,
                                                                     feature_file, burn_in, False,
                                                                     module, multithreading, featuresFile)
    logger.info(f"Run feature matrix shape: {features['slip'].shape}")

    logger.info("Predicting - ")
    

def run_system(datadir, outputdir, run, burn_in, plotFile, module,
               featuresFile, multithreading=False):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    datadir : [type]
        [description]
    outputdir : [type]
        [description]
    run : [type]
        [description]
    burn_in : [type]
        [description]
    plotFile : [type]
        [description]
    module : [type]
        [description]
    featuresFile : [type]
        [description]
    multithreading : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if not os.path.exists(datadir):
        error_msg = "Data directory does not exist.  \
                     Point to valid data directory to continue."
        logger.error(error_msg)
        sys.exit(error_msg)

    if not os.path.exists(outputdir):
        logger.info(f"Output directory does not exist. Creating {outputdir}")
        os.makedirs(outputdir)

    logger.info("Merging Test Data")
    file = [f"{datadir}/{run}"]
    data_file = os.path.join(outputdir, 'data.h5')

    logger.debug(f"Computing data in {datadir}")
    data = compute_data.unify_experiment_data(
        datadir, None, None, data_file, file, multithreading)
    run_name = list(data.keys())[0]
    data = data[run_name]

    try:
        low_burn = int(burn_in[0])
        logger.info("Leading burn in set to " + str(low_burn))
    except:
        low_burn = 0
        logger.warning("Leading burn value of " + str(burn_in[0]) +
                       " couldn't be parsed.  Set to default of 0")

    try:
        high_burn = int(burn_in[1])
        logger.info("Trailing burn in set to " + str(high_burn))
    except:
        high_burn = None
        logger.warning("Leading burn value of " + str(burn_in[1]) +
                       " couldn't be parsed.  Set to default of None")

    nt = data['pg'].shape[0]
    lower_t = nt-1 if low_burn >= nt else low_burn

    mask_burnin = np.arange(nt) >= lower_t
    for key, value in data.items():
        if key != "exp_path":
            data[key] = value[mask_burnin]

    feature_file = f"{outputdir}/features"
    logger.debug(f"Generating features in {datadir} using {data_file}")
    _, features, nan_mask = compute_features.\
                                  generate_features(datadir, outputdir,
                                                    data_file,feature_file,
                                                    burn_in, False,
                                                    module, multithreading,
                                                    featuresFile)
    logger.info(f"Run feature matrix shape: {features['slip'].shape}")

    logger.info("Predicting - ")

    for key, value in data.items():
        if key != "exp_path":
            data[key] = value[nan_mask]

    has_feature_data = nan_mask.sum() != 0
    if not has_feature_data:
        logger.warning('Predictions cannot be generated, empty feature \
                        array!')
    else:
        logger.info(f"Number of features computed: "
                    f"{str(features['feature_names'].shape)}")

    slip_array = predict_model\
        .predict_slip(f"{datadir}/{bf_globals.SLIP_MODEL}",
                      f"{datadir}/{bf_globals.SLIP_FEATURE_FILE}",
                      features) \
        if has_feature_data else np.full(nt, np.nan)

    rock_array = predict_model\
        .predict_rock(f"{datadir}/{bf_globals.ROCK_MODEL}",
                      f"{datadir}/{bf_globals.ROCK_FEATURE_FILE}",
                      features) \
        if has_feature_data else np.full(nt, np.nan)

    composition_array = predict_model\
        .predict_composition(f"{datadir}/{bf_globals.COMPOSITION_MODEL}",
                             f"{datadir}/{bf_globals.COMPOSITION_FEATURE_FILE}",
                             features) if has_feature_data \
            else np.full((2, nt), np.nan)

    hydration_array = predict_model\
        .predict_hydration(f"{datadir}/{bf_globals.HYDRATION_MODEL}",
                           f"{datadir}/{bf_globals.HYDRATION_FEATURE_FILE}",
                           features) \
        if has_feature_data else np.full((3, nt), np.nan)

    if has_feature_data:
        pg_data = data['pg']
        imu_bin = data['imu_bin']
        contact_data = tools_pg.contact_area_run(pg_data, imu_bin)
        high_pressure_array = tools_pg.sharpPG(pg_data, contact_data,
                                               npix_lim=15, sharp_lim=4,
                                               pg_lim=80.)
        sinkage_array = tools_pg.sinkagePG(pg_data, imu_bin,
                                           contact_data, s=100)[0]
        lean_array = tools_pg.leanPG(pg_data, contact_data,
                                     q2_perc=0.05)
    else:
        high_pressure_array = np.full(nt, np.nan)
        lean_array = np.full(nt, np.nan)
        sinkage_array = np.full(nt, np.nan)

    #surface_array = predict_model.predict_surface_patterns(datadir, features)
    surface_array = predict_model.\
        predict_surface_patterns(len(lean_array)) if has_feature_data \
        else np.full(nt, np.nan)

    logger.info("Generating prediction plot...")
    t = np.arange(len(slip_array))

    y_labels = ["Slip", "High Pressure", "Lean", "Composition",
                "Hydration", "Sinkage", "Surface Pattern", "Rock"]
    prediction_frames = [slip_array, high_pressure_array, lean_array,
                         composition_array, hydration_array,
                         sinkage_array, surface_array, rock_array]

    if plotFile:
        _, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
        for i in range(7):
            label = y_labels[i]
            p_frame = prediction_frames[i]
            logger.debug(f'Plotting {label}')
            axes[i].plot(t, p_frame if len(p_frame.shape) == 1 \
                                    else p_frame[:,0] )
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(label)
            axes[i].set_title(f"{label} Vs. Time")
            axes[i].grid(True)

        logger.info(f"Saving prediction plot as: {plotFile}")
        plt.savefig(plotFile)

    dictionary = {}
    for i, label in enumerate(y_labels):
        p_frame = prediction_frames[i]
        dictionary[label] = p_frame
        logger.debug(f'y_label: {label}')
        logger.debug(f'type : {type(p_frame)}')
        if has_feature_data:
            logger.info(f"Dimensions of {label} prediction array: {p_frame.shape}")
    dictionary['nan_mask'] = nan_mask

    return dictionary


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--multithreading', help='multithreading trigger',
                        action='store_true')

    parser.add_argument('--datadir',
                        default="/Volumes/MLIA_active_data/data_barefoot/",
                        help="Path to data repository. Defaults to \
                            /Volumes/MLIA_active_data/data_barefoot/")

    parser.add_argument('--outdir',
                        default="/Users/jackal/Documents/BAREFOOT_ROVER/test/",
                        help="")

    parser.add_argument('--testName',
                        default="test/slip/CRT_slip0.171\
                                _terrain_flatlvl_vel_fast_\
                                EISfreqsweep_10-10K_grousers_\
                                full_loading_20_material_\
                                mins30_hydra_00.0_pretreat_\
                                N_date_20181119_rep_01/",
                        help="")

    parser.add_argument('--plotFile', default="test.png",
                        help="File name to save resulting prediction plot. \
        Ex: /data/MLIA_ACTIVE_DATA/data_barefoot/results/prediction.jpeg")

    parser.add_argument('--module', default="rock",
                        help="type of classifier/regressor")

    parser.add_argument('--featureFile', default="full_features.txt",
                         help="Line-seperated file of features")

    args = parser.parse_args()

    run_system(args.datadir, args.outdir, args.testName,
              ["200", "None"], args.plotFile, args.module,
               args.featureFile, args.multithreading)
