'''
Author: Jack Lightholder
Date  : 11/20/17

Brief : Global variables for the Barefoot Rover Project.
            Available from all bf_* libraries

Notes :
'''
import os
import sys
import argparse
import datetime

## Enviornment variable for setting Barefoot root directory.
BAREFOOT_ROOT  = os.getenv('BAREFOOT_ROOT')

BAREFOOT_DATA  = os.getenv("BAREFOOT_DATA")

#TODO Move most (or all) of this to a config (yaml) file
"""
Info: Angular offset between the 0 tare
       of the accelerometer and pressure pad row index 0
"""
ANGULAR_OFFSET = 62.57

"""
Info:  Number of pressure pads (circumference)
"""
PADS_L = 96

"""
Info:  Number of pressure pads (width)
"""
PADS_W = 20

"""
Info:

"""
SCALER = 1e-7

"""
Info: Time step used for tools_pg.align_pg_to_imu()

"""
T = 0.1

"""
Info:

"""
R = 5

"""
Info:

"""
THREADS=5

"""
Info: Relative path to current pickeled version of slip model

"""
SLIP_MODEL = "/models/slip/v9/trained_GB_slip_v9_08132019"

"""
Info: Relative path to current text file showing required features for slip model.

"""
SLIP_FEATURE_FILE = "/models/slip/v9/classifier_slip_v9_08132019.h5"

"""
Info: Relative path to current pickeled version of surface pattern model

"""
PATTERN_MODEL = "/models/patterns/v3/trained_GB_patterns_v3_10212019"

"""
Info: Relative path to current text file showing required features for surface pattern model.

"""
PATTERN_FEATURE_FILE = "/models/patterns/v3/classifier_patterns_v3_10212019.h5"

"""
Info: Relative path to curent pickeled version of hydration model.

"""
HYDRATION_MODEL = "/models/hydration/v1/trained_RF_hydration_v1_05172019"

"""
Info: Relative path to current text file showing required features for hydration model.

"""
HYDRATION_FEATURE_FILE = "/models/hydration/v1/features_hydration_v1_05172019.h5"

"""
Info: Relative path to current pickeled version of composition model.

"""
COMPOSITION_MODEL = "/models/material/v3/trained_GB_material_v3_11112019"

"""
Info: Relative path to current text file showing required features for composition model.

"""
COMPOSITION_FEATURE_FILE = "/models/material/v3/classifier_material_v3_11112019.h5"

"""
Info: Relative path to current pickeled version of pressure model.

"""
PRESSURE_MODEL = ""

"""
Info: Relative path to current text file showing required features for pressure model.

"""
PRESSURE_FEATURE_FILE = ""

"""
Info: Relative path to current pickeled version of rock model.

"""
ROCK_MODEL = "/models/rock/v4/trained_GB_rock_v4_08222019"

"""
Info: Relative path to current text file showing required features for rock model.

"""
ROCK_FEATURE_FILE = "/models/rock/v4/classifier_rock_v4_08222019.h5"

CALIBRATION_FILE_DATES = sorted(
    ['20190522', '20190205', '20191031'], reverse=True)
