#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:19:23 2019

@author: marchett
"""
import glob
import numpy as np


#--------
module = 'rock'
version = 'v4'
date = '08222019'
model_type = 'classifier'
burnin = [50, None]
featuresFile = 'full_features.txt'
    
datadir = '/Volumes/MLIA_active_data/data_barefoot/'
#datadir = '/data/MLIA_active_data/data_barefoot/'

other_slip = glob.glob(datadir + 'train/slip_detection/stringpot_corrected/*')
np.random.seed(6729)
sub_idx_slip = np.random.choice(np.arange(len(other_slip)), 30, replace=False)
other_sub_slip = [x.split('/')[-1] for x in [other_slip[i] for i in sub_idx_slip]]

other_flat = glob.glob(datadir + 'train/composition/*')
np.random.seed(3203)
sub_idx_flat = np.random.choice(np.arange(len(other_flat)), 30, replace=False)
other_sub_flat = [x.split('/')[-1] for x in [other_flat[i] for i in sub_idx_flat]]

regex = ['*', '*'] + other_sub_flat + other_sub_slip
subfolders = ['train/rock_detection/rock_above', 'train/rock_detection/rock_below'] + \
              list(np.repeat('train/composition/', len(other_sub_flat))) + \
              list(np.repeat('train/slip_detection/stringpot_corrected/', len(other_sub_slip)))





'''
#--------
module = 'rock'
version = 'v3'
date = '05132019'
model_type = 'classifier'
burnin = [200, None]
    
datadir = '/Volumes/MLIA_active_data/data_barefoot/'
other = glob.glob(datadir + 'train/slip_detection/stringpot_corrected/*')
np.random.seed(6729)
sub_idx = np.random.choice(np.arange(len(other)), 30, replace=False)
other_sub = [x.split('/')[-1] for x in [other[i] for i in sub_idx]]
regex = ['*rock-above*', '*br*', '*flatlvl*'] + other_sub
subfolders = ['train/rock_detection', 'train/rock_detection', 'train/composition'] + \
              list(np.repeat('train/slip_detection/stringpot_corrected', len(other_sub)))
'''


'''
#--------
module = 'rock'
version = 'v2'
date = '04012019'
model_type = 'classifier'
burnin = [50, None]
    
datadir = '/Volumes/MLIA_active_data/data_barefoot/'

other = glob.glob(datadir + 'train/slip_detection/stringpot_corrected/*')
np.random.seed(6729)
sub_idx = np.random.choice(np.arange(len(other)), 30, replace=False)
other_sub = [x.split('/')[-1] for x in [other[i] for i in sub_idx]]
regex = ['*rock-above*', '*br*', '*flatlvl*'] + other_sub
subfolders = ['train/rock_detection', 'train/rock_detection', 'train/composition'] + \
              list(np.repeat('train/slip_detection/stringpot_corrected', len(other_sub)))


#--------
module = 'rock'
version = 'v1'
date = '02952019'
model_type = 'classifier'
burnin = [50, None]
    
datadir = '/Volumes/MLIA_active_data/data_barefoot/'
regex = ['*rock-above*', '*br*', '*flatlvl*']
subfolders = ['train/rock_detection', 'train/rock_detection', 'train/composition']

'''

