#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:20:25 2019

@author: marchett
"""

#--------
module = 'patterns'
version = 'v3'
date = '10212019'
model_type = 'classifier'
burnin = [50, None]
featuresFile = 'noEIS_features.txt'
    
datadir = '/data/MLIA_active_data/data_barefoot/'
regex = ['*', '*', '*', '*', '*', '*', '*', '*']
subfolders = ['train/bedrock',
			  'train/pebbles',
			  'train/surface_patterns/dunes_smooth',
			  'train/surface_patterns/dunes_sharp',
			  'train/surface_patterns/gullies_2',
			  'train/rock_detection/rock_above',
			  'train/rock_detection/rock_below',
			  'train/composition']


'''
#--------
module = 'patterns'
version = 'v2'
date = '04162019'
model_type = 'classifier'
burnin = [50, None]
    
datadir = '/Volumes/MLIA_active_data/data_barefoot/'
regex = ['*sharp*']
subfolders = ['train/surface_patterns']
'''

'''
#--------
module = 'patterns'
version = 'v1'
date = '02952019'
model_type = 'classifier'
burnin = [50, None]
    
datadir = '/Volumes/MLIA_active_data/data_barefoot/'
regex = ['*', '*', '*flatlvl*']
subfolders = ['train/pebbles', 'train/surface_patterns', 'train/composition']
'''




