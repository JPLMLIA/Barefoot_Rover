'''
Author: Jack Lightholder
Date  : 12/12/17
Brief : IDF file management
Notes :
'''

import argparse
import io
import logging
import math
import os
import re
import shutil
import sys
from datetime import datetime
from os import listdir
from os.path import isdir, isfile, join
#from typing import List

import numpy as np

'''
from bf_config import bf_globals
from bf_logging import bf_log
from bf_util import numpy_util, trig_util

bf_log.setup_logger()
logger = logging.getLogger(__name__)
'''

freqs = [10000, 1000, 100, 10]
freqs_new = [10550.1, 947.86, 111.304, 10.0]


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
        #logger.info("not found")
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
    #print (data)
    #logging.info(f"Parsing {filePath}")
    for x in range(0,len(data)):

        line = data[x].rstrip().rstrip("\r\n")
        match = None

        regex = r"^Peaks=(true|false)$"
        match = re.match(regex, line)
        if match is not None:
            #logging.debug(f"Peaks={match.group(1)}")
            dictionary["Peaks"] = match.group(1)

        regex = r"^timing=(\d+)$"
        match = re.match(regex, line)
        if match is not None:
            #logging.debug(f"timesteps={match.group(1)}")
            dictionary["timesteps"] = match.group(1)

        regex = r"^starttime=(\d{1,2}/\d{1,2}/\d{4}) (\d{1,2}):(\d\d):(\d\d) (AM|PM)$"
        match = re.match(regex, line, flags=re.ASCII)
        if match is not None:
            #logging.debug(f"starttime={line}")
            day = match.group(1)
            hour = int(match.group(2))
            minutes = match.group(3)
            seconds = match.group(4)
            if match.group(5) == "PM" and hour != 12:
                hour = hour + 12
            if match.group(5) == "AM" and hour == 12:
                hour = 0
            #combined_time = f"{day} {hour}:{minutes}:{seconds}"
            combined_time = str(day) + " " + str(hour) +":" + str(minutes) + ":" + str(seconds)
            dt = datetime.strptime(combined_time,'%m/%d/%Y %H:%M:%S')
            utc = (dt-datetime.utcfromtimestamp(0)).total_seconds()
            dictionary["exp_time"] = utc

        regex = r"^timing\.time\[\d+\]=(\d+\.\d+)$"
        match = re.match(regex, line, flags=re.ASCII)
        if(match is not None):
            #logging.debug(f"timing.time={match.group(1)}")
            times.append(float(match.group(1)))

        regex = r"^(primary_data|pretreatmentdata)$"
        match = re.match(regex,line)
        if match is not None:
            #logging.debug(f"Found {match.group(1)}")
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
                for k in keys:
                    if k not in dictionary:
                        dictionary[k] = []
                for i in range(len(keys)):
                    dictionary[keys[i]].append(float(split[i]))

    # Only 'timing' is allowed to be empty
    return dictionary

if __name__ == "__main__":
    file = 'test_idf.idf'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    print (parse_idf(dir_path + "/" + file))
