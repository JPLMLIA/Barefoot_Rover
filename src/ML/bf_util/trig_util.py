""" Utility for polar/rectangular coordinates
"""
import cmath
import logging
import math

import numpy as np

from bf_logging import bf_log

bf_log.setup_logger()
logger = logging.getLogger(__name__)

### ---------------------------------------
def add_polar(indict) -> dict:
    """ Returns the polar coordinates in a dictionary (amplitude and phase)

    Parameters
    ----------
    indict : dict
        Assumes it has the keys 'RE' (real part) and 'IM' (imaginary part).  The values for
        both are expected to be a list of floats.

    Returns
    -------
    outdict : dict
        indict with the following added: {amp: list(float), phase: list(float)}.
        The keys represent amplitude and phase respectively.
    """
    indict['amp'  ] = []
    indict['phase'] = []
    for x in range(len(indict['RE'])):
        amp = []
        phase = []
        for r,i in zip(indict['RE'][x], indict['IM'][x]):
                m,p = cmath.polar(complex(r, i))
                amp.append(m)
                phase.append(p)
        indict['amp'].append(amp)
        indict['phase'].append(phase)
    return indict
