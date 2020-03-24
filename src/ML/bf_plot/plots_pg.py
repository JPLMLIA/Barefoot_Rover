# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:08:03 2018

@author: marchett
"""

import copy
import itertools
import logging
import os
from typing import Dict, List, Union

import imageio
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib as mp

import numpy as np
import pywt
#mp.use('Agg')
from matplotlib import pyplot as plt
from numpy.core.numeric import ndarray
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error)
from sklearn.neighbors.kde import KernelDensity

from bf_config import bf_globals
from bf_logging import bf_log
from bf_tools import tools_pg
from bf_util import numpy_util

plt.set_cmap('afmhot')


bf_log.setup_logger()
logger = logging.getLogger(__name__)

keys = ['all', 'grouser', 'nongrouser', 'ambiguous']
key_colors = {'all': 'green', 'grouser':'sandybrown', 'nongrouser':'dodgerblue',
              'ambiguous':'red'}

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


#--------------
def unwrappedGrid(lag: List[int], pg_imu: np.ndarray, x: np.ndarray,
                  clim=(0, 120), t = None, cb = True):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    lag : [type]
        [description]
    pg_imu : [type]
        [description]
    x : [type]
        [description]
    clim : tuple, optional
        [description], by default (0, 120)
    t : [type], optional
        [description], by default None
    cb : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if t is None:
        t = pg_imu[0].shape[1]
    fig, ax = plt.subplots(len(lag), 1, figsize = (10, 4))
    plt.subplots_adjust(hspace = 0.0)

    for l in range(len(lag)):
        plt.subplot(int(str(len(lag)) + str(1) + str(l+1)))
        im = plt.matshow(pg_imu[l], alpha=0.1, aspect='auto', interpolation = 'none',
                         fignum=False)
        plt.box(False)
        if (pg_imu[l] > 120).sum() < (pg_imu[l].shape[0] * pg_imu[l].shape[1])*0.75:
            plt.clim(clim)

        im = plt.matshow(pg_imu[l][:, 0:t+1], aspect='auto', interpolation = 'none',
                         fignum=False)
        plt.box(False)
        plt.yticks([])
        plt.xticks([])
        plt.xlim((0, len(x[l])))
        if cb:
            plt.ylabel(str(lag[l]))
            plt.yticks([])
        if l == len(lag) - 1:
            nrows = pg_imu[l].shape[1]
            if cb:
                plt.gca().xaxis.tick_bottom()
                locs = np.arange(nrows, step = 100)
                plt.xticks(locs, x[l][locs])
                plt.xlabel('time')
        else:
            if cb:
                plt.xticks([])

        if (pg_imu[l] > 120).sum() < (pg_imu[l].shape[0] * pg_imu[l].shape[1])*0.75:
            plt.clim(clim)

    if cb:
        cbar_ax = fig.add_axes([0.91, 0.1, 0.015, 0.8])
        plt.colorbar(im, cax=cbar_ax, label='norm pressure')

    return fig, ax



#--------------
def animatedFigure(data_binned, t, nt, contact_data, sharp, lean, imu_contact_bin, colmax, jpg_match,
                   pg_imu, x_imu, lag, plotdir, data_pred = None):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data_binned : [type]
        [description]
    t : [type]
        [description]
    nt : [type]
        [description]
    contact_data : [type]
        [description]
    sharp : [type]
        [description]
    lean : [type]
        [description]
    imu_contact_bin : [type]
        [description]
    colmax : [type]
        [description]
    jpg_match : [type]
        [description]
    pg_imu : [type]
        [description]
    x_imu : [type]
        [description]
    lag : [type]
        [description]
    plotdir : [type]
        [description]
    data_pred : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    if jpg_match is not None:
        jpg_time = float('.'.join(jpg_match.split('/')[-1].split('.')[0].split('_')))
    else:
        jpg_time = None
    pg_t = data_binned['pg'][t, :, :]

    anifig, ax = plt.subplots(8, 1, gridspec_kw = {'height_ratios':[1.8, 0.6, 1.5, 2.5, 1.6, 1.2, 1.2, 1.6], 'hspace':0.25},
                              figsize=(9, 10))
    title = ax[0].set_title('XXX')
    ax[0].title.set_position([.5, 1.25])
    im = ax[2].matshow(pg_t, aspect='auto')
    ax[2].axvline(x = imu_contact_bin[t], lw = 0.9, color = 'green')
    im.set_clim((0, 120))
    title_text = f"t = {t}, contact = {contact_data['all']['npix'][t]}, UTC_jpg = {jpg_time}"
    title.set_text(title_text)
    ax[2].title.set_position([.5, 1.25])
    ax[2].set_xticks([])

    ax[1].plot(colmax[t, :], '-', color='0.8', linewidth=1.5)
    ax[1].axvline(x = imu_contact_bin[t], lw = 0.8, color = 'green', label='IMU')
    ax[1].set_xlim(0, 96)
    ax[1].set_ylim(-20, 120)
    ax[1].set_ylabel('max pressure', fontsize = 8)
    ax[1].set_xlabel('pg row', fontsize = 8)
    ax[1].set_xticks([])
    ax[1].legend(loc = 1, fontsize = 6, frameon=False)
    for spine in ax[1].spines.values():
        spine.set_visible(False)

    #plot sink
    zstage = data_binned['sink'] * 0.01
    ax[5].plot(np.arange(0, nt), zstage[0:nt], '.', color = '0.6', ms=1, alpha=0.25)
    ax[5].plot(np.arange(0, t+1), zstage[0:t+1], '.', ms = 1, color='0.3')
    ax[5].plot(np.arange(0, t+1), zstage[0:t+1], '-', lw = 1, color='0.3', alpha=0.3)

    if data_pred is not None:
        ax[5].plot(np.arange(0, nt), data_pred['Sinkage'][0:nt], '.', color = 'b', ms=1, alpha=0.3)
        ax[5].plot(np.arange(0, t+1), data_pred['Sinkage'][0:t+1], '.', ms = 1, color='b')
        ax[5].plot(np.arange(0, t+1), data_pred['Sinkage'][0:t+1], '-', lw=1, color='b', alpha=0.4)

    ax[5].set_xlim(0, nt)
    ax[5].set_ylim(-50, 50)
    ax[5].set_ylabel('sink (mm)', fontsize = 8)
    ax[5].invert_xaxis()
    ax[5].minorticks_on()
    ax[5].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[5].grid(True, which='major', axis='both', alpha=0.3)
    ax[5].xaxis.set_ticklabels([])
    ax[5].xaxis.set_ticks_position('none')
    ax[5].annotate('zstage: ' + '%0.2f' % (zstage[t] * 0.01) + ' mm', xy = (0.8, 0.7),
                   xycoords = 'axes fraction', fontsize = 10, color = '0.3')
    if data_pred is not None:
        ax[5].annotate('%0.2f' % (data_pred['Sinkage'][t] * 0.01) + ' mm', xy = (0.65, 0.7),
                       xycoords = 'axes fraction', fontsize = 10, color='b')

    #plot slip
    slip = data_binned['slip']
    ax[6].plot(np.arange(0, nt), slip[0:nt], '.', color = '0.6', ms=1, alpha=0.25)
    ax[6].plot(np.arange(0, t+1), slip[0:t+1], '.', ms = 1, color='0.3')
    ax[6].plot(np.arange(0, t+1), slip[0:t+1], '-', lw = 1, color='0.3', alpha = 0.4)

    if data_pred is not None:
        ax[6].plot(np.arange(0, nt), data_pred['Slip'][0:nt], '.', color = 'b', ms=1, alpha=0.3)
        ax[6].plot(np.arange(0, t+1), data_pred['Slip'][0:t+1], '.', ms = 1, color='b')
        ax[6].plot(np.arange(0, t+1), data_pred['Slip'][0:t+1], '-', lw = 1, color='b', alpha=0.4)

    ax[6].set_xlim(0, nt)
    ax[6].set_ylim(-0.4, 1.2)
    ax[6].set_ylabel('slip (%)', fontsize = 8)
    ax[6].invert_xaxis()
    ax[6].minorticks_on()
    ax[6].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[6].grid(True, which='major', axis='both', alpha=0.3)
    ax[6].xaxis.set_ticklabels([])
    ax[6].xaxis.set_ticks_position('none')
    ax[6].annotate('strpot: % ' + '%0.2f' % (slip[t] * 100), xy = (0.8, 0.7),
                   xycoords = 'axes fraction', fontsize = 10, color = '0.3')
    if data_pred is not None:
        ax[6].annotate('% ' + '%0.2f' % (data_pred['Slip'][t] * 100), xy = (0.65, 0.7),
                       xycoords = 'axes fraction', fontsize = 10, color='b')


    rock_mask = data_binned['rock'] > 0

    rock_marker = np.repeat(-0.1, nt)
    rock_marker[~rock_mask] = np.nan
    ax[7].plot(np.arange(0, nt), rock_marker[0:nt], 'x', color = 'r', ms=2, alpha=0.2)
    ax[7].plot(np.arange(0, t+1), rock_marker[0:t+1], 'x', ms = 2, color='r', label = 'rock')

    if data_pred is not None:
        surf = data_pred['Surface Pattern']
        lkhd_rock = data_pred['Rock']
        #surface pattern prediction
        if np.isnan(surf).sum() < len(surf):
            class_dict = {'flatlvl': 0, 'gullies': 1, 'pebbles': 2, 'sharpdunes': 3, 'smoodunes': 4}
            vals = np.unique(surf[~np.isnan(surf[:,-1]), -1])
            surf_types = list(class_dict)
            label = np.empty((len(surf), ), dtype=object)
            for i in range(len(vals)):
                label[surf[:, -1] == vals[i]] = surf_types[i]
        else:
            label = 'NaN'

        ax[7].plot(np.arange(0, nt), lkhd_rock[0:nt], '.', color = 'b', ms=1, alpha=0.3)
        ax[7].plot(np.arange(0, t+1), lkhd_rock[0:t+1], '.', ms = 1, color='b')
        ax[7].plot(np.arange(0, t+1), lkhd_rock[0:t+1], '-', lw = 1, color='b', alpha=0.4)

    ax[7].set_xlim(0, nt)
    ax[7].set_ylim(-0.2, 1.5)
    ax[7].set_ylabel('rock lkhd (%)', fontsize = 8)
    ax[7].set_xlabel('<-- time', fontsize = 8)
    ax[7].invert_xaxis()
    ax[7].minorticks_on()
    ax[7].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[7].grid(True, which='major', axis='both', alpha=0.3)
    ax[7].legend(loc = 2, fontsize = 6, frameon=False)

    if data_pred is not None:
        #add rock
        ax[7].annotate('rock: % ' + '%0.1f' % (lkhd_rock[t] * 100), xy = (0.1, 0.7),
                       xycoords = 'axes fraction', fontsize = 10,
                       color='r' if lkhd_rock[t] > 0.5 else 'b')

        #add composition
        lkhd_comp = data_pred['Composition'][t, 0].astype(float)
        class_comp = data_pred['Composition'][t, 1]
        ax[7].annotate('comp: ' + str(class_comp), xy = (0.3, 0.7),
                       xycoords = 'axes fraction', fontsize = 10,
                       color='0.5')
        ax[7].annotate('lkhd: % ' + '%0.1f' % (lkhd_comp * 100), xy = (0.5, 0.7),
                       xycoords = 'axes fraction', fontsize = 10, color='r' if lkhd_comp > 0.5 else 'b')

        #add hydration
        lkhd_hyd = data_pred['Hydration'][t, 1]
        class_hyd = data_pred['Hydration'][t, 2]
        ax[7].annotate('% hyd: ' + '%0.0f' % (class_hyd), xy = (0.7, 0.7),
                       xycoords = 'axes fraction', fontsize = 10, color='0.5')
        ax[7].annotate('lkhd: % ' + '%0.1f' % (lkhd_hyd * 100), xy = (0.85, 0.7),
                       xycoords = 'axes fraction', fontsize = 10, color='r' if lkhd_hyd > 0.5 else 'b')

    #import imageio
    if jpg_match is not None:
        #in case image is corrupted
        try:
            im_jpg = imageio.imread(jpg_match)
        except:
            im_jpg = [[0,0,0,0]]
    else:
        im_jpg = [[0,0,0,0]]
    ax[0].imshow(im_jpg, aspect='auto')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    x = np.arange(nt)
    ax[4].invert_xaxis()
    for k in keys[:-1]:
       mask_amb = np.hstack(contact_data[k]['mask_ambiguous'] )
       npix = np.hstack(contact_data[k]['npix'])
       ax[4].plot(np.arange(0, nt), npix, color=key_colors[k], linewidth=0.8, alpha=0.2)
       ax[4].plot(np.arange(0, nt), npix, '.', color = key_colors[k],
                    ms=1, alpha = 0.01)
       ax[4].plot(np.arange(0, nt)[mask_amb], npix[mask_amb], '.',
                  color = '0.8', ms=2, alpha = 0.01)

    for k in keys[:-1]:
        mask_amb = np.hstack(contact_data[k]['mask_ambiguous'])
        npix = np.hstack(contact_data[k]['npix'])
        ax[4].plot(np.arange(0, t+1), npix[0:t+1], color=key_colors[k], linewidth=0.8,
                  alpha = 0.7)
        ax[4].plot(np.arange(0, t+1), npix[0:t+1], '.', color = key_colors[k],
                    ms=1, alpha = 0.7, label = k)
        ax[4].plot(np.arange(0, t+1)[mask_amb[0:t+1]], npix[0:t+1][mask_amb[0:t+1]], '.',
                  color = '0.95', ms=2, alpha=0.5)
        ax[4].plot(np.arange(0, t+1)[mask_amb[0:t+1]], npix[0:t+1][mask_amb[0:t+1]],
                  color = '0.95', alpha=0.5)

    npix = np.hstack(contact_data['all']['npix'])
    ax[4].plot(x[sharp == 1], npix[sharp == 1], 'x', color='r', ms=5, alpha = 0.5, label='sharp')
    ax[4].plot(x[lean == 1], npix[lean == 1], 'o', color='b', mfc='white',
                ms=5, alpha = 0.5, label='left')
    ax[4].plot(x[lean == -1], npix[lean == -1], 'o', color='m', mfc='white',
                ms=5, alpha = 0.5, label='right')
    ax[4].set_xlim(0, nt)
    ax[4].set_ylim(0, 140)
    ax[4].set_xlabel('<--- time', fontsize=8)
    ax[4].set_ylabel('contact (# pixels)', fontsize = 8)
    ax[4].invert_xaxis()
    ax[4].xaxis.tick_top()

    ax[4].minorticks_on()
    ax[4].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[4].grid(True, which='major', axis='both', alpha=0.3)
    ax[4].xaxis.set_ticklabels([])
    ax[4].xaxis.set_ticks_position('none')
    ax[4].legend(loc = 2, fontsize = 6, ncol=len(keys[:-1])+3, frameon=False)

    fig_unwrap, ax_unwrap = unwrappedGrid(lag, pg_imu, x_imu, t = t, cb = False)
    fig_unwrap.tight_layout(pad = -0.5)
    fig_unwrap.savefig(f"{plotdir}temp.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    im_unwrap = imageio.imread(f"{plotdir}temp.png")
    ax[3].imshow(im_unwrap, aspect='auto')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_xlabel('<--- time', fontsize=8)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].invert_xaxis()
    a=ax[3].get_ylim()
    ax[3].set_yticks(np.linspace(a[0], a[1], len(lag)))
    ax[3].set_yticklabels(lag[::-1], fontsize = 8)
    ax[3].set_ylabel('wheel row lag', fontsize = 8)
    os.remove(f"{plotdir}temp.png")

    return anifig

#--------------
def contactAreaPlot(contact_data: Dict[str, Dict[str, List[Union[int, np.ndarray]]]],
                    sharp: np.ndarray, lean: np.ndarray, ftype: str):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    contact_data : dict
        Results of tools_pg.contact_area_run()
    sharp :
        [description]
    lean : [type]
        [description]
    ftype : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    npix = {x: np.hstack(contact_data[x]['npix']) for x in contact_data.keys()}
    mask_amb = {x: np.hstack(contact_data[x]['mask_ambiguous']) for x in contact_data.keys()}
    contact = {x: contact_data[x]['contact_value'] for x in contact_data.keys()}
    offground = {x: contact_data[x]['noncontact_value'] for x in contact_data.keys()}

    nt = len(npix['all'])
    x = np.arange(nt)
    f, ax = plt.subplots(1, 2, figsize = (10, 4), gridspec_kw = {'width_ratios':[3, 1]})
    for k in keys[:-1]:
       ax[0].plot(x, npix[k], color = key_colors[k], linewidth=0.8, alpha=0.7, label = k)
       ax[0].plot(x, npix[k], '.', color = key_colors[k], ms=1, alpha = 0.7)
       ax[0].plot(x[mask_amb[k]], npix[k][mask_amb[k]], '.', color = '0.95', ms=1, alpha = 0.6)
       ax[0].plot(x[mask_amb[k]], npix[k][mask_amb[k]], color = '0.95', alpha = 0.6)
    ax[0].plot(x[sharp == 1], npix['all'][sharp == 1], 'x', color='r', ms=5, alpha = 0.5, label='sharp')
    ax[0].plot(x[lean == 1], npix['all'][lean == 1], 'o', color='b', mfc='white',
                ms=5, alpha = 0.5, label='left')
    ax[0].plot(x[lean == -1], npix['all'][lean == -1], 'o', color='m', mfc='white',
                ms=5, alpha = 0.5, label='right')

    ax[0].set_xlim(0, nt)
    ax[0].set_ylim(0, 140)
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('#pixels of contact')

    ax[0].minorticks_on()
    ax[0].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[0].grid(True, which='major', axis='both', alpha=0.3)

    ax[0].set_title('contact area vs time\n ' + ftype)
    ax[0].legend(loc = 2, frameon=False, ncol=3)

    noncontact = np.hstack(offground['all'])

    for k in keys[1:-1]:
        ax[1].hist(np.hstack(offground[k]), bins=100, density=True, color = key_colors[k],
                    ls = ':', alpha=0.8, histtype='step', label=f"{k}, off")
        ax[1].hist(np.hstack(contact[k]), bins=100, density=True, color = key_colors[k],
                    histtype='step', label=f"{k}, on")

    ax[1].set_title('pressure off/on ground\n m(off) = ' + '%.2f' % np.nanmean(noncontact), fontsize=10)
    ax[1].set_xlabel('pressure')
    ax[1].legend(frameon=False, fontsize=6)

    return f

#--------------


def contactAreaSmoothPlot(ftype: str,
                          data_binned: Dict[str, np.ndarray],
            contact_data: Dict[str, Dict[str, List[Union[int, np.ndarray]]]]):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    ftype : [type]
        [description]
    data_binned : [type]
        [description]
    contact_data : dict
        Results of tools_pg.contact_area_run()

    Returns
    -------
    [type]
        [description]
    """
    x_rock = np.where(data_binned['rock'] > 0)[0]
    npix_grouser = contact_data['grouser']['npix']
    npix_nongrouser = contact_data['nongrouser']['npix']
    run_mean_grouser = numpy_util.running_mean(npix_grouser, 2,2)
    run_mean_nongrouser = numpy_util.running_mean(npix_nongrouser, 2,2)

    coeffs = pywt.wavedec(npix_grouser, 'haar')
    coeffs_H = copy.deepcopy(coeffs)
    for z in range(4, len(coeffs_H)):
        coeffs_H[z] *= 0
    recon_grouser = pywt.waverec(coeffs_H, 'haar')
    coeffs = pywt.wavedec(npix_nongrouser, 'haar')
    coeffs_H = copy.deepcopy(coeffs)
    for z in range(4, len(coeffs_H)):
        coeffs_H[z] *= 0
    recon_nongrouser = pywt.waverec(coeffs_H, 'haar')

    fig = plt.figure(figsize = (12, 3))
    plt.plot(npix_grouser, '.', color = colors[1], label = 'gr')
    plt.plot(run_mean_grouser, color = colors[1])
    plt.plot(recon_grouser, color = colors[1], ls = '--')

    plt.plot(npix_nongrouser, '.', color = colors[0], label = 'non')
    plt.plot(run_mean_nongrouser, color = colors[0])
    plt.plot(recon_nongrouser, color = colors[0], ls = '--')
    plt.plot(x_rock, np.repeat(0, len(x_rock)), 'x', color='r', label='rock')
    plt.title('running mean and wavelet, ' + ftype)
    plt.ylabel('contact area (# pixels)')
    plt.xlabel('time')
    plt.legend(loc = 'best')

    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)

    return fig


def contactAreaLowFiltPlot(ftype: str, data_binned: Dict[str, np.ndarray],
                        contact_data: Dict[str, Dict[str, List[Union[int, np.ndarray]]]]):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    ftype : str
        [description]
    data_binned : dict
        [description]
    contact_data : dict
        Results of tools_pg.contact_area_run()

    Returns
    -------
    [type]
        [description]
    """
    x_rock = np.where(data_binned['rock'] > 0)[0]
    npix_grouser = contact_data['grouser']['npix']
    npix_nongrouser = contact_data['nongrouser']['npix']

    coeffs=pywt.wavedec(npix_grouser, 'haar')
    coeffs_flat = copy.deepcopy(coeffs)
    for z in range(0, len(coeffs_flat)-1):
        coeffs_flat[z] *= 0
    recon_grouser = pywt.waverec(coeffs_flat, 'haar')
    coeffs = pywt.wavedec(npix_nongrouser, 'haar')
    coeffs_flat = copy.deepcopy(coeffs)
    for z in range(0, len(coeffs_flat)-1):
        coeffs_flat[z] *= 0
    recon_nongrouser = pywt.waverec(coeffs_flat, 'haar')

    fig = plt.figure(figsize = (12, 3))
    plt.plot(recon_grouser, color = colors[1], ls = '--', label = 'gr')
    plt.plot(recon_nongrouser, color = colors[0], ls = '--', label = 'non')
    x_min = np.min([recon_grouser, recon_nongrouser])
    plt.plot(x_rock, np.repeat(x_min, len(x_rock)), 'x', color='r', label='rock')
    plt.title(ftype)
    plt.title('low pass filter, ' + ftype)
    plt.ylabel('contact area low pass reconstructed')
    plt.xlabel('time')
    plt.legend(loc = 'best')

    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)

    return fig


#--------------
def contactAreaHistPlot(ftype: str, data_binned: Dict[str, np.ndarray],
                        contact_data: Dict[str, Dict[str, List[Union[int, np.ndarray]]]]):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    ftype : [type]
        [description]
    data_binned : [type]
        [description]
    contact_data : dict
        Results of tools_pg.contact_area_run()

    Returns
    -------
    [type]
        [description]
    """

    npix_grouser = contact_data['grouser']['npix']
    npix_nongrouser = contact_data['nongrouser']['npix']

    X = np.hstack(npix_grouser)[:, np.newaxis]
    X_new_gr = np.linspace(X.min(), X.max(), 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    scores_grouser = kde.score_samples(X_new_gr)
    X = np.hstack(npix_nongrouser)[:, np.newaxis]
    X_new_non = np.linspace(X.min(), X.max(), 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    scores_nongrouser = kde.score_samples(X_new_non)

    fig = plt.figure()
    plt.hist(npix_nongrouser, bins = 100, histtype = 'step', density = True, label = 'non')
    plt.hist(npix_grouser, bins = 100, histtype = 'step', density = True, label = 'gr')
    plt.plot(X_new_gr, np.exp(scores_grouser), color=colors[1])
    plt.plot(X_new_non, np.exp(scores_nongrouser), color=colors[0])
    plt.legend()
    plt.title(ftype)
    plt.suptitle('histogram of contact area with kernel density')
    plt.xlabel('contact area (#pixels)')
    plt.ylabel('normalized counts')

    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)

    return fig

#--------------
def IMUunwrapRowMaxPlot(data_binned: Dict[str, np.ndarray],
                        pglag: List[int],
                        ftype: str,
                        time_binned = None):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data_binned : [type]
        [description]
    pglag : [type]
        [description]
    ftype : [type]
        [description]
    time_binned : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    if time_binned is None:
        time_binned = data_binned['time']
    pg_imu_gr, rho_imu, x_gr = tools_pg.unwrap_pg_to_imu(data_binned['pg'], data_binned['imu'],
                                                   time_binned,
                                                   pglag, extraction = 'grouser')
    pg_imu_non, rho_imu, x_non = tools_pg.unwrap_pg_to_imu(data_binned['pg'], data_binned['imu'],
                                                     time_binned,
                                                     pglag, extraction = 'nongrouser')
    fig, ax = plt.subplots(len(pg_imu_gr), 1, figsize=(9, 7.5))
    plt.subplots_adjust(hspace = 0.05)
    for l in range(len(pg_imu_gr)):
        x_rock = np.where(data_binned['rock'][x_gr[l]] > 0)[0]
        row_max_gr = pg_imu_gr[l].max(axis = 0)
        row_max_non = pg_imu_non[l].max(axis = 0)
        ax[l].plot(row_max_non, '-', color=colors[0], linewidth=1.5, label='non')
        ax[l].plot(row_max_gr, '-', color=colors[1], linewidth=1.5, label='gr')
        ax[l].plot(x_rock, np.repeat(-1, len(x_rock)), 'x', color='r', label='rock')
        ax[l].set_ylim((0, 120))
        ax[l].set_ylabel(str(pglag[l]))
        ax[l].minorticks_on()
        ax[l].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
        ax[l].grid(True, which='major', axis='both', alpha=0.3)
    ax[l].set_xlabel('time')
    plt.text(0.5,1.2, 'max pressure per row of IMU-unwrapped', horizontalalignment='center',
             transform=ax[0].transAxes)
    plt.suptitle(ftype)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=3)

    return fig

#--------------
def regressorPlots(data: Dict[str, np.ndarray],
                   y_pred: np.ndarray,
                   imps: np.ndarray,
                   version: str, n_top = 15):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data : [type]
        [description]
    y_pred : [type]
        [description]
    imps : [type]
        [description]
    version : [type]
        [description]
    n_top : int, optional
        [description], by default 15

    Returns
    -------
    [type]
        [description]
    """

    feature_names = data['feature_names'][:]
    names = ['predicted_scatter', 'hist_residuals', 'importance', 'residuals_by_run',
             'residual_by_material', 'residuals_by_slip']
    yhat = y_pred.mean(axis = 1)
    rmse = np.sqrt(mean_squared_error(data['y'], yhat))

    #scatter
    figs = [None] * len(names)
    figs[0], ax = plt.subplots()
    plt.plot(yhat, data['y'], '.', color='0.5', ms=0.8, alpha = 0.3)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, alpha = 0.7)
    plt.xlim((-0.1, 1))
    plt.ylim((-0.1, 1.1))
    plt.xlabel('gb predicted slip')
    plt.ylabel('true slip')
    plt.title('gb, rmspe = ' + '%.3f' % rmse + ', ' + version)

    bins = np.arange(0, 1.1, 0.05)
    idx = np.digitize(data['y'], bins)

    uidx = np.unique(idx)
    resid_mean = []
    resid_std = []
    ticks = []
    text = []
    for i, u in enumerate(uidx):
        mask = np.in1d(idx, u)
        rmse_u = np.sqrt(mean_squared_error(data['y'][mask], yhat[mask]))
        resid_mean.append([data['y'][mask].mean(), yhat[mask].mean()])
        resid_std.append([data['y'][mask].std(), yhat[mask].std()])
    resid_mean = np.array(resid_mean)
    resid_std = np.array(resid_std)
    plt.errorbar(resid_mean[:, 1], resid_mean[:, 0], yerr=resid_std[:, 1],
                 capsize=2, lw = 0.8, alpha = 0.6, linestyle = ':', color = 'b', elinewidth=1)
    plt.plot(resid_mean[:, 1], resid_mean[:, 0], '.', color = 'b', alpha = 0.6, ms = 5)
    plt.grid(alpha=0.5, linestyle=':')


    #residual histogram
    residuals = np.hstack(yhat - data['y'])
    rmse = np.sqrt(np.mean(residuals**2))
    figs[1], ax = plt.subplots()
    plt.hist(residuals, bins=50, histtype='step')
    plt.axvline(np.mean(residuals))
    plt.xlabel('gb residuals')
    plt.ylabel('density/counts')
    plt.title('gb, rmspe = ' + '%.3f' % rmse + ', ' + version)

    #importance
    imps_mean = imps.mean(axis = 0)
    imps_std = imps.std(axis = 0)
    imps_max = imps_mean.max()
    sort_idx = np.argsort(imps_mean / imps_max)[-n_top:]
    figs[2], ax = plt.subplots(figsize = (10, 3.5))
    plt.barh(np.arange(len(sort_idx)), (imps_mean / imps_max)[sort_idx],
             yerr = imps_std[sort_idx], color = 'r', alpha=0.5)
    plt.yticks(np.arange(0, len(sort_idx)), feature_names[sort_idx].astype(str),
               rotation='horizontal', fontsize = 16)
    plt.subplots_adjust(left=0.4)
    plt.xlim((0, 1.1))
    plt.xlabel('relative importance of features')
    plt.minorticks_on()
    plt.grid(alpha=0.5, linestyle=':')
    plt.tight_layout()


    #by trial run
    runnames = data['files'][:].astype(str)
    ufiles = np.unique(runnames)
    resid_list = []
    ticks = []
    for u in ufiles:
        mask = np.in1d(runnames, u)
        ticks.append(runnames[mask][0].split('_')[0].split('p')[-1])
        resid_list.append(yhat[mask] - data['y'][mask])

    figs[3], ax = plt.subplots(figsize = (12, 4))
    plt.boxplot(resid_list, positions=np.arange(0, len(resid_list)),
                showfliers=False, whiskerprops=dict(linestyle='--',linewidth=1.0, color='0.7'))
    plt.axhline(y=0, color='red', ls = '--', lw = 1)
    plt.xticks(np.arange(len(ticks))[::5], ticks[::5], rotation = 70)
    plt.title('residuals by run')
    plt.xlabel('value per run')
    plt.ylabel('residual (y_pred - y)')
    plt.minorticks_on()
    plt.grid(alpha=0.5, linestyle=':')
    plt.tight_layout()

    #residual by material
    boxnames = data['material'][:].astype(str)
    ufiles = np.unique(boxnames)
    resid_list = []
    ticks = []
    text = []
    for u in ufiles:
        mask = np.in1d(boxnames, u)
        ticks.append(boxnames[mask][0].split('_')[0].split('p')[-1])
        resid_list.append(yhat[mask] - data['y'][mask])
        rmse_u = np.sqrt(mean_squared_error(data['y'][mask], yhat[mask]))
        text.append('%.2f' % rmse_u)

    figs[4], ax = plt.subplots(figsize = (8, 8))
    bp = plt.boxplot(resid_list, positions=np.arange(0, len(resid_list)),
                     showfliers = True, widths = 0.2)
    for flier in bp['fliers']:
        flier.set_marker('x')
        flier.set_markeredgecolor('0.5')
        flier.set_markersize(1)
        flier.set_alpha(0.3)

    fsize = 24
    wp = np.array(bp['whiskers']).reshape((len(ticks), 2))
    for t in range(len(text)):
        x = wp[t][0].get_xydata()[0, 0]
        y = wp[t][1].get_xydata()[1,1]
        #bp['whiskers'][t+22].get_xydata()
        plt.text(x+0.0, y+0.15, text[t], horizontalalignment='center', fontsize = fsize)

    plt.axhline(y=0, color='red', ls = '--', lw = 1)
    plt.xticks(np.arange(len(ticks)), ticks, rotation = 40, fontsize = fsize)
    ax.tick_params(axis = 'y', labelsize = fsize)

    plt.ylabel('residual', fontsize = 26)
    plt.minorticks_on()
    plt.grid(alpha=0.5, linestyle=':')
    plt.tight_layout()


    #slip vs residual
    fsize = 14
    bins = np.arange(0, 1.1, 0.05)
    idx = np.digitize(data['y'], bins)

    uidx = np.unique(idx)
    resid_list = []
    ticks = []
    text = []
    for i, u in enumerate(uidx):
        mask = np.in1d(idx, u)
        rmse_u = np.sqrt(mean_squared_error(data['y'][mask], yhat[mask]))
        ticks.append('%.2f' % bins[i])
        resid_list.append(yhat[mask] - data['y'][mask])
        text.append('%.2f' % rmse_u)

    figs[5], ax = plt.subplots(figsize = (10, 4))
    bp = plt.boxplot(resid_list, positions=np.arange(0, len(resid_list)),
                     showfliers=False, widths = 0.2,
                     whiskerprops=dict(linestyle='--',linewidth=1.0, color='0.7'))
    plt.axhline(y=0, color='red', ls = '--', lw = 1)
    plt.xticks(np.arange(len(ticks)), ticks, rotation = 40, fontsize = fsize)
    ax.tick_params(axis = 'y', labelsize = fsize)

    wp = np.array(bp['whiskers']).reshape((len(ticks), 2))
    for t in range(len(text)):
        x = wp[t][0].get_xydata()[0, 0]
        y = wp[t][1].get_xydata()[1,1]
        plt.text(x, y+0.03, text[t], horizontalalignment='center', rotation = 40, fontsize = fsize)

    plt.xlabel('true slip ratio', fontsize = fsize)
    plt.ylabel('residual', fontsize = fsize)
    ymin, ymax = ax.get_ylim()
    plt.ylim((ymin, ymax + 0.1))
    plt.minorticks_on()
    plt.grid(alpha=0.5, linestyle=':')
    plt.title('gb, rmspe = ' + '%.3f' % rmse + ', ' + version)
    plt.tight_layout()

    return figs, names


#--------------
'''
TODO This may be better as three seperate functions (possibly a factory
     method of some sort).  One for each of the 'module' types.
'''
def classifierPlots(data, y_pred, prob_mean, imps, module, n_top = 15):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data : [type]
        [description]
    y_pred : [type]
        [description]
    prob_mean : [type]
        [description]
    imps : [type]
        [description]
    module : [type]
        [description]
    n_top : int, optional
        [description], by default 15

    Returns
    -------
    [type]
        [description]
    """
    figs = []

    names = ['confusion_matrix', 'hist_probs', 'importance']
    labels = data['y']
    labels_pred = y_pred.copy()
    uclass = np.unique(labels)
    if module == 'rock':

        classes = ['no rock', 'rock', 'buried']
        mask_br = data['rock_depth'] > 0.
        labels[mask_br] = 2

        mask_br_pred = labels_pred == 1.
        mask_ = mask_br & mask_br_pred
        labels_pred[mask_] = 2

    elif module == 'patterns':
        class_dict = tools_pg.terrain_dict
        classes = list(class_dict)
        class_idx = np.hstack(class_dict.values())
        class_mask = np.in1d(class_idx, uclass)
        classes = np.hstack(classes)[class_mask]
        names = ['confusion_matrix', 'hist_probs', 'pr_curves', 'importance']

    elif module == 'material':
        class_dict = tools_pg.composition_dict
        classes = list(class_dict)
        names = ['confusion_matrix', 'hist_probs', 'pr_curves', 'importance']

    # TODO Should throw error for slip and hydration?


    cm = confusion_matrix(labels, labels_pred)
    fig = plot_confusion_matrix(cm, classes = classes, normalize=True)
    figs.append(fig)

    #prob hist
    if module == 'rock':
        fsize = 26
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)
        for u in range(len(uclass)):
            mask = data['y'] == uclass[u]
            plt.hist(prob_mean[mask, 1], bins = 100, histtype = 'step', density = True,
                     label = classes[u], linewidth=3)

        plt.axvline(0.5, ls = '--', color = '0.7')
        plt.ylim((0,8))
        plt.legend(loc='best', bbox_to_anchor=(0.55,0.75), frameon= False, prop={'size': 22})
        plt.xlabel('likelihood of rock', fontsize = fsize)
        plt.ylabel('density/norm count', fontsize = fsize)
        plt.minorticks_on()
        plt.grid(alpha=0.5, linestyle=':')
        ax.yaxis.set_ticks_position('none')
        ax.set_yticklabels([], [])
        ax.tick_params(labelsize = fsize)
        plt.tight_layout()
        figs.append(fig)

    from sklearn.metrics import precision_recall_curve, auc
    if module == 'patterns':

        nc = len(uclass)
        nrow = int(np.ceil(nc / 5))
        #fig, ax = plt.subplots(1, 2, figsize = (6, 3))
        plot_data = []
        prob_level= []
        labels = []
        for u in range(len(uclass)):
            mask_hyd = np.in1d(data['y'], [uclass[u]])
            prob_pred = prob_mean[mask_hyd, u]
            plot_data.append(prob_pred)
            prob_level.append(prob_pred.mean())

            mask_label = np.hstack(tools_pg.terrain_dict.values()) == uclass[u]
            labels.append(np.hstack(list(tools_pg.terrain_dict))[mask_label][0])

        boxprops = dict(linestyle='-', color='0.2')
        whiskerprops = dict(linestyle='--', color='0.2')
        flierprops = dict(marker='o', markerfacecolor='0.5', markersize=2,
                  linestyle='none', markeredgecolor='0.5', alpha = 0.05)
        fig = plt.figure()
        bp = plt.boxplot(plot_data, labels = labels, showmeans=True,
                         flierprops=flierprops,
                         boxprops = boxprops, whiskerprops = whiskerprops)
        plt.title('likelihoods of classes')
        plt.ylabel('likelihhod')
        plt.grid(alpha = 0.3, ls = ':')
        plt.xticks(rotation=90)
        plt.tight_layout()
        figs.append(fig)
        plt.close()

        precision = dict()
        recall = dict()
        fig = plt.figure()
        for u in range(len(uclass)):
            mask = data['y'] == uclass[u]
            y_rest = data['y'].copy()
            y_rest[mask] = 1
            y_rest[~mask] = 0

            mask_label = np.hstack(tools_pg.terrain_dict.values()) == uclass[u]
            label = np.hstack(list(tools_pg.terrain_dict))[mask_label]

            precision[u], recall[u], _ = precision_recall_curve(y_rest, prob_mean[:, u])
            auc_class = auc(recall[u], precision[u])
            plt.plot(recall[u], precision[u], lw=2, label = label[0] + ', ' + '%0.2f' % auc_class)

        plt.title('PR curves, surface patterns')
        plt.grid(alpha = 0.3, ls = ':')
        plt.legend()
        #average_precision_score(Y_test[:, i], y_score[:, i])
        figs.append(fig)
        plt.close()

    if module == 'material':
        plot_data = []
        prob_level= []
        for h in range(len(uclass)):
            mask_hyd = np.in1d(labels, [uclass[h]])
            prob_pred = prob_mean[mask_hyd, h]
            plot_data.append(prob_pred)
            prob_level.append(prob_pred.mean())

        boxprops = dict(linestyle='-', color='0.2')
        whiskerprops = dict(linestyle='--', color='0.2')
        flierprops = dict(marker='o', markerfacecolor='0.5', markersize=2,
                  linestyle='none', markeredgecolor='0.5', alpha = 0.05)
        fig = plt.figure()
        bp = plt.boxplot(plot_data, labels = class_dict.keys(), showmeans=True,
                         flierprops=flierprops,
                         boxprops = boxprops, whiskerprops = whiskerprops)
        plt.title('likelihoods of classes')
        plt.ylabel('likelihhod')
        plt.grid(alpha = 0.3, ls = ':')
        figs.append(fig)


        precision = dict()
        recall = dict()
        fig = plt.figure()
        for u in range(len(uclass)):
            mask = data['y'] == uclass[u]
            y_rest = data['y'].copy()
            y_rest[mask] = 1
            y_rest[~mask] = 0

            mask_label = np.hstack(tools_pg.composition_dict.values()) == uclass[u]
            label = np.hstack(list(tools_pg.composition_dict))[mask_label]

            precision[u], recall[u], _ = precision_recall_curve(y_rest, prob_mean[:, u])
            auc_class = auc(recall[u], precision[u])
            plt.plot(recall[u], precision[u], lw=2, label = label[0] + ', ' + '%0.2f' % auc_class)

        plt.title('PR curves, surface patterns')
        plt.grid(alpha = 0.3, ls = ':')
        plt.legend()
        #average_precision_score(Y_test[:, i], y_score[:, i])
        figs.append(fig)
        plt.close()


    imps_mean = imps.mean(axis = 0)
    imps_std = imps.std(axis = 0)
    imps_max = imps_mean.max()
    sort_idx = np.argsort(imps_mean / imps_max)[-n_top:]
    fig, ax = plt.subplots(figsize = (10, 3.5))
    plt.barh(np.arange(len(sort_idx)), (imps_mean / imps_max)[sort_idx],
             yerr = imps_std[sort_idx], color = 'r', alpha=0.5)
    plt.yticks(np.arange(0, len(sort_idx)), data['feature_names'][sort_idx].astype(str),
               rotation='horizontal', fontsize = 16)
    plt.subplots_adjust(left=0.4)
    plt.xlim((0, 1.1))
    plt.xlabel('relative importance of features')
    plt.minorticks_on()
    plt.grid(alpha=0.5, linestyle=':')
    plt.tight_layout()
    figs.append(fig)

    return figs, names



#--------------
def plot_confusion_matrix(cm, classes: List[str],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    cm : [type]
        [description]
    classes : [type]
        [description]
    normalize : bool, optional
        [description], by default False
    title : str, optional
        [description], by default 'Confusion matrix'
    cmap : [type], optional
        [description], by default plt.cm.Blues

    Returns
    -------
    [type]
        [description]
    """

    acc = np.diagonal(cm).sum() / cm.sum(dtype=float)
    logger.info(str(acc))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')

    logger.info(str(cm))

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ', acc = ' + '%.3f' % acc)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig



def plotPredictions(data_binned: Dict[str, np.ndarray], nt: int, data_pred = None):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data_binned : [type]
        [description]
    nt : [type]
        [description]
    data_pred : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    rock_mask = data_binned['rock'] > 0
    rock_marker = np.repeat(-0.1, nt)
    rock_marker[~rock_mask] = np.nan

    fig, ax = plt.subplots(1,1, figsize = (10, 4))
    plt.plot(np.arange(0, nt), rock_marker[0:nt+1], 'x', ms = 3, color='r', label = 'rock')
    plt.axhline(0.5, ls = '--', color = '0.7', lw = 1)

    if data_pred is not None:
        surf = data_pred['Surface Pattern']
        lkhd_rock = data_pred['Rock']
        if np.isnan(surf).sum() < len(surf):
            class_dict = tools_pg.terrain_dict
            vals = np.unique(surf[~np.isnan(surf[:,-1]), -1])
            surf_types = list(class_dict)
            label = np.empty((len(surf), ), dtype=object)
            for i in range(len(vals)):
                label[surf[:, -1] == vals[i]] = surf_types[i]
        else:
            label = 'NaN'

        plt.plot(np.arange(0, nt), lkhd_rock[0:nt+1], '.', ms = 1, color='b')
        plt.plot(np.arange(0, nt), lkhd_rock[0:nt+1], '-', lw = 1, color='b', alpha=0.4)

    ax.set_xlim(0, nt)
    ax.set_ylim(-0.2, 1.1)
    ax.set_ylabel('rock lkhd (%)', fontsize = 12)
    ax.set_xlabel('time -->', fontsize = 12)
    ax.minorticks_on()
    ax.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax.grid(True, which='major', axis='both', alpha=0.3)
    ax.legend(loc = 2, fontsize = 10, frameon=False)
    plt.title('rock likelihoods')

    return fig


def unwrappedGrid_black(lag: List[int], pg_imu: np.ndarray, x: np.ndarray,
                        clim=(0, 120), t = None, cb = True):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    lag : [type]
        [description]
    pg_imu : [type]
        [description]
    x : [type]
        [description]
    clim : tuple, optional
        [description], by default (0, 120)
    t : [type], optional
        [description], by default None
    cb : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if t is None:
        t = pg_imu[0].shape[1]
    fig, ax = plt.subplots(len(lag), 1, figsize = (10, 4))
    plt.subplots_adjust(hspace = 0.0)

    for l in range(len(lag)):
        plt.subplot(int(str(len(lag)) + str(1) + str(l+1)))

        a = np.zeros(pg_imu[l].shape)
        im = plt.matshow(a, aspect='auto', interpolation = 'none',
                         fignum=False)
        plt.box(False)
        if (pg_imu[l] > 120).sum() < (pg_imu[l].shape[0] * pg_imu[l].shape[1])*0.75:
            plt.clim(clim)

        im = plt.matshow(pg_imu[l][:, 0:t+1], aspect='auto', interpolation = 'none',
                         fignum=False)
        plt.box(False)
        plt.yticks([])
        plt.xticks([])
        plt.xlim((0, len(x[l])))
        if cb:
            plt.ylabel(str(lag[l]))
            plt.yticks([])
        if l == len(lag) - 1:
            nrows = pg_imu[l].shape[1]
            if cb:
                plt.gca().xaxis.tick_bottom()
                locs = np.arange(nrows, step = 100)
                plt.xticks(locs, x[l][locs])
                plt.xlabel('time')
        else:
            if cb:
                plt.xticks([])

        if (pg_imu[l] > 120).sum() < (pg_imu[l].shape[0] * pg_imu[l].shape[1])*0.75:
            plt.clim(clim)

    if cb:
        cbar_ax = fig.add_axes([0.91, 0.1, 0.015, 0.8])
        plt.colorbar(im, cax=cbar_ax, label='norm pressure')

    return fig, ax


#--------------
def animatedFigure2(data_binned, t, nt, contact_data, sharp, lean, imu_contact_bin, colmax, jpg_match,
                   pg_imu, x_imu, lag, plotdir, data_pred = None):
    """ Short description

    Longer description

    Parameters
    ----------

    Returns
    -------

    """
    if jpg_match is not None:
        jpg_time = float('.'.join(jpg_match.split('/')[-1].split('.')[0].split('_')))
    else:
        jpg_time = None
    pg_t = data_binned['pg'][t, :, :]

    anifig, ax = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[0.3, 0.7], 'hspace':0.02},
                              figsize=(9, 5))
    title = ax[0].set_title('')
    ax[0].title.set_position([.5, 1.25])

    #import imageio
    if jpg_match is not None:
        #in case image is corrupted
        try:
            im_jpg = imageio.imread(jpg_match)
        except:
            im_jpg = [[0,0,0,0]]
    else:
        im_jpg = [[0,0,0,0]]

    ax[0].imshow(im_jpg, aspect='auto')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    fig_unwrap, ax_unwrap = unwrappedGrid_black(lag, pg_imu, x_imu, t = t, cb = False)
    fig_unwrap.tight_layout(pad = -0.5)
    fig_unwrap.savefig(f"{plotdir}temp.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    im_unwrap = imageio.imread(f"{plotdir}temp.png")
    ax[1].imshow(im_unwrap, aspect='auto')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel('<--- time', fontsize=8)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].invert_xaxis()
    a = ax[1].get_ylim()
    ax[1].set_yticks(np.linspace(a[0], a[1], len(lag)))
    ax[1].set_yticklabels(lag[::-1], fontsize = 8)
    ax[1].set_ylabel('wheel row lag', fontsize = 8)
    os.remove(f"{plotdir}temp.png")

    return anifig


#--------------
def animatedFigure3(data_binned, t, nt, contact_data, sharp, lean, imu_contact_bin, colmax, jpg_match,
                   pg_imu, x_imu, lag, plotdir, data_pred = None):
    """ Short description

    Longer description

    Parameters
    ----------

    Returns
    -------

    """
    if jpg_match is not None:
        jpg_time = float('.'.join(jpg_match.split('/')[-1].split('.')[0].split('_')))
    else:
        jpg_time = None
    pg_t = data_binned['pg'][t, :, :]

    anifig, ax = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[0.4, 0.6], 'hspace':0.1},
                              figsize=(9, 5))
    title = ax[0].set_title('')
    ax[0].title.set_position([.5, 1.25])

    #plot slip
    slip = data_binned['slip']
    ax[1].plot(np.arange(0, nt), slip[0:nt], '.', color = '0.6', ms=1, alpha=0.25)
    ax[1].plot(np.arange(0, t+1), slip[0:t+1], '.', ms = 1, color='0.3')
    ax[1].plot(np.arange(0, t+1), slip[0:t+1], '-', lw = 1, color='0.3', alpha = 0.4)

    if data_pred is not None:
        ax[1].plot(np.arange(0, nt), data_pred['Slip'][0:nt], '.', color = 'b', ms=1, alpha=0.3)
        ax[1].plot(np.arange(0, t+1), data_pred['Slip'][0:t+1], '.', ms = 1, color='b')
        ax[1].plot(np.arange(0, t+1), data_pred['Slip'][0:t+1], '-', lw = 1, color='b', alpha=0.4)

    ax[1].set_xlim(0, nt)
    ax[1].set_ylabel('slip (%)', fontsize = 8)
    ax[1].invert_xaxis()
    ax[1].minorticks_on()
    ax[1].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[1].grid(True, which='major', axis='both', alpha=0.3)
    ax[1].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticks_position('none')
    ax[1].annotate('strpot: % ' + '%0.2f' % (slip[t] * 100), xy = (0.8, 0.7),
                   xycoords = 'axes fraction', fontsize = 10, color = '0.3')
    if data_pred is not None:
        ax[1].annotate('% ' + '%0.2f' % (data_pred['Slip'][t] * 100), xy = (0.65, 0.7),
                       xycoords = 'axes fraction', fontsize = 10, color='b')

    if jpg_match is not None:
        #in case image is corrupted
        try:
            im_jpg = imageio.imread(jpg_match)
        except:
            im_jpg = [[0,0,0,0]]
    else:
        im_jpg = [[0,0,0,0]]

    ax[0].imshow(im_jpg, aspect='auto')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    return anifig


#--------------
def animatedFigure4(data_binned, t, nt, contact_data, sharp, lean, imu_contact_bin, colmax, jpg_match,
                   pg_imu, x_imu, lag, plotdir, data_pred = None):
    """ Short description

    Longer description

    Parameters
    ----------

    Returns
    -------

    """
    if jpg_match is not None:
        jpg_time = float('.'.join(jpg_match.split('/')[-1].split('.')[0].split('_')))
    else:
        jpg_time = None
    pg_t = data_binned['pg'][t, :, :]

    anifig, ax = plt.subplots(3, 1, gridspec_kw = {'height_ratios':[0.38, 0.25, 0.38], 'hspace':0.1},
                              figsize=(9, 6))
    title = ax[0].set_title('')
    ax[0].title.set_position([.5, 1.25])
    im = ax[1].matshow(pg_t, aspect='auto')
    ax[1].axvline(x = imu_contact_bin[t], lw = 0.9, color = 'green')
    im.set_clim((0, 120))
    title_text = f"t = {t}, contact = {contact_data['all']['npix'][t]}, UTC_jpg = {jpg_time}"
    title.set_text(title_text)
    ax[1].title.set_position([.5, 1.25])
    ax[1].set_xticks([])

    rock_mask = data_binned['rock'] > 0
    rock_marker = np.repeat(-0.1, nt)
    rock_marker[~rock_mask] = np.nan
    ax[2].plot(np.arange(0, nt), rock_marker[0:nt], 'x', color = 'r', ms=2, alpha=0.2)
    ax[2].plot(np.arange(0, t+1), rock_marker[0:t+1], 'x', ms = 2, color='r', label = 'rock')

    if data_pred is not None:
        surf = data_pred['Surface Pattern']
        lkhd_rock = data_pred['Rock']
        #surface pattern prediction
        if np.isnan(surf).sum() < len(surf):
            class_dict = {'flatlvl': 0, 'gullies': 1, 'pebbles': 2, 'sharpdunes': 3, 'smoodunes': 4}
            vals = np.unique(surf[~np.isnan(surf[:,-1]), -1])
            surf_types = list(class_dict)
            label = np.empty((len(surf), ), dtype=object)
            for i in range(len(vals)):
                label[surf[:, -1] == vals[i]] = surf_types[i]
        else:
            label = 'NaN'

        ax[2].plot(np.arange(0, nt), lkhd_rock[0:nt], '.', color = 'b', ms=1, alpha=0.3)
        ax[2].plot(np.arange(0, t+1), lkhd_rock[0:t+1], '.', ms = 1, color='b')
        ax[2].plot(np.arange(0, t+1), lkhd_rock[0:t+1], '-', lw = 1, color='b', alpha=0.4)

    ax[2].set_xlim(0, nt)
    ax[2].set_ylim(-0.2, 1.5)
    ax[2].set_ylabel('rock lkhd (%)', fontsize = 8)
    ax[2].set_xlabel('<-- time', fontsize = 8)
    ax[2].invert_xaxis()
    ax[2].minorticks_on()
    ax[2].grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    ax[2].grid(True, which='major', axis='both', alpha=0.3)
    ax[2].legend(loc = 2, fontsize = 6, frameon=False)

    if data_pred is not None:
        #add rock
        text_h = 0.8
        ax[2].annotate('rock: % ' + '%0.1f' % (lkhd_rock[t] * 100), xy = (0.1, text_h),
                       xycoords = 'axes fraction', fontsize = 10,
                       color='r' if lkhd_rock[t] > 0.5 else 'b')
        #add composition
        lkhd_comp = data_pred['Composition'][t, 0].astype(float)
        class_comp = data_pred['Composition'][t, 1]
        ax[2].annotate('comp: ' + str(class_comp), xy = (0.3, text_h),
                       xycoords = 'axes fraction', fontsize = 10,
                       color='0.5')
        ax[2].annotate('lkhd: % ' + '%0.1f' % (lkhd_comp * 100), xy = (0.5, text_h),
                       xycoords = 'axes fraction', fontsize = 10, color='r' if lkhd_comp > 0.5 else 'b')

        #add hydration
        lkhd_hyd = data_pred['Hydration'][t, 1]
        class_hyd = data_pred['Hydration'][t, 2]

        ax[2].annotate('% hyd: ' + '%0.0f' % (class_hyd), xy = (0.7, text_h),
                       xycoords = 'axes fraction', fontsize = 10, color='0.5')
        ax[2].annotate('lkhd: % ' + '%0.1f' % (lkhd_hyd * 100), xy = (0.85, text_h),
                       xycoords = 'axes fraction', fontsize = 10, color='r' if lkhd_hyd > 0.5 else 'b')

    #import imageio
    if jpg_match is not None:
        #in case image is corrupted
        try:
            im_jpg = imageio.imread(jpg_match)
        except:
            im_jpg = [[0,0,0,0]]
    else:
        im_jpg = [[0,0,0,0]]

    ax[0].imshow(im_jpg, aspect='auto')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    return anifig
