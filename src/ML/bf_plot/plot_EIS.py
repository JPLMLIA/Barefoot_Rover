'''
Author: Lukas Mandrake
Date  : 2/6/18

Brief : Production plots for EIS initial exploration

Notes :
'''
import argparse
import glob
import itertools
import logging
import os
import sys
from collections import OrderedDict

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np
import pylab as P
from matplotlib import cm as colormap
from sklearn.metrics import accuracy_score, confusion_matrix

from bf_config import bf_globals
from bf_logging import bf_log
from bf_tools import tools_eis

bf_log.setup_logger()
logger = logging.getLogger(__name__)

# Hz, frequency below which EIS response is extremely noisy
FREQ_LIMIT = 300

mat_marker_dict = {'redgar': '.', 'wed730': 'd', 'mmdust': 's', 'mm.2mm': '>', 'mmcrse': '+',
                   'bst110': 'o', 'grc-01': 'p', 'mins30': 'h', 'mmintr': '*'}

mat_col_dict = {'redgar': 'red', 'wed730': 'blue', 'mmdust': 'green', 'mm.2mm': 'cyan',
                'mmcrse': 'magenta', 'bst110': 'gold', 'grc-01': 'lime', 'mins30': 'dimgray',
                'mmintr': 'blueviolet'}

hyd_col_dict = {'0': 'k', '0.2': 'yellow', '0.4': 'pink', '0.6': 'purple', '0.8': 'green',  '0.5': 'cadetblue', '1': 'steelblue', '1.5': 'skyblue',
                '1.6': 'skyblue', '1.7': 'skyblue', '2': 'dodgerblue', '3': 'cornflowerblue', '3.1': 'cornflowerblue',
                '4': 'royalblue', '5': 'deepskyblue',
                '10': 'turquoise', '15': 'cyan'}

rot_col_dict = {'0': 'green', '90': 'blue', '180': 'red', '270': 'gold'}


dep_marker_dict = {'0': 's', '2': '.', '5': 'x', '10': '>', '20': '*', '30': 'p',
                   '40': '+', '50': 'o'}


dep_col_dict = {'0': '0.1', '2': '0.2', '5': '0.3', '10': '0.4', '20': '0.5', '30': '0.6',
                '40': '0.8', '50': '1'}


grain_size_dict = {'redgar': '1.55', 'wed730': '0.14', 'mmdust': '0.025', 'mm.2mm': '0.04', 'mmcrse': '1.0',
                   'bst110': '0.08', 'grc01': '0.39', 'mins30': '0.19', 'mmintr': '0.22'}


seq = np.linspace(0, 256, len(hyd_col_dict.keys())-1, dtype=int)
colors = plt.get_cmap('rainbow')(seq)
keys = sorted(hyd_col_dict.keys(), key=lambda x: float(x))[1:]
for j, k in enumerate(keys):
    hyd_col_dict[k] = list(colors[j])

 #colors = categorical_cmap(len(dep_col_dict.keys()), 1, cmap='Accent')
seq = np.linspace(0, 256, len(dep_col_dict.keys()), dtype=int)
colors = plt.get_cmap('Dark2')(seq)
for j, k in enumerate(dep_col_dict.keys()):
    dep_col_dict[k] = list(colors[j])


seq = np.linspace(0, 256, len(hyd_col_dict.keys())-1, dtype=int)
colors = plt.get_cmap('rainbow')(seq)
keys = sorted(hyd_col_dict.keys(), key=lambda x: float(x))[1:]
for j, k in enumerate(keys):
    hyd_col_dict[k] = list(colors[j])


# -------------
def set_colors(hyd_levels):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    hyd_levels : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    hyd_col_dict = {}
    hyd_col_dict['0.0'] = 'black'
    # NOTE: This used to be len(hyd_levels)-1 but tests failed
    seq = np.linspace(0, 256, len(hyd_levels), dtype=int)
    colors = plt.get_cmap('rainbow')(seq)
    # NOTE:  Used to ignore the first element but tests were failing
    keys = sorted(hyd_levels, key=lambda x: float(x))
    for j, k in enumerate(keys):
        hyd_col_dict[str(k)] = list(colors[j])
    return hyd_col_dict


# -------------
def plotBodeEIS_byRot(EISdata, rot_levels=[0, 90, 180, 270]):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    EISdata : [type]
        [description]
    rot_levels : list, optional
        [description], by default [0, 90, 180, 270]

    Returns
    -------
    [type]
        [description]
    """

    amp_dry = [10**5.1, 10**6.1, 10**7.1, 10**8.1]
    phase_dry = [-85, -85, -85, -85]
    # BODE----------
    fig = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for r in range(len(rot_levels)-1):
        rot1 = rot_levels[r]
        rot2 = rot_levels[r+1]
        mask = (EISdata['rot'] >= rot1) & (EISdata['rot'] < rot2)
        mean_rot = np.nanmean(EISdata['amp'][mask], axis=0)
        mean_phase = np.nanmean(EISdata['phase'][mask], axis=0)

        ax1 = plt.subplot(1, 2, 1)
        p1 = ax1.loglog(EISdata['freq'][mask], EISdata['amp'][mask], '.',
                        color=rot_col_dict[str(rot1)], alpha=0.5, label=str(rot1))
        ax1.plot(tools_eis.old_freqs, amp_dry,
                 ls=':', color='0.0', label='dry')
        if len(EISdata['freq'][mask]) > 0:
            p1 = ax1.loglog(EISdata['freq'][mask][0], mean_rot, 'x-', ms=7,
                            color=rot_col_dict[str(rot1)], alpha=0.5, lw=1)
        ax2 = plt.subplot(1, 2, 2)
        ax2.semilogx(EISdata['freq'][mask], EISdata['phase'][mask], '.',
                     color=rot_col_dict[str(rot1)], alpha=0.5, label=str(rot1))
        ax2.plot(tools_eis.old_freqs, phase_dry,
                 ls=':', color='0.0', label='dry')
        if len(EISdata['freq'][mask]) > 0:
            ax2.semilogx(EISdata['freq'][mask][0], mean_phase, 'X-', ms=7,
                         color=rot_col_dict[str(rot1)], alpha=0.5, lw=1)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('abs(Z)', color='b')
    ax1.set_ylim((100, 10**8.5))
    ax1.tick_params('y', colors='b')
    ax1.margins(0.05, 0.05)
    ax1.grid(True, alpha=0.25)
    ax1.set_title('amplitude')

    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::len(rot_levels)], labels[::len(rot_levels)],
                      loc='best', frameon=False, fontsize='small', ncol=2, borderaxespad=0.,
                      numpoints=1, markerscale=1.5)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (Deg)', color='k')
    ax2.tick_params('y', colors='k')
    ax2.set_ylim([-180, 180])
    ax2.margins(0.05, 0.05)
    ax2.grid(True, alpha=0.25)
    ax2.set_title('phase')
    plt.suptitle(
        f"Bode Plot, hydr {EISdata['hyd'][0]}, material {EISdata['mat'][0]}", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    return fig


# -------------
def plotFreqEIS_byTime(EISdata, freqs=[10000, 1000, 100, 10]):
    """ Short description

    Longer description

    Parameters
    ----------

    Returns
    -------

    """
    amp_dry = [10**5.1, 10**6.1, 10**7.1, 10**8.1]
    seq = np.linspace(70, 256, len(freqs), dtype=int)
    colors = plt.get_cmap('Blues')(seq)

    fig = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    for j in range(len(freqs)):
        ax1 = plt.subplot(1, 1, 1)
        p1 = ax1.semilogy(EISdata['amp'][:, j], '.-',
                          color=colors[j], alpha=0.5, label=f"{freqs[j]}Hz")
        ax1.axhline(amp_dry[j], ls=':', color=colors[j])

    ax1.set_ylabel('abs(Z)', color='b')
    ax1.set_xlabel('time step')
    ax1.set_ylim((10, 10**8.5))
    ax1.legend()
    ax1.set_title('amplitude over time')
    ax1.grid(True, alpha=0.25)
    plt.suptitle(
        f"Bode Plot, hydr {EISdata['hyd'][0]}, material {EISdata['mat'][0]}")

    return fig


# ----------------------------------------
def bodePlot(data, mat_levels, hyd_levels, group, freq_mask=None, rot_mask=None, add_mean=False,
             alim=[10, 10**8.5], flim=None):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    data : [type]
        [description]
    mat_levels : [type]
        [description]
    hyd_levels : [type]
        [description]
    group : [type]
        [description]
    freq_mask : [type], optional
        [description], by default None
    rot_mask : [type], optional
        [description], by default None
    add_mean : bool, optional
        [description], by default False
    alim : list, optional
        [description], by default [10, 10**8.5]
    flim : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    if freq_mask is None:
        freq_mask = np.repeat(True, data['amp'].shape[1])
    if rot_mask is None:
        rot_mask = np.repeat(True, len(data['rot']))
    amp = data['amp'][rot_mask][:, freq_mask]
    phase = data['phase'][rot_mask][:, freq_mask]
    hydration = data['hyd'][rot_mask]
    material = data['mat'][rot_mask]
    freq = data['freq'][rot_mask][:, freq_mask]

    hyd_levels = np.atleast_1d(hyd_levels)
    mat_levels = np.atleast_1d(mat_levels)
    hyd_col_dict = set_colors(hyd_levels)
    N = amp.shape[0]

    if flim is None:
        flim = [freq.min() - freq.min()*0.2, freq.max() + freq.max()*0.2]
    if alim is None:
        alim = [np.nanmin(amp) - np.nanmin(amp)*0.2,
                np.nanmax(amp) + np.nanmax(amp)*0.2]

    fig = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    ax = [None] * 2
    for i, hyd in enumerate(hyd_levels):
        mask_hyd = np.in1d(hydration, hyd)

        for j, m in enumerate(mat_levels):
            if group == 'hydration':
                key = str(hyd)
                idx = np.argmin(
                    np.abs(np.array(list(hyd_col_dict)).astype(float) - float(key)))
                key = list(hyd_col_dict)[int(idx)]
                color = hyd_col_dict[key]
            if group == 'composition':
                key = str(m)
                color = mat_col_dict[key]
            mask_mat = np.in1d(material, m)
            mask_plot = mask_mat & mask_hyd
            ax[0] = plt.subplot(1, 2, 1)
            print(f"TEST: m in mat_marker = {m in mat_marker_dict}")
            ax[0].loglog(freq[mask_plot], amp[mask_plot], mat_marker_dict[m], color=color,
                         alpha=0.3, markeredgecolor=color, zorder=10)
            ax[1] = plt.subplot(1, 2, 2)
            ax[1].semilogx(freq[mask_plot], phase[mask_plot], mat_marker_dict[m], color=color,
                           alpha=0.3, markeredgecolor=color,
                           label=key)
            if add_mean:
                if mask_plot.sum() != 0:
                    mean_amp = np.nanmedian(amp[mask_plot, :], axis=0)
                    mean_phase = np.nanmedian(phase[mask_plot, :], axis=0)
                    mean_freq = freq[mask_plot, :][0, :]
                else:
                    mean_amp = np.repeat(np.nan, freq.shape[1])
                    mean_phase = np.repeat(np.nan, freq.shape[1])
                    mean_freq = freq[0, :]

                ax[0].loglog(mean_freq, mean_amp, mat_marker_dict[m], color='k',
                             ms=8, zorder=10)
                ax[0].loglog(mean_freq, mean_amp, mat_marker_dict[m] + '-', color=color,
                             alpha=0.3, lw=1, markeredgecolor=color, zorder=10)
                ax[1].semilogx(mean_freq, mean_phase, mat_marker_dict[m], color='k',
                               ms=8, zorder=10)
                ax[1].semilogx(mean_freq, mean_phase, mat_marker_dict[m] + '-', color=color,
                               alpha=0.3, lw=1, markeredgecolor=color, zorder=10)

    fsize = 24
    ax[0].tick_params(labelsize=fsize)
    ax[0].set_xlabel('Frequency (Hz)', fontsize=fsize)
    ax[0].set_ylabel('amplitude', color='b', fontsize=fsize)
    ax[0].tick_params('y', colors='b')
    ax[0].margins(0.05, 0.05)
    ax[0].minorticks_on()
    ax[0].grid(True, alpha=0.25)
    ax[0].set_xlim(flim)
    ax[0].set_ylim(alim)

    ax[1].set_xlabel('Frequency (Hz)', fontsize=fsize)
    ax[1].set_ylabel('Phase (Deg)', color='k', fontsize=fsize)
    ax[1].tick_params('y', colors='k', labelsize=fsize)
    ax[1].tick_params('x', labelsize=fsize)

    handles, ticks = ax[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(ticks, handles))
    handles = by_label.values()
    ticks = by_label.keys()

    if group == 'hydration':
        group = 'hyd %'
        mat_mar = ['k'] * len(mat_levels)
        plt.rcParams['legend.title_fontsize'] = 'xx-large'
        lgd1 = ax[1].legend(handles, ticks,
                            loc='upper center', frameon=False, fontsize=fsize - 2, ncol=1, borderaxespad=0.,
                            numpoints=1, markerscale=3, bbox_to_anchor=(1.16, 0.85), title=group)

        handles2 = [plt.plot([], mm, color=cc) for cc, mm in zip(
            mat_mar, [mat_marker_dict[i] for i in mat_levels])]
        ax[1].add_artist(lgd1)

    ax[1].set_ylim([-150, 100])
    ax[1].margins(0.05, 0.05)
    ax[1].minorticks_on()
    ax[1].grid(True, alpha=0.25)
    ax[1].set_xlim(flim)
    plt.tight_layout()

    return fig, ax


# ----------------------------------------
def regr_plot(y, yhat, hyd_levels, materials, mat_levels, FREQ_LIMIT=300):
    """ Short description

    Longer description

    Parameters
    ----------

    Returns
    -------

    """
    rmse = np.sqrt(np.mean((y - yhat.mean(axis=1))**2))
    fig, ax = plt.subplots()
    for j, m in enumerate(mat_levels):
        mask_mat = np.in1d(materials, m)
        mask_zoom = np.in1d(y[mask_mat], hyd_levels)
        plt.plot(y[mask_mat][mask_zoom], yhat.mean(axis=1)[mask_mat][mask_zoom],
                 mat_marker_dict[m], color=mat_col_dict[m],
                 alpha=0.5, label=m)

    plt.margins(0.05, 0.05)
    plt.xlabel('response')
    plt.ylabel('predicted response')
    plt.xticks(hyd_levels, map(str, hyd_levels))
    plt.yticks(hyd_levels, map(str, hyd_levels))
    plt.legend(numpoints=1, loc='best')
    ax.grid(True, alpha=0.25)
    plt.plot(y[mask_mat][mask_zoom], y[mask_mat]
             [mask_zoom], '--', color='0.5', alpha=0.5)
    plt.title('regression, rmse = ' + str(np.round(rmse, 2)))

    return fig, ax


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    cm : [type]
        [description]\n
    classes : [type]
        [description]\n
    normalize : bool, optional
        [description], by default False\n
    title : str, optional
        [description], by default 'Confusion matrix'\n
    cmap : [type], optional
        [description], by default plt.cm.Blues\n

    Returns
    -------
    [type]
        [description]
    """

    acc = np.diagonal(cm).sum() / cm.sum(dtype=float)
    logger.info(str(acc))

    add_to_title = ', class counts'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100
        add_to_title = ', percent of each class'
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')

    logger.info(str(cm))

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + add_to_title + '\n accuracy = ' + '%.3f' % acc)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig

# nc -- number of categories, nsc -- number of subcategories


def categorical_cmap(nc, nsc, cmap="tab10"):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    nc : [type]
        [description]
    nsc : [type]
        [description]
    cmap : str, optional
        [description], by default "tab10"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = mp.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = mp.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc, :] = rgb
    cmap = mp.colors.ListedColormap(cols)
    return cols
