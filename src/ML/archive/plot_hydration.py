# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:13:09 2018

@author: marchett
"""

import sys, glob, os
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tools_eis as tools
import plot_EIS as plot
from src.ML.bf_config import bf_globals
from src.ML.bf_util import numpy_util
from src.ML.bf_util import trig_util
import argparse
import os

def hydration_plotting(datadir, plotdir):
    '''
    Inputs:

    Outputs:

    Notes:

    Examples:
    >>> datadir = bf_globals.BAREFOOT_DATA + "/cal/EIS/hydration/"
    >>> plotdir = bf_globals.BAREFOOT_ROOT + "/tmp/"

    #>>> hydration_plotting(datadir, plotdir)
    '''

    bf_globals.bf_log("hydration_plotting datadir: " + datadir)
    bf_globals.bf_log("hydration_plotting plotdir: " + plotdir)

    ###HYDRATION----------
    EISdata = {}
    EISdata['hyd'] = {'RE':[],'IM':[],'freq':[], 'hydra':[], 'mat':[], 'rot': []}

    for idffile in glob.glob(datadir + '*.idf'):
        data = tools.parse_idf(idffile)
        EISdata['hyd']['RE'  ].append(data[     'real_primary'])
        EISdata['hyd']['IM'  ].append(data['imaginary_primary'])
        EISdata['hyd']['freq'].append(data['frequency_primary'])
        EISdata['hyd']['hydra'].append(float(idffile.split('_')[14]))
        EISdata['hyd']['mat'].append(str(idffile.split('_')[12]))
        EISdata['hyd']['rot'].append(float(idffile.split('_')[6]))

    EISdata['hyd'] = numpy_util.numpyify(trig_util.add_polar(EISdata['hyd']))

    allhyd = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 20]
    lowhyd = [0, 0.5, 1, 1.5]
    midhyd = [2, 3, 4, 5]
    highhyd = [4, 5, 10, 20]


    ##set plotting params--------
    rot_levels = [0]
    hyd_levels  = lowhyd
    mat_levels  = ['wed730']
    alias = 'lowhyd_wed730'

    FREQ_LIMIT = 10000

    mask_sane      = EISdata['hyd']['freq'][0, :] >= FREQ_LIMIT
    freq_n = mask_sane.sum()
    x = EISdata['hyd']['freq'][:, mask_sane]
    y1 = EISdata['hyd']['amp'][:, mask_sane]
    y2 = EISdata['hyd']['phase'][:, mask_sane]

    ###BODE----------
    fig = plt.subplots(nrows=1, ncols=2, figsize = (12, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['hyd']['rot'], rot)
        for i, hyd in enumerate(hyd_levels):
            mask_hyd = np.in1d(EISdata['hyd']['hydra'], hyd)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['hyd']['mat'], m)
                mask = mask_mat & mask_hyd
                ax1 = plt.subplot(1, 2, 1)
                p1 = ax1.loglog(x[mask], y1[mask], plot.mat_marker_dict[m], color = plot.hyd_col_dict[str(hyd)],
                           alpha=0.3, label = str(hyd), markeredgecolor= plot.hyd_col_dict[str(hyd)])
                ax2 = plt.subplot(1, 2, 2)
                ax2.semilogx(x[mask], y2[mask], plot.mat_marker_dict[m], color = plot.hyd_col_dict[str(hyd)],
                           alpha=0.3, label = str(hyd), markeredgecolor = plot.hyd_col_dict[str(hyd)])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('abs(Z)', color = 'b')
    ax1.tick_params('y',colors='b')

    ax1.margins(0.05, 0.05)
    ax1.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n][::len(mat_levels)], labels[::freq_n][::len(mat_levels)],
                    loc = 'best', frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0.,
                    numpoints = 1, markerscale = 1.5)
    handles2 = [plt.plot([], mm, color=cc)
                for cc, mm in zip(['k', 'k'], [plot.mat_marker_dict[i] for i in mat_levels])]
    lgd2 = ax1.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                      numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))
    ax1.add_artist(lgd1)


    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (Deg)',color='k')
    ax2.tick_params('y',colors='k')

    ax2.set_ylim([-180,180])
    ax2.margins(0.05, 0.05)
    ax2.grid(True, alpha=0.25)
    lgd1 = ax2.legend(handles[::freq_n][::len(mat_levels)], labels[::freq_n][::len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)
    handles2 = [plt.plot([], mm, color=cc)
                for cc, mm in zip(['k', 'k'], [plot.mat_marker_dict[i] for i in mat_levels])]
    lgd2 = ax2.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                      numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))
    ax2.add_artist(lgd1)

    plt.suptitle('Bode Plot, Hydration')
    if not os.path.exists(plotdir + 'hydration/'):
        os.makedirs(plotdir + 'hydration/')
    plt.savefig(plotdir + 'hydration/bode_' + alias + '_f' + str(FREQ_LIMIT) + '.png')


    ##NYQUIST----------
    fig = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
    for i, hyd in enumerate(hyd_levels):
        mask_hyd = np.in1d(EISdata['hyd']['hydra'], hyd)
        for j, m in enumerate(mat_levels):
            mask_mat = np.in1d(EISdata['hyd']['mat'], m)
            mask = mask_mat & mask_hyd
            ax1 = plt.subplot(1, 1, 1)
            ax1.plot(y1[mask], -y2[mask], plot.mat_marker_dict[m],
                          color = plot.hyd_col_dict[str(hyd)], alpha=0.5, label = str(hyd))
            ax1.plot(y1[mask][0,:]* bf_globals.SCALER, -y2[mask][0,:] * bf_globals.SCALER, '-',
                          color = plot.hyd_col_dict[str(hyd)], alpha=0.2)

    plt.margins(0.05, 0.05)
    plt.xlabel('Real(Z)')
    plt.ylabel('-Imag(Z)')
    plt.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n*len(mat_levels)], labels[::freq_n*len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)
    handles2 = [plt.plot([], mm, color=cc)
                for cc, mm in zip(['k', 'k'], [plot.mat_marker_dict[i] for i in mat_levels])]
    lgd2 = ax1.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                      numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))
    ax1.add_artist(lgd1)

    plt.suptitle('Nyquist Plot, Hydration, freq limit ' + str(FREQ_LIMIT))
    plt.savefig(plotdir + 'hydration/nyquist_' + alias + '_f' + str(FREQ_LIMIT) + '.png')
    plt.close()


    ##AMP and PHASE as a func of HYDRATION-----------
    FREQ_LIMIT = 0
    mask_sane      = EISdata['hyd']['freq'][0, :] >= FREQ_LIMIT
    freq_n = mask_sane.sum()
    x = EISdata['hyd']['freq'][0][mask_sane]
    y1 = EISdata['hyd']['amp']
    y2 = EISdata['hyd']['phase']

    hyd_levels = allhyd
    mat_levels = ['redgar', 'wed730']
    for fq in range(len(x)):
        fig = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
        for i, hyd in enumerate(hyd_levels):
            mask_hyd = np.in1d(EISdata['hyd']['hydra'], hyd)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['hyd']['mat'], m)
                mask = mask_mat & mask_hyd
                ax1 = plt.subplot(1, 2, 1)
                ax1.plot(np.log(EISdata['hyd']['hydra'][mask] + 1), np.log(y1[mask][:, fq]),
                         plot.mat_marker_dict[m], color = plot.hyd_col_dict[str(hyd)],
                         alpha=0.5, label = str(hyd), markeredgecolor = plot.hyd_col_dict[str(hyd)])
                ax2 = plt.subplot(1, 2, 2)
                ax2.plot(np.log(EISdata['hyd']['hydra'][mask] + 1), y2[mask][:, fq],
                         plot.mat_marker_dict[m], color = plot.hyd_col_dict[str(hyd)],
                         alpha=0.5, label = str(hyd), markeredgecolor = plot.hyd_col_dict[str(hyd)])

        ax1.set_xlabel('hydration')
        ax1.set_ylabel('abs(Z)', color = 'b')
        ax1.tick_params('y',colors='b')

        ax1.margins(0.05, 0.05)
        ax1.grid(True, alpha=0.25)
        ax1.set_xticks([np.log(a + 1) for a in hyd_levels])
        ax1.set_xticklabels([str(a) for a in hyd_levels])
        handles, labels = ax1.get_legend_handles_labels()
        handles2 = [plt.plot([], mm, color=cc)
                    for cc, mm in zip(['k', 'k'], [plot.mat_marker_dict[i] for i in mat_levels])]
        lgd2 = ax1.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                          numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))

        ax2.set_xlabel('hydration')
        ax2.set_ylabel('Phase (Deg)',color='k')
        ax2.tick_params('y',colors='k')
        ax2.set_xticks([np.log(a + 1) for a in hyd_levels])
        ax2.set_xticklabels([str(a) for a in hyd_levels])
        ax2.set_ylim([-180,180])
        ax2.margins(0.05, 0.05)
        ax2.grid(True, alpha=0.25)
        handles2 = [plt.plot([], mm, color=cc)
                    for cc, mm in zip(['k', 'k'], [plot.mat_marker_dict[i] for i in mat_levels])]
        lgd2 = ax2.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                          numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))

        plt.suptitle('Amp and Phase vs. Hydration, freq ' + str(x[fq]))
        plt.savefig(plotdir + 'hydration/hyd_levels' + '_' + str(fq) + '.png')
        plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', default="/Volumes/MLIA_active_data/data_barefoot/cal/EIS/hydration/", help="Full path to barefoot data calibration directory.")
    parser.add_argument('--plotdir', help="Folder to store resulting plots")

    args = parser.parse_args()

    hydration_plotting(args.datadir, args.plotdir)



