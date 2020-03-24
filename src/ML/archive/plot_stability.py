# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:57:17 2018

@author: marchett
"""

import sys, glob
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tools_eis
import tools_pg as tools
from src.ML.bf_config import bf_globals
import bf_doctest
from src.ML.bf_util import numpy_util
from src.ML.bf_util import trig_util
import argparse
import os

def stability_plotting(datadir, plotdir):
    '''
    Inputs:

    Outputs:

    Notes:

    Examples:

    '''

    ###STABILITY ----------
    filepath = datadir + '/stability/'
    filedirs = glob.glob(filepath + '*')
    filenames = []
    for d in filedirs:
        filenames.extend(glob.glob(d + '/' + '*.idf'))
    EISdata = {}

    EISdata['rot'] = {'RE':[],'IM':[],'freq':[], 'hydra':[], 'mat':[], 'rot': []}
    for idffile in filenames:
        data = tools_eis.parse_idf(idffile)
        EISdata['rot']['RE'  ].append(data[     'real_primary'])
        EISdata['rot']['IM'  ].append(data['imaginary_primary'])
        EISdata['rot']['freq'].append(data['frequency_primary'])
        EISdata['rot']['rot'].append(float(idffile.split('_')[7]))
        EISdata['rot']['mat'].append(str(idffile.split('_')[13]))
        EISdata['rot']['hydra'].append(float(idffile.split('_')[15]))

    EISdata['rot'] = numpy_util.numpyify(trig_util.add_polar(EISdata['rot']))

    ##set plotting params--------
    hyd_levels  = [0]
    rot_levels = [0, 90, 180, 270]
    mat_levels  = ['redgar']
    alias = 'allrot'

    FREQ_LIMIT = 300

    mask_sane = EISdata['rot']['freq'][0, :] >= FREQ_LIMIT
    freq_n = mask_sane.sum()
    x = EISdata['rot']['freq'][:, mask_sane]
    y1 = EISdata['rot']['amp'][:, mask_sane]
    y2 = EISdata['rot']['phase'][:, mask_sane]

    ###BODE----------
    fig = plt.subplots(nrows=1, ncols=2, figsize = (12, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['rot']['rot'], rot)
        for i, hyd in enumerate(hyd_levels):
            mask_hyd = np.in1d(EISdata['rot']['hydra'], hyd)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['rot']['mat'], m)
                mask = mask_rot & mask_mat & mask_hyd
                ax1 = plt.subplot(1, 2, 1)
                p1 = ax1.loglog(x[mask], y1[mask], tools.mat_marker_dict[m], color = tools.rot_col_dict[str(rot)],
                           alpha=0.3, label = str(rot), markeredgecolor=tools.rot_col_dict[str(rot)])
                ax2 = plt.subplot(1, 2, 2)
                ax2.semilogx(x[mask], y2[mask], tools.mat_marker_dict[m], color = tools.rot_col_dict[str(rot)],
                           alpha=0.3, label = str(rot), markeredgecolor=tools.rot_col_dict[str(rot)])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('abs(Z)', color = 'b')
    ax1.tick_params('y',colors='b')
    ax1.margins(0.05, 0.05)
    ax1.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n*len(mat_levels)], labels[::freq_n*len(mat_levels)],
                    loc = 'best', frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0.,
                    numpoints = 1, markerscale = 1.5)


    ax2.set_ylabel('Phase (Deg)',color='k')
    ax2.tick_params('y',colors='k')
    ax2.set_ylim([-180,180])
    ax2.margins(0.05, 0.05)
    ax2.grid(True, alpha=0.25)
    lgd1 = ax2.legend(handles[::freq_n*len(mat_levels)], labels[::freq_n*len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)

    plt.suptitle('Bode Plot, Rotation')

    if not os.path.exists(plotdir + 'stability/'):
        os.makedirs(plotdir + 'stability/')
    plt.savefig(plotdir + 'stability/bode_' + alias + '_' + str(FREQ_LIMIT) + '.png')

    ##NYQUIST---------
    fig = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['rot']['rot'], rot)
        for i, hyd in enumerate(hyd_levels):
            mask_hyd = np.in1d(EISdata['rot']['hydra'], hyd)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['rot']['mat'], m)
                mask = mask_rot & mask_mat & mask_hyd
                ax1 = plt.subplot(1, 1, 1)
                ax1.plot(y1[mask] * bf_globals.SCALER, -y2[mask] * bf_globals.SCALER, tools.mat_marker_dict[m],
                         color = tools.rot_col_dict[str(rot)], alpha=0.5, label = str(m),
                         markeredgecolor=tools.rot_col_dict[str(rot)])
                for z in range(len(y1[mask])):
                    ax1.plot(y1[mask][z,:] * bf_globals.SCALER, -y2[mask][z,:] * bf_globals.SCALER, '-',
                             color = tools.rot_col_dict[str(rot)],
                             alpha=0.2)

    plt.margins(0.05, 0.05)
    plt.xlabel('Real(Z)')
    plt.ylabel('-Imag(Z)')
    plt.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n*len(mat_levels)], labels[::freq_n*len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)

    plt.suptitle('Nyquist Plot, Rotation, freq limit ' + str(FREQ_LIMIT))
    plt.savefig(plotdir + 'stability/nyquist_' + alias + '_' + str(FREQ_LIMIT) + '.png')

    ##AMP and PHASE as a func of ROTATION-----------
    FREQ_LIMIT = 300
    mask_sane = EISdata['rot']['freq'][0, :] >= FREQ_LIMIT
    x = EISdata['rot']['freq'][0][mask_sane]
    y1 = EISdata['rot']['amp']
    y2 = EISdata['rot']['phase']

    rot_levels = [0, 90, 180, 270]
    for fq in range(len(x)):
        fig = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
        for r, rot in enumerate(rot_levels):
            mask_rot = np.in1d(EISdata['rot']['rot'], rot)
            for i, hyd in enumerate(hyd_levels):
                mask_hyd = np.in1d(EISdata['rot']['hydra'], hyd)
                for j, m in enumerate(mat_levels):
                    mask_mat = np.in1d(EISdata['rot']['mat'], m)
                    mask = mask_rot & mask_mat & mask_hyd
                    ax1 = plt.subplot(1, 2, 1)
                    plt.plot(EISdata['rot']['rot'][mask], np.log(y1[mask][:, fq]),
                             tools.mat_marker_dict[m], color = tools.rot_col_dict[str(rot)],
                             alpha=0.7, label = str(rot))
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.plot(EISdata['rot']['rot'][mask], y2[mask][:, fq],
                             tools.mat_marker_dict[m], color = tools.rot_col_dict[str(rot)],
                             alpha=0.7, label = str(rot))

        ax1.set_xlabel('rotation')
        ax1.set_ylabel('abs(Z)', color = 'b')
        ax1.tick_params('y',colors='b')
        ax1.margins(0.05, 0.05)
        ax1.grid(True, alpha=0.25)
        ax1.set_xticks([a for a in rot_levels])
        ax1.set_xticklabels([str(a) for a in rot_levels])
        handles, labels = ax1.get_legend_handles_labels()
        handles2 = [plt.plot([], mm, color=cc)
                    for cc, mm in zip(['k', 'k'], [tools.mat_marker_dict[i] for i in mat_levels])]
        lgd2 = ax1.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                          numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))


        ax2.set_xlabel('rotation')
        ax2.set_ylabel('Phase (Deg)',color='k')
        ax2.tick_params('y', colors='k')
        ax2.set_xticks([a for a in rot_levels])
        ax2.set_xticklabels([str(a) for a in rot_levels])
        ax2.set_ylim([-180,180])
        ax2.margins(0.05, 0.05)
        ax2.grid(True, alpha=0.25)
        handles2 = [plt.plot([], mm, color=cc)
                    for cc, mm in zip(['k', 'k'], [tools.mat_marker_dict[i] for i in mat_levels])]
        lgd2 = ax2.legend(sum(handles2, []), mat_levels, fontsize = 12, ncol = len(mat_levels),
                          numpoints = 1, frameon = False, loc='upper center', bbox_to_anchor=(0.5, 1.07))


        plt.suptitle('Amp and Phase vs. Rotation, freq ' + str(x[fq]))
        plt.savefig(plotdir + 'stability/rot_levels_' + alias + '_' + str(fq) + '.png')
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', default="/Volumes/MLIA_active_data/data_barefoot/cal/EIS/", help="Full path to barefoot data calibration directory.")
    parser.add_argument('--plotdir', help="Folder to store resulting plots")

    args = parser.parse_args()

    stability_plotting(args.datadir, args.plotdir)
