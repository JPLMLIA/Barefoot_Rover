# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:45:19 2018

@author: marchett
"""

import sys, glob
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tools_eis
import tools_pg as tools
from src.ML.bf_util import numpy_util
from src.ML.bf_util import trig_util
import argparse
from src.ML.bf_config import bf_globals

def depth_plotting(datadir, plotdir):
    '''
    Inputs:

    Outputs:

    Notes:

    Examples:

    '''

    ###DEPTH ----------
    filepath = datadir + '/depth/'
    filedirs = glob.glob(filepath + '*')
    filenames = []
    for d in filedirs:
        filenames.extend(glob.glob(d + '/' + '*.idf'))
    EISdata = {}

    EISdata['dep'] = {'RE':[],'IM':[],'freq':[], 'hydra':[], 'mat':[], 'rot': [], 'dep': []}
    for idffile in filenames:
        data = tools_eis.parse_idf(idffile)
        EISdata['dep']['RE'  ].append(data[     'real_primary'])
        EISdata['dep']['IM'  ].append(data['imaginary_primary'])
        EISdata['dep']['freq'].append(data['frequency_primary'])
        EISdata['dep']['rot'].append(float(idffile.split('_')[6]))
        EISdata['dep']['mat'].append(str(idffile.split('_')[12]))
        EISdata['dep']['hydra'].append(float(idffile.split('_')[14]))
        EISdata['dep']['dep'].append(float(idffile.split('/')[11].split('mm')[0]))

    EISdata['dep'] = numpy_util.numpyify(trig_util.add_polar(EISdata['dep']))

    ##set plotting params--------
    rot_levels = [0]
    dep_levels  = [0, 10, 50]
    mat_levels  = ['bucket', 'redgar']
    alias = 'dep_0_10_50'

    FREQ_LIMIT = 10000

    mask_sane = EISdata['dep']['freq'][0, :] >= FREQ_LIMIT
    freq_n = mask_sane.sum()
    x = EISdata['dep']['freq'][:, mask_sane]
    y1 = EISdata['dep']['amp'][:, mask_sane]
    y2 = EISdata['dep']['phase'][:, mask_sane]

    ###BODE----------
    fig = plt.subplots(nrows=1, ncols=2, figsize = (12, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['dep']['rot'], rot)
        for i, dep in enumerate(dep_levels):
            mask_dep = np.in1d(EISdata['dep']['dep'], dep)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['dep']['mat'], m)
                mask = mask_rot & mask_mat & mask_dep
                ax1 = plt.subplot(1, 2, 1)
                p1 = ax1.loglog(x[mask], y1[mask], tools.dep_marker_dict[str(dep)], color = tools.dep_col_dict[str(dep)],
                           alpha = 0.3, label = str(dep), markeredgecolor=tools.dep_col_dict[str(dep)])
                ax2 = plt.subplot(1, 2, 2)
                ax2.semilogx(x[mask], y2[mask], tools.dep_marker_dict[str(dep)], color = tools.dep_col_dict[str(dep)],
                           alpha = 0.3, label = str(dep), markeredgecolor=tools.dep_col_dict[str(dep)])

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
    ax1.set_xlabel('Frequency (Hz)')
    ax2.tick_params('y',colors='k')
    ax2.set_ylim([-180,180])
    ax2.margins(0.05, 0.05)
    ax2.grid(True, alpha=0.25)
    lgd1 = ax2.legend(handles[::freq_n*len(mat_levels)], labels[::freq_n*len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)

    plt.suptitle('Bode Plot, Depth')
    plt.savefig(plotdir + 'depth/bode_' + alias + '_f' + str(FREQ_LIMIT) + '.png')

    ##NYQUIST----------
    fig = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['dep']['rot'], rot)
        for i, dep in enumerate(dep_levels):
            mask_dep = np.in1d(EISdata['dep']['dep'], dep)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['dep']['mat'], m)
                mask = mask_rot & mask_mat & mask_dep
                ax1 = plt.subplot(1, 1, 1)
                ax1.plot(y1[mask], -y2[mask], tools.dep_marker_dict[str(dep)],
                         color = tools.dep_col_dict[str(dep)], alpha=0.5, label = str(dep),
                         markeredgecolor=tools.dep_col_dict[str(dep)])
                for z in range(len(y1[mask])):
                    ax1.plot(y1[mask][z,:], -y2[mask][z,:], '-',
                             color = tools.dep_col_dict[str(dep)],
                             alpha=0.2)

    plt.margins(0.05, 0.05)
    plt.xlabel('Real(Z)')
    plt.ylabel('-Imag(Z)')
    plt.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n][::len(mat_levels)], labels[::freq_n][::len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)
    #plt.subplots_adjust(left=0.2)
    plt.suptitle('Nyquist Plot, Depth, freq limit ' + str(FREQ_LIMIT) + ', rot ' + str(rot_levels))
    plt.savefig(plotdir + 'depth/nyquist_' + alias + '_f' + str(FREQ_LIMIT) + '.png')

    ##AMP and PHASE as a func of DEPTH-----------
    x = EISdata['dep']['freq'][0]
    y1 = EISdata['dep']['amp']
    y2 = EISdata['dep']['phase']
    rot_levels = [0]
    alias = 'all'
    mat_levels = np.unique(EISdata['dep']['mat'])
    dep_levels = np.unique(EISdata['dep']['dep']).astype(int)

    for fq in range(len(x)):
        fig = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
        for r, rot in enumerate(rot_levels):
            mask_rot = np.in1d(EISdata['dep']['rot'], rot)
            for i, dep in enumerate(dep_levels):
                mask_dep = np.in1d(EISdata['dep']['dep'], dep)
                for j, m in enumerate(mat_levels):
                    mask_mat = np.in1d(EISdata['dep']['mat'], m)
                    mask = mask_rot & mask_mat & mask_dep
                    ax1 = plt.subplot(1, 2, 1)
                    plt.plot(EISdata['dep']['dep'][mask], np.log(y1[mask][:, fq]),
                             tools.dep_marker_dict[str(dep)], color = tools.dep_col_dict[str(dep)],
                             alpha=0.7, label = str(rot))
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.plot(EISdata['dep']['dep'][mask], y2[mask][:, fq],
                             tools.dep_marker_dict[str(dep)], color = tools.dep_col_dict[str(dep)],
                             alpha=0.7, label = str(rot))
        ax1.set_xlabel('depth')
        ax1.set_ylabel('abs(Z)', color = 'b')
        ax1.tick_params('y',colors='b')
        ax1.margins(0.05, 0.05)
        ax1.grid(True, alpha=0.25)
        ax1.set_xticks(dep_levels)
        ax1.set_xticklabels([str(a) for a in dep_levels], rotation = 70)
        handles, labels = ax1.get_legend_handles_labels()

        ax2.set_xlabel('depth')
        ax2.set_ylabel('Phase (Deg)',color='k')
        ax2.tick_params('y', colors='k')
        ax2.set_xticks(dep_levels)
        ax2.set_xticklabels([str(a) for a in dep_levels], rotation = 70)
        ax2.set_ylim([-180,180])
        ax2.margins(0.05, 0.05)
        ax2.grid(True, alpha=0.25)

        plt.suptitle('Amp and Phase vs. Depth, freq ' + str(x[fq]) + ', rot ' + str(rot_levels))
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(plotdir + 'depth/dep_levels_' + alias + '_' + str(fq) + '.png')
        plt.close()


if __name__== "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', default="/Volumes/MLIA_active_data/data_barefoot/cal/EIS/", help="Full path to barefoot data calibration directory.")
    parser.add_argument('--plotdir', help="Full path to directory for storing resulting plots.")

    args = parser.parse_args()

    depth_plotting(args.datadir, args.plotdir)
