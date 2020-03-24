# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:15:48 2018

@author: marchett
"""

import sys, glob
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tools_eis as tools
import plot_EIS as plot
from src.ML.bf_config import bf_globals
import bf_doctest
from src.ML.bf_util import numpy_util
from src.ML.bf_util import trig_util
import argparse
import os

def generate_composition_plots(datadir, plotdir):
    '''
    Inputs:

    Outputs:

    Notes:

    Examples:
    >>> datadir = bf_globals.BAREFOOT_DATA + "/cal/EIS/composition/"
    >>> plotdir = bf_globals.BAREFOOT_ROOT + "/tmp/"
    >>> generate_composition_plots(datadir, plotdir)
    generate_composition_plots datadir: .../cal/EIS/composition/
    <BLANKLINE>
    generate_composition_plots plotdir: ...
    <BLANKLINE>
    '''

    bf_globals.bf_log("generate_composition_plots datadir: " + datadir)
    bf_globals.bf_log("generate_composition_plots plotdir: " + plotdir)

    ###COMPOSITION----------
    filedirs = glob.glob(datadir + '*')
    filenames = []

    # If no filedirs are found, warn the user and return None
    if not filedirs:
        bf_globals.bf_log("WARNING: No filedirs found in generate_composition_plots")
        return None

    for d in filedirs:
        filenames.extend(glob.glob(d + '/' + '*.idf'))
    EISdata = {}

    # If no filenames are found, warn the user and return None
    if not filenames:
        bf_globals.bf_log("WARNING: No filenames found in generate_composition_plots")
        return None

    EISdata['mat'] = {'RE':[],'IM':[],'freq':[], 'hydra':[], 'mat':[], 'rot': []}
    for idffile in filenames:
        data = tools.parse_idf(idffile)
        EISdata['mat']['RE'  ].append(data[     'real_primary'])
        EISdata['mat']['IM'  ].append(data['imaginary_primary'])
        EISdata['mat']['freq'].append(data['frequency_primary'])
        EISdata['mat']['rot'].append(float(idffile.split('_')[6]))
        EISdata['mat']['mat'].append(str(idffile.split('_')[12]))
        EISdata['mat']['hydra'].append(float(idffile.split('_')[14]))

    EISdata['mat'] = numpy_util.numpyify(trig_util.add_polar(EISdata['mat']))

    ##set plotting params--------
    hyd_levels  = [0]
    rot_levels = [0]
    mat_levels = np.unique(EISdata['mat']['mat'])
    mat_levels  = ['mmcrse', 'mmdust', 'mm.2mm', 'bst110']
    alias = 'allmat_mmdust_mm2mm_bst_110_mmcrse_rot0'

    FREQ_LIMIT = 10000

    mask_sane = EISdata['mat']['freq'][0, :] >= FREQ_LIMIT
    freq_n = mask_sane.sum()
    x = EISdata['mat']['freq'][:, mask_sane]
    y1 = EISdata['mat']['amp'][:, mask_sane]
    y2 = EISdata['mat']['phase'][:, mask_sane]

    ###BODE----------
    fig = plt.subplots(nrows=1, ncols=2, figsize = (12, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['mat']['rot'], rot)
        for i, hyd in enumerate(hyd_levels):
            mask_hyd = np.in1d(EISdata['mat']['hydra'], hyd)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['mat']['mat'], m)
                mask = mask_rot & mask_mat & mask_hyd
                ax1 = plt.subplot(1, 2, 1)
                p1 = ax1.loglog(x[mask], y1[mask], plot.mat_marker_dict[m], color = plot.mat_col_dict[m],
                           alpha=0.3, label = str(m), markeredgecolor=plot.mat_col_dict[m])
                ax2 = plt.subplot(1, 2, 2)
                ax2.semilogx(x[mask], y2[mask], plot.mat_marker_dict[m], color = plot.mat_col_dict[m],
                           alpha=0.3, label = str(m), markeredgecolor=plot.mat_col_dict[m])

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('abs(Z)', color = 'b')
    ax1.tick_params('y',colors='b')

    ax1.margins(0.05, 0.05)
    ax1.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n][:len(mat_levels)], labels[::freq_n][:len(mat_levels)],
                    loc = 'best', frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0.,
                    numpoints = 1, markerscale = 1.5)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (Deg)',color='k')
    ax2.tick_params('y',colors='k')

    ax2.set_ylim([-180,180])
    ax2.margins(0.05, 0.05)
    ax2.grid(True, alpha=0.25)
    lgd_idx = freq_n*len(hyd_levels)
    lgd1 = ax2.legend(handles[::freq_n][:len(mat_levels)],
                      labels[::freq_n][:len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)

    plt.suptitle('Bode Plot, Composition' + ', rot ' + str(rot_levels))
    if not os.path.exists(plotdir + "/composition"):
        os.makedirs(plotdir + "/composition")
    plt.savefig(plotdir + 'composition/bode_' + alias + '_f' + str(FREQ_LIMIT) + '.png')
    plt.close()

    ##NYQUIST----------
    fig = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
    for r, rot in enumerate(rot_levels):
        mask_rot = np.in1d(EISdata['mat']['rot'], rot)
        for i, hyd in enumerate(hyd_levels):
            mask_hyd = np.in1d(EISdata['mat']['hydra'], hyd)
            for j, m in enumerate(mat_levels):
                mask_mat = np.in1d(EISdata['mat']['mat'], m)
                mask = mask_rot & mask_mat & mask_hyd
                ax1 = plt.subplot(1, 1, 1)
                ax1.plot(y1[mask], -y2[mask], plot.mat_marker_dict[m],
                         color = plot.mat_col_dict[m], alpha=0.5, label = str(m),
                         markeredgecolor=plot.mat_col_dict[m])
                for z in range(len(y1[mask])):
                    ax1.plot(y1[mask][z,:]* bf_globals.SCALER, -y2[mask][z,:] * bf_globals.SCALER, '-', color = plot.mat_col_dict[m],
                             alpha=0.2)

    plt.margins(0.05, 0.05)
    plt.xlabel('Real(Z)')
    plt.ylabel('-Imag(Z)')
    plt.grid(True, alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles[::freq_n][:len(mat_levels)], labels[::freq_n][:len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)

    plt.suptitle('Nyquist Plot, Composition, freq limit ' + str(FREQ_LIMIT) + ', rot ' + str(rot_levels))
    plt.savefig(plotdir + 'composition/nyquist_' + alias + '_f' + str(FREQ_LIMIT) + '.png')


    ##AMP and PHASE as a func of COMPOSITION-----------
    x = EISdata['mat']['freq'][0]
    y1 = EISdata['mat']['amp']
    y2 = EISdata['mat']['phase']
    rot_levels = [0, 90, 180, 270]
    alias = 'allrot'

    mat_levels = np.unique(EISdata['mat']['mat'])
    isort = np.argsort([plot.grain_size_dict[i] for i in mat_levels])
    mat_levels = [mat_levels[i] for i in isort]
    for fq in range(len(x)):
        fig = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
        for r, rot in enumerate(rot_levels):
            mask_rot = np.in1d(EISdata['mat']['rot'], rot)
            for i, hyd in enumerate(hyd_levels):
                mask_hyd = np.in1d(EISdata['mat']['hydra'], hyd)
                for j, m in enumerate(mat_levels):
                    mask_mat = np.in1d(EISdata['mat']['mat'], m)
                    mask = mask_rot & mask_mat & mask_hyd
                    ax1 = plt.subplot(1, 2, 1)
                    plt.plot(np.repeat(j, mask.sum()), np.log(y1[mask][:, fq]),
                             plot.mat_marker_dict[m], color = plot.mat_col_dict[str(m)],
                             alpha=0.7, label = str(rot))
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.plot(np.repeat(j, mask.sum()), y2[mask][:, fq],
                             plot.mat_marker_dict[m], color = plot.mat_col_dict[str(m)],
                             alpha=0.7, label = str(rot))
        ax1.set_xlabel('material')
        ax1.set_ylabel('abs(Z)', color = 'b')
        ax1.tick_params('y',colors='b')

        ax1.margins(0.05, 0.05)
        ax1.grid(True, alpha=0.25)
        ax1.set_xticks(np.arange(0, len(mat_levels)))
        ax1.set_xticklabels([str(a) + '\n' + plot.grain_size_dict[a] for a in mat_levels], rotation = 70)
        handles, labels = ax1.get_legend_handles_labels()

        ax2.set_xlabel('material')
        ax2.set_ylabel('Phase (Deg)',color='k')
        ax2.tick_params('y', colors='k')
        ax2.set_xticks([a for a in rot_levels])
        ax2.set_xticklabels([str(a) for a in rot_levels])
        ax2.set_ylim([-180,180])
        ax2.margins(0.05, 0.05)
        ax2.grid(True, alpha=0.25)
        ax2.set_xticks(np.arange(0, len(mat_levels)))
        ax2.set_xticklabels([str(a) + '\n' + plot.grain_size_dict[a] for a in mat_levels], rotation = 70)


        plt.suptitle('Amp and Phase vs. Material, freq ' + str(x[fq]) + ', rot ' + str(rot_levels))
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(plotdir + 'composition/mat_levels_' + alias + '_' + str(fq) + '.png')
        plt.close()


    ##AMP and PHASE as a func of ROTATION-----------
    x = EISdata['mat']['freq'][0]
    y1 = EISdata['mat']['amp']
    y2 = EISdata['mat']['phase']
    mat_levels = np.unique(EISdata['mat']['mat'])
    mat_levels  = ['mmcrse', 'mmdust', 'mm.2mm', 'bst110']
    alias = 'mmdust_mm2mm_bst110_mmcrse'

    rot_levels = [0, 90, 180, 270]
    for fq in range(len(x)):
        fig = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
        for r, rot in enumerate(rot_levels):
            mask_rot = np.in1d(EISdata['mat']['rot'], rot)
            for i, hyd in enumerate(hyd_levels):
                mask_hyd = np.in1d(EISdata['mat']['hydra'], hyd)
                for j, m in enumerate(mat_levels):
                    mask_mat = np.in1d(EISdata['mat']['mat'], m)
                    mask = mask_rot & mask_mat & mask_hyd
                    ax1 = plt.subplot(1, 2, 1)
                    plt.plot(EISdata['mat']['rot'][mask], np.log(y1[mask][:, fq]),
                             plot.mat_marker_dict[m], color = plot.mat_col_dict[str(m)],
                             alpha=0.7, label = str(m))
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.plot(EISdata['mat']['rot'][mask], y2[mask][:, fq],
                             plot.mat_marker_dict[m], color = plot.mat_col_dict[str(m)],
                             alpha=0.7, label = str(m))

        ax1.set_xlabel('rotation')
        ax1.set_ylabel('abs(Z)', color = 'b')
        ax1.tick_params('y',colors='b')

        ax1.margins(0.05, 0.05)
        ax1.grid(True, alpha=0.25)
        ax1.set_xticks([a for a in rot_levels])
        ax1.set_xticklabels([str(a) for a in rot_levels])
        col_vec = np.repeat('k', len(mat_levels))
        handles, labels = ax1.get_legend_handles_labels()
        lgd1 = ax1.legend(handles[:len(mat_levels)], labels[:len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)


        ax2.set_xlabel('rotation')
        ax2.set_ylabel('Phase (Deg)',color='k')
        ax2.tick_params('y', colors='k')
        ax2.set_xticks([a for a in rot_levels])
        ax2.set_xticklabels([str(a) for a in rot_levels])
        ax2.set_ylim([-180,180])
        ax2.margins(0.05, 0.05)
        ax2.grid(True, alpha=0.25)
        col_vec = np.repeat('k', len(mat_levels))
        handles, labels = ax2.get_legend_handles_labels()
        lgd1 = ax2.legend(handles[:len(mat_levels)], labels[:len(mat_levels)], loc = 'best',
                    frameon=False, fontsize = 'small', ncol = 2, borderaxespad=0., numpoints = 1,
                    markerscale = 1.5)


        plt.suptitle('Amp and Phase vs. Rotation, freq ' + str(x[fq]))
        plt.savefig(plotdir + 'composition/rot_levels_' + alias + '_' + str(fq) + '.png')
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', default="/Volumes/MLIA_active_data/data_barefoot/cal/EIS/composition/", help="Full path to barefoot data calibration directory.")
    parser.add_argument('--plotdir', help="Full path to directory for storing resulting plots.")

    args = parser.parse_args()

    generate_composition_plots(args.datadir, args.plotdir)
