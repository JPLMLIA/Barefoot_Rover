# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:38:37 2018

@author: marchett


Goes run by run and generate movies with
the contact area, sharp and pressure lean

"""
import argparse
import glob
import logging
import multiprocessing
import os
import subprocess
import sys
import tqdm

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.neighbors import KDTree

import bf_system
from bf_config import bf_globals
from bf_logging import bf_log
from bf_plot import plot_EIS, plots_pg
from bf_tools import tools_eis, tools_pg

plt.set_cmap('afmhot')

bf_log.setup_logger()
logger = logging.getLogger(__name__)


def make_experiment_plot(args):
    """"  Description

    Parameters
    ----------
    datadir : str
        Root of Barefoot Data.  datadir + experiment should put you in experiment folder.
    Experiment : str
        Experiment folder name, relative to datadir.
    replace : bool
        Overwrite data if true.
    skip : bool
        Do not create the movie if true.
    predict : bool
        If true, add prediction with current models to plot.
    module : str
        'rock' or 'slip'
    """
    experiment = args["experiment"]
    replace = args["replace"]
    skip = args["skip"]
    datadir = args["datadir"]
    predict = args["predict"]
    module = args['module']
    featureFile = args['feature_file']

    plotdir = f"{experiment}/plots/"
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    movie_file = glob.glob(f"{plotdir}/status_movie*.mp4")
    if movie_file:
        logger.info('Movie exists')
        if replace:
            os.remove(movie_file[0])
        else:
            return

    framepath = f"{plotdir}/movie_frames/"
    if not os.path.exists(framepath):
        os.makedirs(framepath)

    # read and align data ---
    logger.info(experiment)
    ftype, data, time = tools_pg.read_all_pg_data(
        experiment, f"{datadir}/cal/Xiroku_no_contact/")
    fdate = experiment.split('_')[-3]
    data_binned, time_binned = tools_pg.align_pg_to_imu(
        data, time, bf_globals.T, bf_globals.R)

    # eis files
    files_eis = glob.glob(f"{experiment}/**/*.idf", recursive=True)
    logger.info(f"Processing {len(files_eis)} EIS files")
    EISdata, _ = tools_eis.read_eis(files_eis, time_binned, data_binned['imu'])

    # remove non-stationary ----
    rot_mask = tools_pg.rot_start_stop_motor(data_binned['current'], data_binned['imu'])
    time_binned=time_binned[~rot_mask]
    for k in data_binned.keys():
        data_binned[k]=data_binned[k][~rot_mask]
    imu_contact_bin=tools_pg.contact_imu(data_binned['imu'])
    logger.info(f"Datapoints after align: {data_binned['pg'].shape[0]}")

    if predict:
        burnin=[50, None]
        run=experiment.replace(datadir, '')
        logger.info("Starting prediction...")
        data_pred=bf_system.run_system(datadir, plotdir, run,
                                       [str(burnin[0]), 'None'],
                                       None, module, featureFile)
        logger.info("Finished prediction...")
        mask_pred=np.repeat(False, len(data_binned['pg']))
        mask_pred[burnin[0]:]=data_pred['nan_mask']
        for k in list(data_pred)[:-1]:
            rows=len(data_binned['pg'])
            if len(data_pred[k].shape) > 1:
                cols=data_pred[k].shape[1]
                temp=np.empty((rows, cols))
                temp[:]=np.nan
            else:
                temp=np.empty((rows,))
                temp[:]=np.nan
            if len(data_pred[k]) == mask_pred.sum():
                if k == 'Composition':
                    temp=temp.astype(str)
                    temp[mask_pred]=data_pred[k]
                else:
                    temp[mask_pred]=data_pred[k]

            if len(data_pred[k]) == len(data_pred['nan_mask']):
                temp[burnin[0]:]=data_pred[k]
            data_pred[k]=temp
    else:
        logger.info("No prediction")
        data_pred=None

    # contact area ----
    contact_data=tools_pg.contact_area_run(data_binned['pg'], imu_contact_bin)

    # imu unwrap ----
    lag=[4, 3, 2, 1, 0, -1, -2, -3, -4]
    pg_imu, rho_imu, x_imu=tools_pg.unwrap_pg_to_imu(data_binned['pg'], data_binned['imu'], time_binned,
                                                 lag, extraction='all')
    # images ----
    jpgs=glob.glob(experiment + '/*[0-9]/*.jpg')
    if len(jpgs) == 0:
        jpg_time=None
    else:
        jpg_time=np.hstack(
            [float('.'.join(a.split('/')[-1].split('.')[0].split('_'))) for a in jpgs])
        X=time_binned.reshape((-1, 1))
        X1=jpg_time.reshape((-1, 1))
        tree=KDTree(X1)
        ndist, ind=tree.query(X, k=1)

    # detect sharp points
    sharp=tools_pg.sharpPG(data_binned['pg'], contact_data)

    # detect leaning
    lean=tools_pg.leanPG(data_binned['pg'], contact_data)

    # maximum per row of pg
    colmax=data_binned['pg'].max(axis=1)

    # create movie plots
    nt=data_binned['pg'].shape[0]
    if not skip:
        logger.info('Making movie frames...')

        for t in tqdm.trange(nt):
            if jpg_time is None:
                jpg_match=None
            else:
                jpg_match=jpgs[ind[t][0]]
            logger.info("Creating animated figure")
            anifig=plots_pg.animatedFigure(data_binned, t, nt, contact_data,
                                          sharp, lean, imu_contact_bin, colmax, jpg_match,
                                          pg_imu, x_imu, lag, plotdir, data_pred)
            logger.info(f"Saving animated figure {t:5d}")
            anifig.savefig(f'{framepath}/{t:05d}.png')
            plt.close(anifig)

        mp4name=f'status_movie_{ftype}_{fdate}'
        framepath_split=framepath.split('&')
        if len(framepath_split) > 1:
            framepath=r'\&'.join(framepath.split('&'))
            mp4name=r'\&'.join(mp4name.split('&'))
        mp4file_path=f"{'/'.join(framepath.split('/')[:-2])}/{mp4name}.mp4"

        movie_path=glob.glob(f"{plotdir}/status*")
        if len(movie_path) > 0:
            os.remove(movie_path[0])

        command=(f"cd {framepath} && "
                 f"ffmpeg -r 4 -i %05d.png -c:v "
                 f"libx264 -r 25 -pix_fmt "
                 f"yuv420p ../{mp4name}.mp4")

        subprocess.check_call(command, shell=True)
        filelist=glob.glob(os.path.join(
            '&'.join(framepath.split('\\&')), '*.png'))
        for f in filelist:
            os.remove(f)

    # contact plot -----
    plot_path=glob.glob(f"{plotdir}contact*")
    if len(plot_path) > 0:
        os.remove(plot_path[0])
    logger.info("Creating contact-area-plot")
    fig=plots_pg.contactAreaPlot(contact_data, sharp, lean, ftype)
    logger.info("Saving contact-area-plot")
    plt.savefig(
        f"{plotdir}{'_'.join(['contact', str(bf_globals.T), str(bf_globals.R), ftype, fdate])}.png")
    plt.close()

    plot_path=glob.glob(f"{plotdir}imu*")
    if len(plot_path) > 0:
        for j in range(len(plot_path)):
            os.remove(plot_path[j])

    # imu unwrap plot ----
    extract=['grouser', 'nongrouser', 'all']
    for ename in extract:

        pg_imu, rho_imu, x=tools_pg.unwrap_pg_to_imu(data_binned['pg'], data_binned['imu'],
                                                 time_binned, lag, extraction=ename)

        logger.debug(f"Creating {ename} plot for unwrappedGrid")
        fig=plots_pg.unwrappedGrid(lag, pg_imu, x)
        plt.suptitle(
            f"imu matched pressure, time bin  {bf_globals.T}, imu mean {bf_globals.R}, {ftype}")
        logger.debug(f"Saving {ename} plot for unwrappedGrid")
        plt.savefig(
            f"{plotdir}{'_'.join(['imu_unwrap', ename, str(bf_globals.T), str(bf_globals.R), ftype, fdate])}.png")
        plt.close()

        # wavelet smoothing/degrousing
        # use bior2.2 level 2 or db2 level 2
        logger.info(f"Starting wavelet smoothing/degrousing for {ename}")
        pg_recon=tools_pg.waveletDegrouse(pg_imu, wtype='db2', level=2)
        logger.info(f"Finished wavelet smoothing/degrousing for {ename}")

        fig=plots_pg.unwrappedGrid(lag, pg_recon, x)
        plt.suptitle(
            f"Wavelet recon pressure, time bin {bf_globals.T}, imu mean {bf_globals.R}, {ftype}")
        plt.savefig(
            f"{plotdir}{'_'.join(['imu_unwrap_wsmoo', ename, str(bf_globals.T), str(bf_globals.R), ftype])}.png")
        plt.close()

    logger.info("Creating predition plot")
    fig=plots_pg.plotPredictions(data_binned, nt, data_pred)
    logger.info("Saving prediction plot")
    plt.savefig(f"{plotdir}{'_'.join(['rock_lkhd', ftype])}.png")
    plt.close()

    # slip plots
    n_rot, mean_slip=tools_pg.number_rot(data_binned['imu'], d=200.)
    mask_out=tools_pg.clean_slip(data_binned['slip'])

    x=np.arange(0, len(data_binned['slip']))

    sigma=10
    y_pot=data_binned['slip'][mask_out]
    x_pot=x[mask_out]
    y_gf_pot=ndimage.gaussian_filter1d(y_pot, sigma)
    err_pot=np.sqrt(np.mean((y_pot - y_gf_pot)**2))

    mask_nan=np.isnan(data_binned['slip_fiducials'][mask_out])
    y_fid=data_binned['slip_fiducials'][mask_out][~mask_nan]
    x_fid=x[mask_out][~mask_nan]
    y_gf_fid=ndimage.gaussian_filter1d(y_fid, sigma)
    err_fid=np.sqrt(np.mean((y_fid - y_gf_fid)**2))

    plot_path=glob.glob(f"{plotdir}slip*")
    if len(plot_path) > 0:
        os.remove(plot_path[0])

    mean_pot=np.mean(y_gf_pot)
    mean_fid=np.mean(y_gf_fid)
    fig, ax=plt.subplots()
    plt.plot(x, data_binned['slip'], '.', ms=2,
             color='0.3', alpha=0.7, label='pot')
    plt.plot(x, data_binned['slip_fiducials'], 'x',
             color='0.5', alpha=0.7, label='fid', ms=5)
    plt.plot(x_pot, y_gf_pot, '.', ms=1, color='0.3')
    plt.plot(x_fid, y_gf_fid, '.', ms=1, color='0.5')
    plt.xlabel('time')
    plt.ylabel('slip (%)')

    if data_pred is not None:
        mean_pred=np.nanmean(data_pred['Slip'])
        nan_pred=~np.isnan(data_pred['Slip'])
        y_pred=data_pred['Slip'][nan_pred]
        y_gf_pred=ndimage.gaussian_filter1d(y_pred, sigma)
        x_pred=np.arange(0, len(data_pred['Slip']))
        plt.plot(x_pred, data_pred['Slip'], '.',
                 ms=2, color='b', label='predicted')
        plt.plot(x_pred, data_pred['Slip'], '-', lw=0.5, color='b', alpha=0.3)
        plt.plot(x_pred[nan_pred], y_gf_pred, '.', ms=1, color='b')

    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    plt.title('slip %, error: fid ' + '%0.3f' %
              err_fid + ', pot ' + '%0.3f' % err_pot)
    plt.suptitle(ftype)
    plt.ylim((-0.5, 1))
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"{plotdir}{'_'.join(['slip', ftype])}.png")
    plt.close()

    fig, ax=plt.subplots()
    plt.plot(x, data_binned['sink'] * 0.01, '.',
             color='0.3', alpha=0.7, label='zstage (relative)')
    plt.ylim((-50, 50))
    plt.ylabel('sink (mm)')
    plt.xlabel('time')

    if data_pred is not None:
        x_pred=np.arange(0, len(data_pred['Sinkage']))
        plt.plot(x_pred, data_pred['Sinkage'], '.',
                 ms=2, color='b', label='predicted')
        plt.plot(x_pred, data_pred['Sinkage'], '-', lw=1, color='b', alpha=0.3)

    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    plt.legend(loc='best', fontsize=8)
    plt.title('sinkage')
    plt.savefig(f"{plotdir}{'_'.join(['sink', ftype])}.png")
    plt.close()

    cc=[]
    for i in range(data_binned['ft_xyz'].shape[1]):
        cc.append(np.corrcoef(
            data_binned['slip'], data_binned['ft_xyz'][:, i])[0, 1])
    cc_max=np.argmax(np.abs(cc))

    fig, ax1=plt.subplots(5, 1, figsize=(8, 8))
    plt.subplot(511)
    plt.plot(data_binned['current'])
    plt.ylabel('current')
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    ax2=ax1[0].twinx()
    ax2.plot(data_binned['slip'], color='0.7')
    ax2.set_ylabel('slip')
    plt.subplot(512)
    plt.plot(data_binned['slip'], data_binned['current'], '.')
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    plt.subplot(513)
    plt.plot(data_binned['slip'], data_binned['ft_xyz'][:, cc_max], '.',
             label='%0.2f' % np.corrcoef(data_binned['slip'], data_binned['current'])[0, 1])
    plt.xlabel('slip')
    plt.ylabel('high corr FT')
    plt.legend(fontsize=12)
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    plt.subplot(514)
    # np.corrcoef(data_binned['slip'], data_binned['ft_xyz'][:, 5])
    plt.plot(data_binned['ft_xyz'][:, 0], label='f x: ' + '%0.2f' % cc[0])
    plt.plot(data_binned['ft_xyz'][:, 1], label='f y: ' + '%0.2f' % cc[1])
    plt.plot(data_binned['ft_xyz'][:, 2], label='f z: ' + '%0.2f' % cc[2])
    plt.legend(fontsize=10)
    plt.xlabel('time')
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    plt.subplot(515)
    plt.plot(data_binned['ft_xyz'][:, 3], label='t x: ' + '%0.2f' % cc[3])
    plt.plot(data_binned['ft_xyz'][:, 4], label='t y: ' + '%0.2f' % cc[4])
    plt.plot(data_binned['ft_xyz'][:, 5], label='t z: ' + '%0.2f' % cc[5])
    plt.legend(fontsize=10)
    plt.xlabel('time')
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
    plt.grid(True, which='major', axis='both', alpha=0.3)
    plt.suptitle('telemetry correlations')
    plt.savefig(f"{plotdir}{'_'.join(['telemetry', ftype])}.png")
    plt.close()


    # -------- EIS plots
    if EISdata is not None:
        logger.info("Creating plots for EIS data")
        from sklearn.metrics import confusion_matrix, accuracy_score
        true_hyd=np.unique(EISdata['hyd'])
        if np.isnan(true_hyd).sum() != len(true_hyd):
            logger.info("Creating plots for hydration data")
            nan_mask=~np.isnan(data_pred['Hydration'][:, 2])
            pred_hyd=data_pred['Hydration'][:, 2]
            acc=accuracy_score(
                np.repeat(true_hyd, nan_mask.sum()), pred_hyd[nan_mask])
            dry_prop=(pred_hyd == 0).sum() / nan_mask.sum()

            fig, ax1=plt.subplots(figsize=(10, 4))
            plt.plot(data_pred['Hydration'][:, 2], '.')
            plt.axhline(true_hyd, ls='--', color='r', alpha=0.5)
            plt.axhline(0, ls='--', color='0.5', alpha=0.5)
            plt.yticks(ticks=[0, 1, 3, 5, 10, 15], labels=[0, 1, 3, 5, 10, 15])
            plt.xlabel('time')
            plt.ylabel('hydration percentage')
            ax2=ax1.twinx()
            ax2.plot(data_pred['Hydration'][:, 1],
                     ls=':', color='0.5', alpha=0.5)
            ax2.set_ylim((0, 1))
            plt.ylabel('likelihood for chosen class')
            plt.minorticks_on()
            ax1.grid(True, which='minor', axis='both', alpha=0.3, ls=':')
            ax1.grid(True, which='major', axis='both', alpha=0.3)
            plt.title('true ratio = ' + '%.2f' %
                      acc + ', dry ratio: ' + '%.2f' % dry_prop)
            logger.info("Saving plot for hydration data")
            plt.savefig(f"{plotdir}{'_'.join(['hyd_predicted', ftype])}.png")
            plt.close()

            logger.info("Creating Bode plots")
            fig=plot_EIS.plotBodeEIS_byRot(EISdata)
            logger.info("Saving bode plots")
            plt.savefig(
                f"{plotdir}{'_'.join(['amplitude_vs_rot', ftype])}.png")
            plt.close()

            logger.info("Creating frequency plots")
            fig=plot_EIS.plotFreqEIS_byTime(EISdata)
            logger.info("Saving frequency plots")
            plt.savefig(
                f"{plotdir}{'_'.join(['amplitude_vs_time', ftype])}.png")
            logger.debug('Finished saving frequency plot')
            plt.close()


def make_diagnostic_plots(datadir, subfolders, regex, featureFile, *,
                          skip, replace, module, predict):
    files=[glob.glob(f"{datadir}/{a}/{b}/") for a, b in zip(subfolders, regex)]
    for subfiles, subfolder in zip(files, subfolders):
        if not subfiles:
            logger.warning(
                (f"No files found for diagnostic plotting in "
                 f"{datadir}/{subfolder} subfolder"))
    files=np.sort(np.hstack(files))

    iterable=[]
    for file in files:
        iterable.append({"experiment": file, "replace": replace,
                        'datadir': datadir, 'predict': predict,
                        'module': module, 'skip': skip,
                        'feature_file': featureFile})

    with multiprocessing.Pool(processes=bf_globals.THREADS) as pool:
        pool.map(make_experiment_plot, iterable)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir',    default="/Volumes/MLIA_active_data/data_barefoot/",
                                        help="Full path to barefoot data directory.")

    parser.add_argument('--replace',    default=False,
                                        help="Overwrite existing plots.  Defaults to false.")

    parser.add_argument('--skip',       default=False,
                                        help="Do not create a movie.  Defaults to false.")

    parser.add_argument('--subfolders', nargs='+',
                        help="Options include: rock_detection, composition, data_andrew")
    parser.add_argument('--regex', default="*", nargs='+',
                        help="list of regex strings.")
    parser.add_argument('--module', default='rock', help="")
    parser.add_argument('--predict', default=False,
                        help="Make prediction.  Defaults to false.")
    parser.add_argument('--featureFile', default="full_features.txt",
                        help="File of line-seperated features relative to datadir.")


    args=parser.parse_args()

    make_diagnostic_plots(args.datadir, args.subfolders, args.regex, args.featureFile,
                         skip=args.skip,replace=args.replace,
                         module=args.module, predict=args.predict)
