import glob
from os.path import exists

import glob
import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir

from bf_config import bf_globals
from bf_tools import tools_pg
from bf_tools.tools_pg import ExperimentMetadata
from plot_diagnostic import make_diagnostic_plots, make_experiment_plot


def test_make_experiment_plot(shared_datadir, full_exp_dir_setup,
                         set_global_model_params, test_config):
    subfolders = ['full_experiment']
    regex = "*"
    files = [glob.glob(f"{str(shared_datadir)}/{a}/{b}/")
             for a, b in zip(subfolders, regex)]
    files = np.sort(np.hstack(files))
    # There is only one file
    for file in files:
        args = {"experiment": file, "replace": True,
                         'datadir': str(shared_datadir), 'predict': True,
                         'module': 'slip', 'skip': False,
                         'feature_file': f'{str(shared_datadir)}/full_features_2.txt'}
        make_experiment_plot(args)

    check_plot_files(test_config['experiments']['slip'],
                     str(shared_datadir))


def test_make_diagnostic_plots(shared_datadir, full_exp_dir_setup,
                              set_global_model_params, test_config):

    make_diagnostic_plots(str(shared_datadir), ['full_experiment'],'*',
                          f'{str(shared_datadir)}/full_features_2.txt',
                          skip=False, replace=True, module='slip',
                          predict=True)
    check_plot_files(test_config['experiments']['slip'],
                     str(shared_datadir))



def check_plot_files(exp: ExperimentMetadata, data_dir: str):
    ftype = tools_pg.extract_exp_metadata(exp)
    plotdir = f"{data_dir}/full_experiment/{exp}/plots"
    frame_path = f"{plotdir}/movie_frames"
    fdate = ftype.date

    plot_files = [(f"{plotdir}/contact_{str(bf_globals.T)}_"
                   f"{str(bf_globals.R)}_{ftype}_{fdate}.png")]
    plot_files.extend([(f"{plotdir}/imu_unwrap_{ename}_"
                        f"{str(bf_globals.T)}_{str(bf_globals.R)}_"
                        f"{ftype}_{fdate}.png")
                       for ename in ['grouser', 'nongrouser', 'all']])
    plot_files.extend([(f"{plotdir}/imu_unwrap_wsmoo_{ename}_"
                        f"{str(bf_globals.T)}_{str(bf_globals.R)}_"
                        f"{ftype}.png")
                       for ename in ['grouser', 'nongrouser', 'all']])
    plot_files.extend([f"{plotdir}/rock_lkhd_{ftype}.png"
                       for prefix in ['rock_lkhd', 'slip', 'sink', 'telemetry',
                                      'hyd_predicted', 'amplitude_vs_rot', 'amplitude_vs_time',
                                      'dark_degrade_corr']])
    for p in plot_files:
        assert exists(p), p.split('/')[-1]

    # Check movie files
    assert not len(glob.glob(f"{frame_path}/*.png")),\
    "Movie frame files removed"
    assert len(glob.glob(f"{plotdir}/*.mp4")) == 1, \
        "MP4 file created"
