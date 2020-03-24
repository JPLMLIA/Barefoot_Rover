import glob
import random

import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir

from bf_plot import plot_EIS
from bf_tools import tools_eis
from bf_util import numpy_util, trig_util


def test_set_colors():
    hyd = [0.0,1.2, 3.4, 4.5, 6.7, 8.9]
    result = plot_EIS.set_colors(hyd)
    assert len(result) == len(hyd)
    assert sorted(list(result.keys())) == sorted(list(map(lambda k: str(k),hyd)))

@pytest.fixture
def eis_data_all(shared_datadir, full_pg_data, exp_setup):
    data_binned = full_pg_data['pg_binned']
    time_binned = full_pg_data['time_binned']
    _, exp_dir, _ = exp_setup
    eis_files = glob.glob(f"{str(exp_dir)}/eis/*.idf")
    e, _ = tools_eis.read_eis(eis_files,time_binned,data_binned['imu'])
    e = trig_util.add_polar(e)
    e = numpy_util.numpyify(e)
    return e

def test_plotBodeEIS_byRot(eis_data_all):
    assert plot_EIS.plotBodeEIS_byRot(eis_data_all)

def test_plotFreqEIS_byTime(eis_data_all):
    assert plot_EIS.plotFreqEIS_byTime(eis_data_all)

def test_bodePlot(eis_data_all):
    assert plot_EIS.bodePlot(eis_data_all,
                              np.unique(eis_data_all['mat']),
                                  np.unique(eis_data_all['hyd']),
                                  'hydration')


def test_bodePlot_add_mean(eis_data_all):
    assert plot_EIS.bodePlot(eis_data_all,
                             np.unique(eis_data_all['mat']),
                             np.unique(eis_data_all['hyd']),
                             'hydration',  add_mean=True)
    assert plot_EIS.bodePlot(eis_data_all,
                             np.unique(eis_data_all['mat']),
                             np.unique(eis_data_all['hyd']),
                             'composition',  add_mean=True)

def test_regr_plot(eis_data_all):
    FREQ_LIMIT_amp = 1
    FREQ_LIMIT_phase = 1
    mask_sane_phase = eis_data_all['freq'][0, :] >= FREQ_LIMIT_phase
    mask_sane_amp = eis_data_all['freq'][0, :] >= FREQ_LIMIT_amp
    mask_rot = np.in1d(eis_data_all['rot'], [270])
    mask_hyd = np.in1d(eis_data_all['hyd'], [])
    mask_bad = mask_rot | mask_hyd
    X = np.column_stack([eis_data_all['amp'][~mask_bad][:, mask_sane_amp],
                        eis_data_all['phase'][~mask_bad, :][:, mask_sane_phase],
                        eis_data_all['rot'][~mask_bad]])
    y = eis_data_all['hyd'][~mask_bad]

    yhat = np.random.rand(X.shape[0], 20)
    hyd_levels = np.unique(eis_data_all['hyd'])

    assert plot_EIS.regr_plot(y, yhat, hyd_levels,
                              eis_data_all['mat'][~mask_bad],
                              np.unique(eis_data_all['mat']))

def test_plot_confusion_matrix(eis_data_all):
    FREQ_LIMIT_amp = 1
    FREQ_LIMIT_phase = 1
    mask_sane_phase = eis_data_all['freq'][0, :] >= FREQ_LIMIT_phase
    mask_sane_amp = eis_data_all['freq'][0, :] >= FREQ_LIMIT_amp
    mask_rot = np.in1d(eis_data_all['rot'], [270])
    mask_hyd = np.in1d(eis_data_all['hyd'], [])
    mask_bad = mask_rot | mask_hyd
    X = np.column_stack([eis_data_all['amp'][~mask_bad][:, mask_sane_amp],
                         eis_data_all['phase'][~mask_bad,
                                               :][:, mask_sane_phase],
                         eis_data_all['rot'][~mask_bad]])
    y = eis_data_all['hyd'][~mask_bad]
    hyd_levels = np.unique(y)

    cm = np.random.randint(2,size=(len(y),len(y)))

    assert plot_EIS.plot_confusion_matrix(cm, hyd_levels, normalize=True)
    assert plot_EIS.plot_confusion_matrix(cm, hyd_levels)

    # TODO Fails with normalize=False due to fmt.

def test_categorical_cmap():
    cols = plot_EIS.categorical_cmap(len(plot_EIS.dep_col_dict),1)
    assert np.all(cols)
    # TODO Try with other values for cmap argument
