

from typing import Dict, List

import h5py
import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir
from sklearn.metrics import confusion_matrix

from bf_plot import plots_pg
from bf_tools import tools_pg
from bf_util import h5_util


class AnimatedFigureData:
    """Holds data that will be used in tools_pg.animatedFigure*() calls

    Attributes
    ----------
    nt: int
        number of time steps\n
    data_binned: np.ndarray
        Time binned pressure grid data\n
    contact_data: dict
        Results of running tools_pg.contact_area_run()\n
    sharp: np.ndarray
        Results of running tools_pg.sharpPG()\n
    lean: np.ndarray
        Results of tools_pg.leanPG()\n
    imu_contact_bin: np.ndarray
        Results of
    col_max: int
    pg_imu: np.ndarray
    x_imu: List[float]
    """
    lag = [4, 3, 2, 1, 0, -1, -2, -3, -4]

    def __init__(self, data_binned, time_binned):
        self.data_binned = data_binned
        self.time_binned = time_binned

        rot_mask = tools_pg.rot_start_stop_motor(data_binned['current'], data_binned['imu'])
        time_binned=time_binned[~rot_mask]
        for k in data_binned.keys():
            data_binned[k]=data_binned[k][~rot_mask]

        self.imu_contact_bin=tools_pg.contact_imu(data_binned['imu'])
        self.contact_data=tools_pg.contact_area_run(data_binned['pg'],
                                                      self.imu_contact_bin)

        # detect sharp points
        self.sharp=tools_pg.sharpPG(data_binned['pg'], self.contact_data)

        # detect leaning
        self.lean=tools_pg.leanPG(data_binned['pg'], self.contact_data)

        self.colmax=data_binned['pg'].max(axis=1)

        self.pg_imu, _, self.x_imu=tools_pg.unwrap_pg_to_imu(data_binned['pg'],
                                                               data_binned['imu'],
                                                               time_binned,
                                                               self.lag,
                                                               extraction='all')

        self.nt=data_binned['pg'].shape[0]

@pytest.fixture
def animatedFigure_plot_data(full_pg_data):
    data_binned=full_pg_data['pg_binned']
    time_binned=full_pg_data['time_binned']

    return AnimatedFigureData(data_binned, time_binned)


def extra_plot_data(features_dense, is_classifier=True):
    S=3
    print(type(features_dense['nF']))
    imps=np.random.rand(S, features_dense['nF'])
    prob=[np.random.rand(len(features_dense['X']),
                        len(np.unique(features_dense['y'])))
            for _ in range(S)
            ]
    prob_mean=np.array(prob).mean(axis=0)
    y_pred=np.argmax(prob_mean, axis=1) if is_classifier else prob[0]

    return features_dense, imps, y_pred, prob_mean


def test_animatedFigure(shared_datadir, animatedFigure_plot_data):
    af=animatedFigure_plot_data
    for func in [plots_pg.animatedFigure, plots_pg.animatedFigure2,
    plots_pg.animatedFigure3, plots_pg.animatedFigure4]:
        for t in range(af.nt):
            assert func(af.data_binned,
                        t,
                        af.nt,
                        af.contact_data,
                        af.sharp,
                        af.lean,
                        af.imu_contact_bin,
                        af.colmax,
                        None,
                        af.pg_imu,
                        af.x_imu,
                        af.lag,
                        str(shared_datadir)), func.__name__
    # TODO Add test with jpg_match != None


def test_plots_with_predictions(shared_datadir, animatedFigure_plot_data):
    af = animatedFigure_plot_data
    slip_array = np.random.randn(af.nt)
    rock_array = np.random.randn(af.nt)
    high_pressure_array = np.random.randn(af.nt)
    lean_array = np.random.randn(af.nt)
    composition_array = np.random.randn(2, af.nt)
    hydration_array = np.random.randn(3, 3)
    sinkage_array = np.random.randn(af.nt)
    surface_array = np.random.randn(af.nt,1)
    data_pred = {"Slip":slip_array, "High Pressure":high_pressure_array,
                 "Lean":lean_array, "Composition":composition_array,
                "Hydration":hydration_array, "Sinkage":sinkage_array,
                "Surface Pattern":surface_array, "Rock":rock_array}

    for func in [plots_pg.animatedFigure, plots_pg.animatedFigure2,
                 plots_pg.animatedFigure3, plots_pg.animatedFigure4]:
        for t in range(af.nt):
            assert func(af.data_binned,
                        t,
                        af.nt,
                        af.contact_data,
                        af.sharp,
                        af.lean,
                        af.imu_contact_bin,
                        af.colmax,
                        None,
                        af.pg_imu,
                        af.x_imu,
                        af.lag,
                        str(shared_datadir),
                        data_pred), func.__name__
    assert plots_pg.plotPredictions(animatedFigure_plot_data.data_binned,
                                    animatedFigure_plot_data.nt, data_pred)


def test_classifierPlots(shared_datadir, random_dense_features):
    # TODO Doesn't work for material data
    for module in ['rock', 'patterns', 'material']:
        features_dense, imps, y_pred, prob_mean=extra_plot_data(
            random_dense_features(module))
        # TODO Fails with 'rock' and 'material' module types. Probably need different data.
        assert plots_pg.classifierPlots(features_dense, y_pred, prob_mean,
                                        imps, module)


def test_contactAreaHistPlot(animatedFigure_plot_data):
    assert plots_pg.contactAreaHistPlot("test", animatedFigure_plot_data.data_binned,
                                        animatedFigure_plot_data.contact_data)


def test_contactAreaLowFiltPlot(animatedFigure_plot_data):
    assert plots_pg.contactAreaLowFiltPlot("test", animatedFigure_plot_data.data_binned,
                                        animatedFigure_plot_data.contact_data)

def test_contactAreaPlot(animatedFigure_plot_data):
    af=animatedFigure_plot_data
    assert plots_pg.contactAreaPlot(af.contact_data, af.sharp, af.lean, "test")


def test_contactAreaSmoothPlot(animatedFigure_plot_data):
    assert plots_pg.contactAreaSmoothPlot("test", animatedFigure_plot_data.data_binned,
                                           animatedFigure_plot_data.contact_data)


def test_IMUunwrapRowMaxPlot(full_pg_data):
    pglag=[4, 3, 2, 1, 0, -1, -2, -3, -4]
    assert plots_pg.IMUunwrapRowMaxPlot(full_pg_data['pg_binned'],
                                        pglag, "test", full_pg_data['time_binned'])
    # TODO Add test with time_binned == None


def test_plot_confusion_matrix(shared_datadir, random_dense_features):
    # TODO Supposed to not work with slip data?
    for module in ['rock', 'patterns', 'material']:
        features_dense, imps, y_pred, prob_mean=extra_plot_data(
            random_dense_features(module))
        labels=features_dense['y']
        cm=confusion_matrix(labels, y_pred.copy())

        assert plots_pg.plot_confusion_matrix(cm, list(tools_pg.terrain_dict))
        assert plots_pg.plot_confusion_matrix(cm, list(tools_pg.terrain_dict),
                                          normalize=True)

    # TODO Try with different classes

def test_plotPredictions(animatedFigure_plot_data):
    assert plots_pg.plotPredictions(animatedFigure_plot_data.data_binned,
                                    animatedFigure_plot_data.nt)

    # TODO Try with 'data_pred'


def test_regressorPlots(shared_datadir, random_dense_features):
    features_dense, imps, y_pred, prob_mean=extra_plot_data(
            random_dense_features('slip'), is_classifier=False)
    assert plots_pg.regressorPlots(features_dense, y_pred, imps, "test")

def test_unwrappedGrid(animatedFigure_plot_data):
    af=animatedFigure_plot_data
    assert plots_pg.unwrappedGrid(af.lag, af.pg_imu, af.x_imu)
    assert plots_pg.unwrappedGrid_black(af.lag, af.pg_imu, af.x_imu)
    for func in [plots_pg.unwrappedGrid, plots_pg.unwrappedGrid_black]:
        for t in range(af.nt):
            assert func(af.lag, af.pg_imu, af.x_imu, clim=(0, 120), t=t)
