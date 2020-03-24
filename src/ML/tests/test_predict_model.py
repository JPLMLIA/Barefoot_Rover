import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir

import predict_model

version = "test_version"
date = "01234567"

def test_predict_slip(shared_datadir, random_dense_features):
    module = 'slip'
    slip_model_file = (f"{shared_datadir}/models/trained_GB_"
                       f"{module}_{version}_{date}")
    slip_features_file = (f"{shared_datadir}/models/classifier_{module}_"
                          f"{version}_{date}.h5")
    r = predict_model.predict_slip(slip_model_file, slip_features_file,
                                      random_dense_features(module))
    assert r.dtype == np.dtype('float64')


def test_predict_composition(shared_datadir, random_dense_features):
    module = 'material'
    model_file = (f"{shared_datadir}/models/trained_GB_"
                       f"{module}_{version}_{date}")
    features_file = (f"{shared_datadir}/models/classifier_{module}_"
                          f"{version}_{date}.h5")
    r = predict_model.predict_composition(model_file, features_file,
                                   random_dense_features(module))
    assert r.dtype == np.dtype('<U32')


def test_predict_rock(shared_datadir, random_dense_features):
    module = 'rock'
    model_file = (f"{shared_datadir}/models/trained_GB_"
                  f"{module}_{version}_{date}")
    features_file = (f"{shared_datadir}/models/classifier_{module}_"
                     f"{version}_{date}.h5")
    r = predict_model.predict_rock(model_file, features_file,
                                   random_dense_features(module))
    assert r.dtype == np.dtype('float64')

def test_predict_surface_patterns():
    # TODO This method does nothing at the moment
    predict_model.predict_surface_patterns(5)

def test_predict_hydration(shared_datadir,random_dense_features):
    model_file = f"{str(shared_datadir)}/models/trained_RF_hydration_v1_05172019"
    features_file = f"{str(shared_datadir)}/models/features_hydration_v1_05172019.h5"
    r = predict_model.predict_hydration(model_file, features_file,
                                        random_dense_features('hydration'))
    assert r.dtype == np.dtype('float64')
