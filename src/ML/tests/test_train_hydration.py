
from os.path import exists

import pytest
from pytest_datadir.plugin import shared_datadir, datadir

from train_hydration import train_hydration_model

version = 'test_version'
date = '01234567'

# @pytest.mark.skip(reason='Not sure if this is going to be used')
def test_train_hydration_model(shared_datadir, datadir):
    train_hydration_model(str(datadir), '*', [''], version, date)
    classdir = f"{str(datadir)}/models/hydration/{version}/"

    data_file = f'data_hydration_{version}_{date}.h5'
    assert exists(f'{classdir}/{data_file}')

    features_file = f'features_hydration_{version}_{date}.h5'
    assert exists(f'{classdir}/{features_file}')

    model_file = '_'.join(['trained_RF', 'hydration', version, date])
    assert exists(f'{classdir}/{model_file}')

    for plot_file in [f'plots/var_imp_{version}.png',
                      f'plots/cm_{version}.png',
                      f'plots/boxplots_lkhd_{version}.png']:
        assert exists(f'{classdir}/{plot_file}')
