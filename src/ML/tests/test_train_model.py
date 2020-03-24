import numpy as np
import os
import pytest
import random
import string
from bf_util import h5_util
from pytest_datadir.plugin import shared_datadir, datadir
from train_model import train_model
from bf_tools.tools_pg import composition_dict, terrain_dict

version = "test_version"
date = "01234567"

def test_train_model_fails_with_wrong_model_type():
    with pytest.raises(SystemExit) as execinfo:
        train_model('','','wrong model type','','','')
    assert 'model_type must be classifier or regressor' \
        in str(execinfo.value)


def test_train_model_fails_with_too_few_data():
    with pytest.raises(SystemExit) as execinfo:
        f = {'X':np.array([[]])}
        train_model(f,'','regressor','','','')
    assert "Too few samples" in str(execinfo.value)


def test_train_model_saves_all_files(shared_datadir, random_dense_features):
    classdir = shared_datadir / "train_model_test"
    classdir.mkdir()
    for module in ['rock','material','patterns','slip']:
        features = random_dense_features(module)
        # classdir = setup_model_dir
        model_type = 'regressor' if module == 'slip' else 'classifier'

        train_model(features, str(classdir), model_type, module, version, date)

        model_file = f"{str(classdir)}/{'_'.join(['trained_GB', module, version, date])}"
        assert os.path.exists(model_file)

        classifier_file=f"{str(classdir)}/{'_'.join(['classifier', module, version, date])}.h5"
        assert os.path.exists(classifier_file)

        plotdir = f"{str(classdir)}/plots"
        assert os.path.exists(plotdir)
        assert os.path.exists(f'{plotdir}/confusion_matrix.png')
        assert os.path.exists(f'{plotdir}/hist_probs.png')
        assert os.path.exists(f'{plotdir}/importance.png')
        if module != 'rock':
            assert os.path.exists(f'{plotdir}/pr_curves.png')
