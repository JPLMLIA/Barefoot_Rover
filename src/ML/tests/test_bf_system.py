
from os.path import exists

import pytest
from pytest_datadir.plugin import shared_datadir

from bf_config import bf_globals
from bf_system import run_system

def test_run_system_no_data_dir():
    with pytest.raises(SystemExit) as excinfo:
        run_system("/I-Don't_exist", None,
                   None, None, None,
                   None, None)
    assert "Data directory does not exist" \
        in str(excinfo.value)

def test_run_system_no_plot(shared_datadir, test_config,
                   set_global_model_params):
    exp = test_config['experiments']['rock']
    run_system(f"{str(shared_datadir)}",f"{str(shared_datadir)}/test_output",
               f'rock/{exp}', ["250", None], None,
               'rock', f"{str(shared_datadir)}/full_features_2.txt")


def test_run_system_with_plot(shared_datadir, test_config,
                           set_global_model_params):
    exp = str(test_config['experiments']['rock'])
    plot_file = f"{str(shared_datadir)}/test_plot.png"
    run_system(f"{str(shared_datadir)}", str(shared_datadir),
               f'rock/{exp}', ["250", None], plot_file,
               'rock', f"{str(shared_datadir)}/full_features_2.txt")
    assert exists(plot_file)
