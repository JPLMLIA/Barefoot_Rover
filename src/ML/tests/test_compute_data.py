import os

import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir

import compute_data

N = 14
cols = 96
rows = 20
sample_exp_name = "CRT_pebbles_terrain_sparse_vel_fast_EISfreqsweep_10-10K_grousers_full_loading_none_material_grc-01_hydra_00.0_pretreat_N_date_20180717_rep_05"
test_datadir = "compute_data_test_data"
no_contact_path = "cal/Xiroku_no_contact"


def test_unify_experiment_single(shared_datadir, exp_setup):

    _, exp_path, _ = exp_setup
    args = {'i': 0, 'files': [str(exp_path)],
            'datadir': str(shared_datadir), 'no_contact_dir': no_contact_path}
    results = compute_data.unify_experiment_single(args)

    for v in results.values():
        assert v is not None
    for k in ["time_binned", "imu_contact_bin", "data_binned", "ftype",
              "exp_path", "eis_binned"]:
        assert k in results.keys()


def test_unify_experiment_single_single(shared_datadir, exp_setup):

    _, exp_path, _ = exp_setup
    results = compute_data.unify_experiment_single_single_thread(
        str(exp_path), f"{str(shared_datadir)}", no_contact_path)

    for v in results.values():
        assert v is not None
    for k in ["time_binned", "imu_contact_bin", "data_binned", "ftype", "exp_path", "eis_binned"]:
        assert k in results.keys()

def test_unify_experiment_data_fails_with_no_datadir(shared_datadir):
    with pytest.raises(SystemExit) as excinfo:
        compute_data.unify_experiment_data(
            f"{str(shared_datadir)}/fake_dir", "", "", "*")
    assert "Data directory does not exist." in str(excinfo.value)


def test_unify_experiment_data_fails_with_no_outfile(shared_datadir):
    with pytest.raises(SystemExit) as excinfo:
        compute_data.unify_experiment_data(
            f"{str(shared_datadir)}", "", "*", None)
    assert "Outfile file path is a required argument." in str(excinfo.value)


def test_unify_experiment_data_fails_with_no_experiment_files(shared_datadir):
    empty_dir = shared_datadir / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        compute_data.unify_experiment_data(
            f"{str(empty_dir)}", "", "*", f"{str(shared_datadir)}/outfile")
    assert "Glob resulted in no experiment files." in str(excinfo.value)


def test_unify_experiment_data_returns_data(shared_datadir):

    for exp_type in ['rock', 'material', 'slip', 'patterns']:
        results = compute_data.unify_experiment_data(f"{shared_datadir}",
                                                    [exp_type], "*",
                                                    f"{shared_datadir}/{exp_type}_data.h5",
                                                    multithreading=True)
        assert len(results.values()) != 0


        pg_data_keys = ['slip', 'sink', 'current', 'ft_xyz', 'rock',
                    'rock_depth', 'slip_fiducials', 'imu', 'material',
                    'pattern', 'pg']
        compute_data_keys = ['time', 'imu_bin', 'exp_path',
                    'amp', 'phase', 'hyd', 'freq']
        for main_key in results.keys():
            assert sorted(results[main_key].keys()) == \
                sorted(list(pg_data_keys + compute_data_keys)), \
                "All keys present in dictionary"
            for val in results[main_key].values():
                assert val is not None
