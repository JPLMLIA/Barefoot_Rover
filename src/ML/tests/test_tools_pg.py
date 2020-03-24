import glob
import os
from builtins import SystemExit
from os import mkdir, path
from os.path import isfile
from posix import mkdir
from random import sample

import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir
from tqdm import tqdm

import compute_data
from bf_config import bf_globals
from bf_tools import tools_pg
from bf_util import h5_util, numpy_util
from compute_features import computed_dense_feature_names

N = 14
cols = 96
rows = 20
test_file = "pressure_grid/experiment/large_pg.dat"
sample_exp_name = "CRT_pebbles_terrain_sparse_vel_fast_EISfreqsweep_10-10K_grousers_full_loading_none_material_grc-01_hydra_00.0_pretreat_N_date_20180717_rep_05"


def test_convert_xiroku(shared_datadir):

    # Contains two frame with 2 two data points each
    data_flattened, data_timestamp, data_xy = tools_pg.convert_xiroku(
        str(shared_datadir / "pressure_grid/large_pg.dat"))

    assert len(data_timestamp) == len(data_xy)
    assert data_timestamp.shape == (N, 1)
    assert data_xy.shape == (N, cols, rows)
    assert data_flattened.shape == (N, rows*cols+1)

    flattened_timestamps = np.reshape(data_flattened[:, 0], (N, 1))
    assert np.array_equal(data_timestamp, flattened_timestamps)

    # Unroll flattened data
    flat_data = data_flattened[:, 1:]  # remove time stamp
    flat_split = np.split(flat_data, N)  # Seperate each frame
    unrolled_data = np.stack(  # Stack the arrays to create an array of shape (N,cols,rows)
        [
            # Create list of (cols x rows) arrays
            np.vstack(
                np.hsplit(flat_split[i], rows)).T for i in range(N)  # unroll the data into a (cols x rows) array
        ])
    assert np.array_equal(unrolled_data, data_xy)


def test_get_calibration_data_saves_new_cal_files(shared_datadir):
    tools_pg\
        .get_calibration_data(str(shared_datadir / "pressure_grid"),
                              '20180101')
    assert os.path.exists((f"{str(shared_datadir)}/pressure_grid/"
                           f"mean_offset.npy"))
    assert os.path.exists((f"{str(shared_datadir)}/pressure_grid/"
                           f"std_offset.npy"))


def test_get_calibration_data_uses_correct_files(shared_datadir):
    cal_dir = str(shared_datadir/"cal/Xiroku_no_contact")
    test_data = [('20180101', cal_dir),
                 ('20190523', f"{cal_dir}/cal_20190522"),
                 ('20190207', f"{cal_dir}/cal_20190205"),
                 ('20191101', f"{cal_dir}/cal_20191031")]
    for date, d in test_data:
        m, s = tools_pg.get_calibration_data(cal_dir, date)
        assert np.array_equal(m, np.load(f"{d}/mean_offset.npy"))
        assert np.array_equal(s, np.load(f"{d}/std_offset.npy"))


def test_extract_mean_std(shared_datadir):

    large_file = str(shared_datadir / "pressure_grid/large_pg.dat")
    small_file = f"{str(shared_datadir)}/pressure_grid/small_pg.dat"
    m, s = tools_pg.extract_mean_std([large_file])
    assert m.shape == (rows, cols)
    assert s.shape == (rows, cols)

    _, _, data = tools_pg.convert_xiroku(large_file)
    assert np.array_equal(m, np.rot90(data.mean(axis=0)))
    assert np.array_equal(s, np.rot90(data.std(axis=0)))

    # Test with multiple data files
    m, s = tools_pg.extract_mean_std([large_file, small_file])
    _, _, data_2 = tools_pg.convert_xiroku(small_file)
    stacked_data = np.vstack([data, data_2])
    assert np.array_equal(m, np.rot90(stacked_data.mean(axis=0)))
    assert np.array_equal(s, np.rot90(stacked_data.std(axis=0)))


def test_extract_mean_std_fails_with_no_files():
    with pytest.raises(SystemExit):
        tools_pg.extract_mean_std([])


def test_locate_on_pg():
    # Case 1: idx_imu > end_val - 1
    assert 2 == tools_pg.locate_on_pg(idx_imu=98)

    # Case 2: idx_imu < end_val - 1
    assert 94 == tools_pg.locate_on_pg(idx_imu=94)

    assert 90 == tools_pg.locate_on_pg(-5, end_val=95)

    assert 5 == tools_pg.locate_on_pg(205, 100)


def test_bad_impute_pixel_map():
    impute_masks, idx_mask = tools_pg.bad_impute_pixel_map()

    bad_column = 23

    # Assumes number of rows is 20
    for i in range(20):
        assert idx_mask[i, 0] == i
        assert idx_mask[i, 1] == bad_column

    # Use previous or next column instead of bad column
    # TODO: Add some check for the first column.
    for x in impute_masks:
        for y in x[:, 1]:
            assert (y == bad_column-1) or (y == bad_column+1)


def test_read_all_pg_data_fails_with_no_data_files(tmp_path):

    # Make sample mean and std numpy file
    no_contact_dir = tmp_path / "tmp_no_contact"
    no_contact_dir.mkdir()
    np.save(str(no_contact_dir) + "/mean_offset", np.arange(3))
    np.save(str(no_contact_dir) + "/std_offset", np.arange(3))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        tools_pg.read_all_pg_data(
            str(empty_dir), str(no_contact_dir), plot=False)
    assert "No PG data in" in str(excinfo.value)


def test_read_all_pg_data_fails_with_empty_data_file(tmp_path):

    data_dir = tmp_path / sample_exp_name
    data_dir.mkdir()
    empty_data_file = data_dir / "empty.dat"
    np.save(f'{str(data_dir/"wheel.npy")}', np.arange(3))
    empty_data_file.write_text("No data")

    with pytest.raises(SystemExit) as excinfo:
        tools_pg.read_all_pg_data(
            str(data_dir), "", plot=False)
    assert "No PG data recorded in" in str(excinfo.value)


def test_read_all_pg_data_fails_with_no_imu_files(tmp_path, shared_datadir):

    # Make sample mean and std numpy file
    no_contact_dir = tmp_path / "tmp_no_contact"
    no_contact_dir.mkdir()
    np.save(str(no_contact_dir) + "/mean_offset", np.arange(3))
    np.save(str(no_contact_dir) + "/std_offset", np.arange(3))

    # Make sample dat file
    data_dir = tmp_path / "tmp_data"
    data_dir.mkdir()
    empty_data_file = data_dir / "tmp.dat"
    with open(f"{str(shared_datadir)}/pressure_grid/small_pg.dat", 'r') as f:
        empty_data_file.write_text(f.read())

    with pytest.raises(SystemExit) as excinfo:
        tools_pg.read_all_pg_data(
            str(data_dir), str(no_contact_dir), plot=False)
    assert "No IMU file" in str(excinfo.value)


def test_read_all_pg_data_fails_with_incorrect_experiment_dirname_format(shared_datadir):
    # File format should be as follows:
    # [somePrefix]_[label]_terrain_[terrain]_vel_[velocit/speed]_EISfreqsweep_[EIS Frequency]
    # _grousers_[grouser_type]_loading_[loading_type]_material_[material type]_hydra_[hydration percentage]
    # _pretreat_[pretreat type]_date_[date (YYYYMMDD)]_rep_[reps]

    datadir = shared_datadir / "tmp/sample"
    datadir.mkdir(parents=True)

    datadir.touch("large.dat")
    (datadir /
     "large.dat").symlink_to(f"{str(shared_datadir)}/pressure_grid/large_pg.dat")

    # Make sample imu data file
    datadir.touch("wheel.npy")
    (datadir /
     "wheel.npy").symlink_to(f"{str(shared_datadir)}/pressure_grid/wheel.npy")

    with pytest.raises(ValueError) as excinfo:
        tools_pg.read_all_pg_data(str(datadir), str(datadir), plot=False)
    assert "sample" == str(excinfo.value)


def test_read_all_pg_data_returns_correct_data_with_no_experiment_data(shared_datadir):

    # ----------------
    # START Directory setup
    datadir = shared_datadir / "tmp" / sample_exp_name
    datadir.mkdir(parents=True)

    datadir.touch("large.dat")
    (datadir /
     "large.dat").symlink_to(f"{str(shared_datadir)}/pressure_grid/large_pg.dat")

    # Make sample imu data file
    datadir.touch("wheel.npy")
    (datadir /
     "wheel.npy").symlink_to(f"{str(shared_datadir)}/pressure_grid/wheel.npy")
    # END Directory set up

    _, pg_data_dict, time_data_dict = tools_pg.read_all_pg_data(
        str(datadir), str(datadir), plot=False)

    assert pg_data_dict['pg'].shape == (N, rows, cols)
    assert not np.any(np.isnan(pg_data_dict['imu']))

    pg_data_dict['pattern'] is not None and pg_data_dict['pattern'] != 'unknown'

    assert pg_data_dict['ft_xyz'].shape == (N, 6)

    for k in ['slip', 'sink', 'current', 'rock',
              'rock_depth', 'slip_fiducials']:
        assert pg_data_dict[k].shape == (N,)
        is_all_nan = np.all(np.isnan(pg_data_dict[k]))
        is_all_zero = np.count_nonzero(pg_data_dict[k]) == 0
        assert is_all_nan or is_all_zero

    for k in time_data_dict.keys():
        if k != 'imu':
            assert time_data_dict[k].shape == (N,)
    assert time_data_dict['pg'].shape == (N,)
    # TODO Add check for 'material' type


def test_read_all_pg_data_returns_correct_data_with_experiment_data(shared_datadir, full_pg_data):

    pg_data_dict, time_data_dict = full_pg_data['pg'], full_pg_data['time']

    assert pg_data_dict['pg'].shape == (N, rows, cols)
    assert not np.all(np.isnan(pg_data_dict['pg']))

    assert time_data_dict['pg'].shape == (N,)
    assert not np.all(np.isnan(time_data_dict['pg']))

    data_keys = ['slip', 'sink', 'current', 'ft_xyz', 'rock',
                 'rock_depth', 'slip_fiducials', 'imu']
    for k in data_keys:
        assert not np.any(np.isnan(pg_data_dict[k]))

    pattern_map = {'rock': 'rock_below', 'patterns': 'unknown',
                   'slip': 'flatlvl', 'material': 'unkown'}

    # TODO Add check for time_data_dict

    # TODO Add check for 'material' type

    # TODO Add Check for ftype


def test_read_all_pg_data_generates_plots(shared_datadir, test_config):
    for module in ['rock', 'patterns', 'material', 'slip']:
        exp = test_config['experiments'][module]
        exp_dir = f"{str(shared_datadir)}/{module}/{exp}"
        cal_dir = f"{str(shared_datadir)}/cal/Xiroku_no_contact"
        tools_pg.read_all_pg_data(exp_dir, cal_dir)
        assert os.path.exists(f"{exp_dir}/plots/")
        assert len(glob.glob(f"{exp_dir}/plots/*.png"))


def test_rot_start_stop_motor(get_experiment_data):
    for exp in ['rock', 'slip', 'material', 'patterns']:
        pg, time, data_binned, time_binned = get_experiment_data(exp)
        stat_mask = tools_pg.rot_start_stop_motor(
            data_binned['current'], data_binned['imu'])

        assert stat_mask.shape == data_binned['imu'].shape
    # TODO write more details test for this functions


def test_rot_start_stop_motor_graph(get_experiment_data):
    for exp in ['rock', 'slip', 'material', 'patterns']:
        pg, time, data_binned, time_binned = get_experiment_data(exp)
        stat_mask, fig = tools_pg.rot_start_stop_motor(
            data_binned['current'], data_binned['imu'], graphs=True)
        assert stat_mask.shape == data_binned['imu'].shape
        assert fig
    # TODO write more details test for this functions


def test_rot_start_stop_motor_returns_graph(full_pg_data):
    data_binned, time_binned = tools_pg.align_pg_to_imu(full_pg_data['pg'], full_pg_data['time'],
                                                        bf_globals.T, bf_globals.R)
    stat_mask, fig = tools_pg.rot_start_stop_motor(
        data_binned['current'], data_binned['imu'], graphs=True)
    assert fig


def test_align_pg_to_imu(full_pg_data):

    data_binned, time_binned = tools_pg.align_pg_to_imu(full_pg_data['pg'], full_pg_data['time'],
                                                        bf_globals.T, bf_globals.R)

    assert full_pg_data['pg'].keys() == data_binned.keys()

    # TODO Add test for time
    # TODO write more details test for this functions


def test_clean_slip(full_pg_data):
    assert not np.any(tools_pg.clean_slip(np.arange(4)))
    assert np.any(tools_pg.clean_slip(full_pg_data['pg']['slip']))
    # TODO write more details test for this functions


def test_unwrap_pg_to_imu_fails_with_wrong_extraction_type():
    with pytest.raises(SystemExit) as excinfo:
        tools_pg.unwrap_pg_to_imu(
            None, None, None, None, extraction="wrong_type")
    assert f"Extraction type wrong_type is not" in str(excinfo.value)


def test_unwrap_pg_to_imu(full_pg_data):
    data_binned, time_binned = tools_pg.align_pg_to_imu(full_pg_data['pg'], full_pg_data['time'],
                                                        bf_globals.T, bf_globals.R)
    rot_mask = tools_pg.rot_start_stop_motor(
        data_binned['current'], data_binned['imu'])
    time_binned = time_binned[~rot_mask]
    lag = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    pg_imu_rock, rho_imu, x_imu = tools_pg.unwrap_pg_to_imu(data_binned['pg'], data_binned['imu'], time_binned,
                                                            lag, extraction='grouser')

    assert len(pg_imu_rock) == len(lag)
    for a in pg_imu_rock:
        assert a.shape[1]
    # TODO write more details test for this function


@pytest.mark.skip(reason="Test takes too long")
def test_waveletDegrouse(full_pg_data):
    data_binned, time_binned = tools_pg.align_pg_to_imu(full_pg_data['pg'],
                                                        full_pg_data['time'], bf_globals.T, bf_globals.R)
    lag = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    pg_imu_rock, _, _ = tools_pg.unwrap_pg_to_imu(data_binned['pg'],
                                                  data_binned['imu'], time_binned, lag, extraction='grouser')

    pg_recon = tools_pg.waveletDegrouse(pg_imu_rock, wtype='db2')
    assert len(pg_imu_rock) == len(pg_recon)


def test_number_rot(shared_datadir, full_pg_data):

    data_binned, _ = tools_pg.align_pg_to_imu(
        full_pg_data['pg'], full_pg_data['time'], bf_globals.T, bf_globals.R)
    n_rot, mean_slip = tools_pg.number_rot(data_binned['imu'])
    assert mean_slip >= 0.0
    # TODO Add check for n_rot
    # TODO write more details test for this functions


def test_contact_imu(full_pg_data):
    imu_binned = full_pg_data['pg_binned']['imu']
    idx_imu = tools_pg.contact_imu(imu_binned)
    for i in list(idx_imu):
        assert i >= 0. and i <= 96.0


def test_contact_area_pg(full_pg_data):
    data_binned = full_pg_data['pg_binned']
    pg_binned = data_binned['pg']
    nt = pg_binned.shape[0]
    remove_lag = 5
    for t in range(0, nt):
        for k in tools_pg.contact_area_keys[1:]:
            pg_t = pg_binned[t, :, tools_pg.contact_area_indexes[k]].T
            output = tools_pg.contactAreaPG(pg_t, tools_pg.contact_area_indexes[k],
                                            data_binned['imu'][t], remove_lag=remove_lag)

            assert output['npix'] == len(output['contact_coords'])

            if len(output['contact_coords']) > 0:
                for cv in np.nditer(output['contact_value'], flags=['zerosize_ok']):
                    assert cv <= remove_lag
    # TODO Add check for 'contact_coords', 'noncontact_value', 'mask_ambiguous'


def test_contact_area_run_contains_correct_keys(full_pg_data):
    data_binned = full_pg_data['pg_binned']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(
        data_binned['pg'], imu_contact_bin)
    assert list(contact_data.keys()) == tools_pg.contact_area_keys
    # TODO write more details test for this functions


def test_contact_area_tsfresh_features_contains_correct_keys(shared_datadir, full_pg_data):
    data_binned = full_pg_data['pg_binned']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(
        data_binned['pg'], imu_contact_bin)

    with open(f"{str(shared_datadir)}/tsfresh_features.txt", 'r') as f:
        feature_list = [line.rstrip() for line in f.readlines()]

    nt = data_binned['pg'].shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)

    for t in tqdm(t_seq):
        features_subset = tools_pg.contact_area_tsfresh_features(
            t, data_binned, contact_data, feature_list)
        for k in features_subset.keys():
            assert k in feature_list

    # TODO write more details test for this functions


def test_sharpPG(get_experiment_data):
    for exp in ['rock', 'slip', 'material', 'patterns']:
        pg, time, data_binned, time_binned = get_experiment_data(exp)
        imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
        contact_data = tools_pg.contact_area_run(
            data_binned['pg'], imu_contact_bin)

        num_pix_limit = 15
        sharp_lim = 4
        pg_lim = 80.0
        pg_binned = data_binned['pg']

        sharp = tools_pg.sharpPG(pg_binned, contact_data,
                                 npix_lim=num_pix_limit,
                                 sharp_lim=sharp_lim,
                                 pg_lim=pg_lim)
        contact_idx = contact_data['all']['contact_coords']
        n = pg_binned.shape[0]
        assert n == sharp.shape[0]

        for i in range(n):
            if len(contact_idx[i]) == 0:
                assert not sharp[i]
            # Criteria 1: contact area < npix_limit
            elif len(contact_idx[i]) >= num_pix_limit:
                assert not sharp[i]
            else:
                # Criteria 2: the number of extreme values greater than 95th
                # percentile of the distribution of the contact pressure
                # values <= sharp_lim
                pg_t = pg_binned[i, :, :]
                pg_contact = np.hstack(
                    pg_t[contact_idx[i][:, 0], contact_idx[i][:, 1]])
                _, u = np.percentile(pg_contact, [5, 95])
                mask_sharp = pg_contact > u
                npix_sharp = mask_sharp.sum()
                if npix_sharp > sharp_lim:
                    assert not sharp[i]
                # Criteria 3: the mean of the pressure values is greater than a specified number
                elif np.mean(pg_contact[mask_sharp]) <= pg_lim:
                    assert sharp[i]
                else:
                    assert not sharp[i]


def test_read_all_sparse_features(shared_datadir):
    for module in ['rock', 'patterns', 'slip', 'material']:
        sparse_feature_file = f"{str(shared_datadir)}/h5/compute_features/{module}/{module}_sparse.h5"
        data = tools_pg.read_all_sparse_features(sparse_feature_file, module)
        assert sorted(data.keys()) == computed_dense_feature_names
        for v in data.values():
            assert type(v) == np.ndarray or \
                type(v) == int


def test_remove_nan_returns_none_with_no_data():
    assert not tools_pg.remove_nan(None, '')


def test_remove_nan(shared_datadir):
    for module in ['rock', 'patterns', 'slip', 'material']:
        sparse_feature_file = f"{str(shared_datadir)}/h5/compute_features/{module}/{module}_sparse.h5"
        data = tools_pg.read_all_sparse_features(sparse_feature_file, module)

        dense, nan_mask = tools_pg.remove_nan(data, module)
        for v in dense.keys():
            if isinstance(v, np.ndarray):
                assert all(~np.isnan(v))
                assert all(~np.isinf(v))
            else:
                assert v != 0


def test_leanPG(get_experiment_data):
    for exp in ['rock', 'slip', 'material', 'patterns']:
        pg, time, data_binned, time_binned = get_experiment_data(exp)
        imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
        contact_data = tools_pg.contact_area_run(
            data_binned['pg'], imu_contact_bin)
        lean = tools_pg.leanPG(data_binned['pg'], contact_data)
        assert lean.shape == (data_binned['pg'].shape[0],)
        for i in list(lean):
            assert i in [-1, 0, 1]


def test_selectFeatures(shared_datadir):
    for module in ['rock', 'patterns', 'slip', 'material']:
        sparse_feature_file = f"{str(shared_datadir)}/h5/compute_features/{module}/{module}_sparse.h5"
        data = tools_pg.read_all_sparse_features(sparse_feature_file, module)
        feature_selected_sparse = tools_pg.selectFeatures(data)

        assert sorted(feature_selected_sparse.keys()) == sorted(
            computed_dense_feature_names)

        assert np.array_equal(feature_selected_sparse['feature_names'],
                              data['feature_names'])

        assert feature_selected_sparse['nF'] == len(data['feature_names'])

    # TODO Try with preset=True and some names_to_exclude


def test_contactAreaFeatures(rock_data):
    pg, time_pg, data_binned, time_binned = rock_data
    check_contactAreaFeatures(
        data_binned['pg'], data_binned['imu'], no_contact=False)


def test_contactAreaFeatures_with_no_contact_data(full_pg_data):
    data_binned = full_pg_data['pg_binned']
    check_contactAreaFeatures(data_binned['pg'], data_binned['imu'])


def check_contactAreaFeatures(pg_binned, imu_bin, no_contact=True):
    # make sure features are in list
    feature_list = []
    for k in tools_pg.contact_area_keys[:-1]:
        for c in tools_pg.contact_area_feature_names:
            feature_list.append(f"{c}_{k}")

    imu_contact_bin = tools_pg.contact_imu(imu_bin)
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)
    for t in tqdm(t_seq):
        results = tools_pg.contactAreaFeatures(t, pg_binned,
                                               imu_contact_bin,
                                               contact_data,
                                               feature_list)
        keys = results.keys()
        for k in tools_pg.contact_area_keys[:-1]:
            if no_contact:
                assert not len(contact_data[k]['contact_coords'][t])
            else:
                assert len(contact_data[k]['contact_coords'][t])
            for key in tools_pg.contact_area_feature_names:
                feature_name = f"{key}_{k}"
                assert feature_name in keys
                if no_contact:
                    assert np.isnan(results[feature_name])
                else:
                    assert not np.isnan(results[feature_name])


def test_contactAreaLaggedFeatures(rock_data):
    pg, time_pg, data_binned, time_binned = rock_data

    # make sure features are in list
    feature_list = ['npix_max_diff', 'npix_min_diff']
    for k in tools_pg.contact_area_keys[:-1]:
        for c in tools_pg.contact_area_lag_featue_names:
            if c not in ['npix_max_diff', 'npix_min_diff']:
                feature_list.append(f"{c}_{k}")

    pg_binned = data_binned['pg']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)
    for t in tqdm(t_seq):
        results = tools_pg.contactAreaLaggedFeatures(t, imu_contact_bin,
                                                     contact_data,
                                                     feature_list)
        keys = results.keys()
        for i in ['npix_max_diff', 'npix_min_diff']:
            assert i in keys
            assert not np.isnan(results[i])
        for k in tools_pg.contact_area_keys[:-1]:
            for key in tools_pg.contact_area_lag_featue_names:
                feature_name = f"{key}_{k}"
                if key not in ['npix_max_diff', 'npix_min_diff']:
                    assert feature_name in keys
                    assert not np.isnan(results[feature_name])


def test_contactAreaLaggedFeatures_no_contact(full_pg_data):
    data_binned = full_pg_data['pg_binned']

    # make sure features are in list
    feature_list = ['npix_max_diff', 'npix_min_diff']
    for k in tools_pg.contact_area_keys[:-1]:
        for c in tools_pg.contact_area_lag_featue_names:
            if c not in ['npix_max_diff', 'npix_min_diff']:
                feature_list.append(f"{c}_{k}")

    pg_binned = data_binned['pg']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)
    for t in tqdm(t_seq):
        results = tools_pg.contactAreaLaggedFeatures(t, imu_contact_bin,
                                                     contact_data,
                                                     feature_list)
        keys = results.keys()
        for i in ['npix_max_diff', 'npix_min_diff']:
            assert i in keys
            assert np.isnan(results[i])


def test_sinkFeatures(rock_data):
    pg, time_pg, data_binned, time_binned = rock_data
    pg_binned = data_binned['pg']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)
    for t in tqdm(t_seq):
        results = tools_pg.sinkFeatures(t, pg_binned, imu_contact_bin,
                                        contact_data, tools_pg.sink_feature_names)
        assert sorted(results) == sorted(tools_pg.sink_feature_names)
    '''
     TODO Simple test for now though I am not sure we can test much
          more than this.  Most of the work is done in sinkagePG()
    '''


def test_IMUunwrapFeatures(rock_data):
    pg, time_pg, data_binned, time_binned = rock_data
    check_IMUunwrapFeatures(data_binned['pg'], data_binned['imu'], time_binned)
    # TODO Add more detail or more tests later if possible


def check_IMUunwrapFeatures(pg_binned, imu_bin, time_binned, small_data=False):
    pglag = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    imu_contact_bin = tools_pg.contact_imu(imu_bin)
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)

    # Make sure each feature is in the feature list
    feature_list = []
    for p in pglag:
        for f in tools_pg.imu_unwrap_feature_names:
            for k in tools_pg.contact_area_keys[1:-1]:
                feature_list.append(f"{f}{p}_{k}")

    for t in tqdm(t_seq):
        results = tools_pg.IMUunwrapFeatures(t, pg_binned, imu_contact_bin,
                                             time_binned, contact_data, pglag, feature_list)
        for k in results:
            assert k in feature_list
            if small_data:
                assert np.isnan(results[k])
            else:
                assert not np.isnan(results[k])


def test_sinkagePG(rock_data):
    pg, time_pg, data_binned, time_binned = rock_data
    pg_binned = data_binned['pg']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    smoothed, sinkage = tools_pg.sinkagePG(
        pg_binned, imu_contact_bin, contact_data)
    assert np.array_equal(smoothed, numpy_util.running_mean(sinkage, 100, 0))


def test_sinkagePG_return_theta(rock_data):
    pg, time_pg, data_binned, time_binned = rock_data
    pg_binned = data_binned['pg']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    smoothed, sinkage, theta_rs, theta_fs, theta_m = tools_pg.sinkagePG(
        pg_binned, imu_contact_bin, contact_data, return_theta=True)


def test_IMUunwrapImageGeoFeatures(rock_data, feature_list):

    pg, time_pg, data_binned, time_binned = rock_data
    pg_binned = data_binned['pg']
    imu = data_binned['imu']
    imu_contact_bin = tools_pg.contact_imu(imu)
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)
    pglag = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    for t in tqdm(t_seq):
        tools_pg.IMUunwrapImageGeoFeatures(t, pg_binned, imu_contact_bin,
                                           time_binned, imu, contact_data, pglag, feature_list)
    '''
     TODO The features here are very tricky.  It's going to take
          some effort and serious knowledge of the data to test
          this thoroughly
    '''


def test_extract_exp_metadata_raises_error():
    wrong_format = "wrong_format"
    with pytest.raises(ValueError) as excinfo:
        assert tools_pg.extract_exp_metadata(wrong_format)
    assert wrong_format == str(excinfo.value)


def test_extract_exp_metadata_returns_data(test_config):
    exp = ("CRT_regprop_terrain_rock-below_vel_fast_EISfreqsweep_10-"
           "10K_grousers_full_loading_none_material_mins30_hydra_01.0"
           "_pretreat_N_date_20191219_rep_1")
    metadata = tools_pg.extract_exp_metadata(exp)
    assert metadata.label == 'regprop'
    assert metadata.terrain == 'rock-below'
    assert metadata.loading == "none"
    assert metadata.material == "mins30"
    assert metadata.hydration == '01.0'
    assert metadata.date == '20191219'
    assert metadata.rep == '1'
    assert str(metadata) == \
        "regprop_terrain_rock-below_none_mins30_01.0_20191219_1"

    exp = ("CRT_slip050_terrain_flatlvl_vel_fast_EISfreqsweep_10-10K"
           "_grousers_full_loading_none_material_mm.2mm_hydra_00.0"
           "_pretreat_N_date_20190424_rep_04")
    metadata = tools_pg.extract_exp_metadata(exp)
    assert metadata.label == 'slip050'
    assert metadata.terrain == 'flatlvl'
    assert metadata.loading == "none"
    assert metadata.material == "mm.2mm"
    assert metadata.hydration == '00.0'
    assert metadata.date == '20190424'
    assert metadata.rep == '04'
    assert str(metadata) == \
        "slip050_terrain_flatlvl_none_mm.2mm_00.0_20190424_04"


def test_scale_pg_data(shared_datadir, test_config):
    exp = test_config['experiments']['rock']
    data_file = glob.glob(f"{shared_datadir}/rock/{exp}/*.dat")[0]
    _, _, pg = tools_pg.convert_xiroku(data_file)
    exp_date = tools_pg.extract_exp_metadata(exp).date
    m = np.load(f"{shared_datadir}/cal/Xiroku_no_contact/mean_offset.npy")
    s = np.load(f"{shared_datadir}/cal/Xiroku_no_contact/std_offset.npy")

    pg_degrad, pg_scaled =\
        tools_pg.scale_pg_data(pg, m, s, exp_date)
    num_of_rock_frames = 1102
    assert pg_degrad.shape == (rows, cols)
    assert pg_scaled.shape == (num_of_rock_frames, rows, cols)


def test_load_crossbow_data_with_no_crossbow_data(tmp_path):
    # Wheel file must be present
    data_dir = tmp_path / sample_exp_name
    data_dir.mkdir()
    N = 6
    wheel_array = np.arange(N).reshape((3, 2))
    np.save(f'{str(data_dir/"wheel.npy")}', wheel_array)

    crossbow_data, time_dict = tools_pg.load_crossbow_data(data_dir, N)

    assert np.array_equal(crossbow_data['imu'], wheel_array[:, -1])
    assert np.array_equal(time_dict['imu'], wheel_array[:, 0])

    for k, v in crossbow_data.items():
        if k != 'imu':
            if k not in ['rock', 'rock_depth']:
                assert np.all(np.isnan(v)), f"key = {k}"
            else:
                assert not np.count_nonzero(v), f"key = {k}"
    for k, v in time_dict.items():
        if k != 'imu':
            if k != 'rock':
                assert np.all(np.isnan(v)), f"key = {k}"
            else:
                assert not np.count_nonzero(v), f"key = {k}"


def test_load_crossbow_data_with_data(tmp_path):
    # Wheel file must be present
    data_dir = tmp_path / sample_exp_name
    data_dir.mkdir()
    N = 12
    for crossbow_file in ['slip', 'ati', 'cart', 'wheel']:
        arr = np.arange(N).reshape((6, 2)) if crossbow_file != 'cart'\
            else np.arange(N).reshape((3, 4))
        np.save(f'{data_dir/crossbow_file}', arr)

    crossbow_data, time_dict = tools_pg.load_crossbow_data(data_dir, N)

    for k, v in crossbow_data.items():
        if k != 'imu':
            if k not in ['rock', 'rock_depth']:
                assert not np.any(np.isnan(v)), f"key = {k}"
    for k, v in time_dict.items():
        if k != 'imu':
            if k != 'rock':
                assert not np.any(np.isnan(v)), f"key = {k}"


def test_rot_start_stop(get_experiment_data):
    for exp in ['rock', 'slip', 'material', 'patterns']:
        pg, time, data_binned, time_binned = get_experiment_data(exp)
        _,_,fig = tools_pg.rot_start_stop(data_binned['imu'],True)
