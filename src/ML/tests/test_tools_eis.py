import glob

import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir
from tqdm import tqdm

from bf_tools import tools_eis, tools_pg


def test_parse_idf(datadir):

    # morning.idf
    results = tools_eis.parse_idf(str(datadir/"morning.idf"))
    assert all(results.values())
    keys = results.keys()
    assert "exp_time" in keys
    assert results['exp_time'] == 1534415759.0
    assert "Peaks" in keys
    assert results["Peaks"] == "false"
    assert len(results['times']) == 4
    assert results['times'] == [1.18199985008687,
                                1.78699996322393, 2.39400041755289, 4.82999982777983]
    assert len(results['real_primary']) == 4
    assert results['real_primary'] == [60508.8, -4271.65, 1581.07, -818526.0]
    assert len(results['imaginary_primary']) == 4
    assert results['imaginary_primary'] == [-78589.7, -
                                            94272.3, -2742.52, -5986690.0]
    assert len(results['frequency_primary']) == 4
    assert results['frequency_primary'] == [10000, 1000, 100, 10]

    # afternoon.idf
    results = tools_eis.parse_idf(str(datadir/"afternoon.idf"))
    assert all(results.values())
    keys = results.keys()
    assert "exp_time" in keys
    assert results['exp_time'] == 1529343736.0
    assert "Peaks" in keys
    assert results["Peaks"] == "false"
    assert len(results['times']) == 4
    assert results['times'] == [1.20399983134121,
                                2.01600005384535, 2.62400005012751, 4.85800020396709]
    assert len(results['real_primary']) == 4
    assert results['real_primary'] == [-18244.8, 14962.6, 2086.99, 1870620.0]
    assert len(results['imaginary_primary']) == 4
    assert results['imaginary_primary'] == [-174373.0,
                                            85141.2, 7752.81, -3233540.0]
    assert len(results['frequency_primary']) == 4
    assert results['frequency_primary'] == [10000, 1000, 100, 10]

    # noon.idf
    results = tools_eis.parse_idf(str(datadir/"noon_hour.idf"))
    # noon_hour.idf has no times
    # Only 'times' can be empty
    for k, v in results.items():
        if not v:
            assert k is 'times'
            assert results['timesteps'] == '0'
    keys = results.keys()
    assert "exp_time" in keys
    assert results['exp_time'] == 1531137734.0
    assert "Peaks" in keys
    assert results["Peaks"] == "false"


def test_read_eis_returns_no_data_with_no_time_data(datadir):
    eis_data, _ = tools_eis.read_eis([str(datadir/"no_time.idf")], [])
    assert len(eis_data.values()) != 0
    for v in eis_data.values():
        assert len(v) == 1


def test_read_eis_returns_data(rock_data, shared_datadir, test_config):
    pg, time_pg, data_binned, time_binned = rock_data
    rock_exp = test_config['experiments']['rock']

    # Prep data as in compute_data
    rot_mask = tools_pg.rot_start_stop_motor(
        data_binned['current'], data_binned['imu'])
    time_binned = time_binned[~rot_mask]
    for k in data_binned.keys():
        data_binned[k] = data_binned[k][~rot_mask]
    eis_files = glob.glob(f"{str(shared_datadir)}/rock/{rock_exp}/EIS/*.idf")
    assert len(eis_files) == 11
    eis_data, eis_binned = tools_eis.read_eis(eis_files, time_binned,
                                              data_binned['imu'])
    keys = ['RE', 'IM', 'freq', 'hyd', 'mat', 'rot', 'amp', 'phase']
    assert list(eis_data.keys()) == list(eis_binned.keys()) == keys
    for v in eis_data.values():
        assert type(v) == np.ndarray
    for k, v in eis_binned.items():
        assert type(v) == np.ndarray


def test_collect_hyd_features(rock_data, all_features_list, test_config):
    pg, time_pg, data_binned, time_binned = rock_data
    rock_exp = test_config['experiments']['rock']

    # Prep data as in compute_data
    rot_mask = tools_pg.rot_start_stop_motor(
        data_binned['current'], data_binned['imu'])
    time_binned = time_binned[~rot_mask]
    for k in data_binned.keys():
        data_binned[k] = data_binned[k][~rot_mask]
    eis_data, eis_binned = tools_eis.\
        read_eis(glob.glob(f"{str(shared_datadir)}/rock/{rock_exp}/EIS/*.idf"),
                 time_binned, data_binned['imu'])

    # add hyd features to ensure they are in the result
    pg_binned = data_binned['pg']
    imu_contact_bin = tools_pg.contact_imu(data_binned['imu'])
    contact_data = tools_pg.contact_area_run(pg_binned, imu_contact_bin)
    nt = pg_binned.shape[0]
    lower_t = nt-1
    t_seq = np.arange(lower_t, nt, 1)
    freq_binned = eis_binned['freq']
    print(freq_binned)
    for t in tqdm(t_seq):
        results = tools_eis.collect_hyd_features(
            t, freq_binned, eis_binned['amp'], eis_binned['phase'],
            data_binned['imu'], contact_data, all_features_list)
        assert list(sorted(results.keys())) == sorted(tools_eis.hyd_features)
        for k, v in results.items():
            if "amplitude" in k or "phase" in k:
                assert np.isnan(v) and \
                    np.isnan(freq_binned).sum() == len(freq_binned)
            # TODO Need to force this not to be the case or try different data
