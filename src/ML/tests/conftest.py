"""Used for common test fixures.
This file is automatically discovered by pytest.
"""
import glob
import random
import string
from os import symlink

import h5py
import numpy as np
import pytest
import yaml
from pytest_datadir.plugin import shared_datadir

import compute_data
import compute_features
from bf_config import bf_globals
from bf_tools import tools_eis, tools_pg
from bf_util import h5_util

sample_exp_name = (f"CRT_pebbles_terrain_sparse_vel_fast_EISfreqsweep_"
                   f"10-10K_grousers_full_loading_none_material_grc-01_"
                   f"hydra_00.0_pretreat_N_date_01234567_rep_05")
sample_exp_name_2 = (f"CRT_bedrock_terrain_rock-above_vel_fast_EISfreqsweep_"
                    f"10-10K_grousers_full_loading_none_material_wed730_"
                    f"hydra_00.0_pretreat_N_date_01234567_rep_05")

@pytest.fixture
def test_config(shared_datadir):
    with open(f'{str(shared_datadir)}/test_config.yaml', 'r') as f:
        config = yaml.load(f)
        return config

@pytest.fixture
def set_global_model_params(shared_datadir):

    bf_globals.SLIP_MODEL = (f'models/'
                             f'trained_GB_slip_test_version_01234567')
    bf_globals.SLIP_FEATURE_FILE = (f'models/'
                                    f'classifier_slip_test_version_01234567.h5')
    bf_globals.ROCK_MODEL = (f'models/'
                             f'trained_GB_rock_test_version_01234567')
    bf_globals.ROCK_FEATURE_FILE = (f'models/'
                                    f'classifier_rock_test_version_01234567.h5')
    bf_globals.PATTERN_MODEL = (f'models/'
                                f'trained_GB_patterns_test_version_01234567')
    bf_globals.PATTERN_FEATURE_FILE = (f'models/'
                                       f'classifier_patterns_test_version_01234567.h5')
    bf_globals.COMPOSITION_MODEL = (f'models/'
                                    f'trained_GB_material_test_version_01234567')
    bf_globals.COMPOSITION_FEATURE_FILE = (f'models/'
                                           f'classifier_material_test_version_01234567.h5')
    bf_globals.HYDRATION_MODEL = (f'models/'
                                  f'trained_RF_hydration_v1_05172019')
    bf_globals.HYDRATION_FEATURE_FILE = (f'models/'
                                         f'features_hydration_v1_05172019.h5')

@pytest.fixture
def full_exp_dir_setup(shared_datadir, test_config):
    # Create data directory
    datadir = shared_datadir / 'full_experiment'
    datadir.mkdir()

    # Add links to experiment directories
    for module, exp_name in test_config['experiments'].items():
        if module not in ['rock', 'patterns', 'hydration', 'material']:
            exp_dir = datadir / exp_name
            exp_dir.symlink_to(f"{str(shared_datadir)}/{module}/{exp_name}")

    # Make link to no contact directory
    # cal_d = "cal/Xiroku_no_contact"
    # calibration_dir = shared_datadir / cal_d
    # calibration_dir.mkdir(parents=True)
    # (calibration_dir/"mean_offset.npy")\
    #     .symlink_to(f"{str(shared_datadir)}/{cal_d}/'mean_offset.npy'")
    # (calibration_dir/"std_offset.npy")\
    #     .symlink_to(f"{str(shared_datadir)}/{cal_d}/'std_offset.npy'")

    # for d in bf_globals.CALIBRATION_FILE_DATES:
    #     cal_dir = datadir / f"{calibration_dir}/cal_{d}"
    #     cal_dir.symlink_to(f"{str(shared_datadir)}/{calibration_dir}/cal_{d}")

    return datadir #, calibration_dir

@pytest.fixture
def all_features_list(shared_datadir):
    features = ['npix_max_diff', 'npix_min_diff']
    all_features = tools_pg.contact_area_feature_names +\
        tools_pg.contact_area_lag_featue_names
    for k in tools_pg.contact_area_keys[:-1]:
        for c in all_features:
            if c not in ['npix_max_diff', 'npix_min_diff']:
                features.append(f"{c}_{k}")
    pglag = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    for p in pglag:
        for f in tools_pg.imu_unwrap_feature_names:
            for k in tools_pg.contact_area_keys[1:-1]:
                features.append(f"{f}{p}_{k}")
    features.extend(tools_pg.sink_feature_names)
    features.extend(tools_eis.hyd_features)

    featuresFile = f"{str(shared_datadir)}/full_features.txt"
    with open(featuresFile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(features))
    return features


@pytest.fixture
def model_features(shared_datadir):
    all_features = set()
    for f in glob.glob(f'{str(shared_datadir)}/models/*.h5'):
        d = h5_util.load_h5(f)
        if 'feature_names' in d:
            for feat in d['feature_names'][:]:
                all_features.add(feat.decode('utf-8'))

    featuresFile = f"{str(shared_datadir)}/full_features_2.txt"
    with open(featuresFile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_features))
    return list(all_features)


@pytest.fixture
def get_experiment_data(test_config, shared_datadir):
    def get_data(exp_type: str):
        return get_exp_data(exp_type,test_config,shared_datadir)
    return get_data

@pytest.fixture
def rock_data(test_config, shared_datadir):
    """[summary]


    Returns
    -------
    pg_data, time_data (from read_all_pg_data)
    and data_binned, time_binned (from align_pg_to_imu)
    """
    return get_exp_data('rock', test_config, shared_datadir)

@pytest.fixture
def material_data(test_config, shared_datadir):
    return get_exp_data('material', test_config, shared_datadir)

@pytest.fixture
def slip_data(test_config, shared_datadir):
    return get_exp_data('slip', test_config, shared_datadir)

@pytest.fixture
def patterns_data(test_config, shared_datadir):
    return get_exp_data('patterns', test_config, shared_datadir)

@pytest.fixture
def all_data(rock_data,
             material_data,
             slip_data,
             patterns_data):
    return {"rock": rock_data, "material": material_data,
                "slip": slip_data, "patterns": patterns_data}


def get_exp_data(exp_type: str, config_dict, shared_datadir):
    exp = config_dict['experiments'][exp_type]
    exp_dir = f"{shared_datadir}/{exp_type}/{exp}"
    return get_pg_data(exp_dir,
                       str(shared_datadir / "cal/Xiroku_no_contact"))

def get_pg_data(exp_dir, no_contact_dir):
    ftype, pg_data_dict, time_data_dict = tools_pg.read_all_pg_data(
        str(exp_dir), no_contact_dir, plot=False)
    data_binned, time_binned = tools_pg.align_pg_to_imu(
        pg_data_dict, time_data_dict, bf_globals.T, bf_globals.R)

    return pg_data_dict, time_data_dict, data_binned, time_binned

@pytest.fixture
def exp_setup(shared_datadir):
    # Create data directory
    datadir = shared_datadir / "compute_data_test_data"
    datadir.mkdir()

    # Add experiment directory
    exp_dir = datadir / sample_exp_name
    exp_dir.mkdir()
    exp_dir.touch("large.dat")
    (exp_dir /
     "large.dat").symlink_to(f"{str(shared_datadir)}/pressure_grid/large_pg.dat")

    # Make sample numpy files for all experiment metadata types
    exp_metadata_file_names = ["ati", "slip", "wheel", "cart"]
    for m in exp_metadata_file_names:
        metadata_filename = f"{m}.npy"
        exp_dir.touch(metadata_filename)
        (exp_dir / metadata_filename).symlink_to(
            f"{str(shared_datadir)}/pressure_grid/{metadata_filename}")

    # Make no_contact directory.  Use large_pg.dat for now
    no_contact_dir = shared_datadir / "cal/no_contact"
    no_contact_dir.mkdir(parents=True)
    no_contact_dir.touch("large_pg.dat")
    (no_contact_dir /
     "large_pg.dat").symlink_to(f"{str(shared_datadir)}/pressure_grid/large_pg.dat")

    # Make eis dir
    eis_dir = exp_dir / "eis"
    eis_dir.mkdir()
    for e in glob.glob(f"{str(shared_datadir)}/eis/*.idf"):
        eis_file = e.split("/")[-1]
        eis_dir.touch(eis_file)
        (eis_dir / eis_file).symlink_to(e)

    return datadir, exp_dir, no_contact_dir


@pytest.fixture
def full_pg_data(exp_setup):

    datadir, exp_dir, no_contact_dir = exp_setup

    _, pg_data_dict, time_data_dict = tools_pg.read_all_pg_data(
        str(exp_dir), str(no_contact_dir), plot=False)
    data_binned, time_binned = tools_pg.align_pg_to_imu(
        pg_data_dict, time_data_dict, bf_globals.T, bf_globals.R)

    return {"pg": pg_data_dict, "time": time_data_dict, "pg_binned": data_binned, "time_binned": time_binned}

@pytest.fixture
def feature_list(shared_datadir):
    features = []
    # NOTE This file may need to change in the future
    with open(f"{str(shared_datadir / 'full_features_2.txt')}", 'r') as f:
        features = list(f.readlines())
    return features

@pytest.fixture
def random_dense_features():
    def feature_data(module):
        features = {}
        features['time_to_compute'] = 1.0

        # don't use all data to speed up training
        max_data_points = 100  # min(10, len(np.unique(features['files'])))
        files = []
        for _ in range(max_data_points):
            files.append(''.join(
                [random.choice(string.ascii_letters + string.digits)
                    for n in range(32)]))
        features['files'] = np.array(files)

        features['X'] = 50*np.random\
            .random_sample((max_data_points, max_data_points))
        features['X_hyd'] = 100*np.random.random_sample((max_data_points,max_data_points))

        hyd_names = ['rot_angle',
                     'amplitude_947.86', 'amplitude_10.0',
                     'amplitude_10550.1', 'phase_10550.1',
                     'amplitude_111.304', 'phase_111.304',
                     'phase_947.86', 'phase_10.0']
        features['feature_names_hyd'] = np.array(hyd_names)
        feature_names = ['area_ratio_all', 'row_wheel_ratio_grouser',
                        'contact_std_nongrouser', 'maxcol_imu_diff_grouser',
                        'contact_min_grouser', 'euler_number1_lag-3_mean_grouser',
                        'sink_mean', 'sink_std', 'sink_theta_m'
                         ]
        features['feature_names'] = np.array((feature_names+hyd_names)*10)

        if module == 'rock':
            features['y'] = 1.0*np.random.randint(2, size=max_data_points)
        elif module == 'material':
            features['y'] = np.random.randint(
                len(tools_pg.composition_dict), size=max_data_points)
        elif module == 'patterns':
            features['y'] = np.random.randint(
                len(tools_pg.terrain_dict), size=max_data_points)
        else:  # slip
            features['y'] = 0.1*np.random.random_sample(max_data_points)

        features['material'] = np.array(random
                    .choices(list(tools_pg.composition_dict.keys()),
                             k=max_data_points))

        features['hydration'] = 100*np.random.random_sample(max_data_points)

        features['nF'] = max_data_points

        features['rock_depth'] = np.random.random_sample(max_data_points)

        features['imu_contact_bin'] = 1.0*np.random\
                                   .randint(low=1,high=97,
                                         size=max_data_points)

        features['patterns'] = np.array(random
                        .choices(list(tools_pg.terrain_dict.keys()),
                                k=max_data_points))

        features['rock'] = 1.0*np.random\
            .randint(low=1, high=2,
                     size=max_data_points)

        features['sink'] = np.random.random_sample(max_data_points)

        features['slip'] = np.random.random_sample(max_data_points)

        assert features['nF'] == features['X'].shape[1]
        assert features['X'].shape[0] == features['y'].shape[0]

        return features
    return feature_data

    @pytest.fixture
    def set_up_datadir(shared_datadir):
        # Make sample datadir
        test_dir = shared_datadir / 'test_data'
        test_dir.mkdir()
        # for exp in [sample_exp_name, sample_exp_name_2]:
