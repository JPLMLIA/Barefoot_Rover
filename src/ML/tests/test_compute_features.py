import glob
import os

import h5py
import numpy as np
import pytest
from pytest_datadir.plugin import shared_datadir

import compute_data
import compute_features

sample_exp_name = "CRT_pebbles_terrain_sparse_vel_fast_EISfreqsweep_10-10K_grousers_full_loading_none_material_grc-01_hydra_00.0_pretreat_N_date_20180717_rep_05"
compute_data_out_file = "compute_data_test"


def test_make_features(shared_datadir, all_features_list):
    for module in ['rock', 'patterns', 'slip', 'material']:

        data_file = f"{str(shared_datadir)}/h5/compute_data/{module}_data.h5"
        with h5py.File(data_file) as results:
            # Set up list of argument dicts to be passed into make_features.
            #   Needed since input arguments >1
            experiments = []
            files = list(results.keys())
            assert len(files) == 1
            assert sorted(results[files[0]].keys()) == \
                   sorted(compute_data.compute_data_feature_names)
            for i, ftype in enumerate(results):
                data_binned = {k: results[ftype][k][:] for k in results[ftype].keys()}
                experiments.append({"i": i,
                                    "ftype": ftype,
                                    "files": files,
                                    "data_binned": data_binned,
                                    "low_burn": 100000,
                                    "high_burn": 100000,
                                    "exp_path": "Only used for plotting",
                                    "datadir": "Only used for plotting",
                                    "plot": False,
                                    'module': module,
                                    'features': all_features_list})

            for exp in experiments:
                x = compute_features.make_features(exp)
                assert len(x['features'])
                assert sorted(list(x.keys())) == \
                    sorted(["features", "features_hyd", "labels", "other", "time_to_compute"])
                for k,v in x.items():
                    assert v is not None


def test_make_features_makes_graphs(shared_datadir, all_features_list):
    for module in ['rock', 'patterns', 'slip', 'material']:
        data_file = f"{str(shared_datadir)}/h5/compute_data/{module}_data.h5"
        with h5py.File(data_file) as results:
            # Set up list of argument dicts to be passed into make_features.
            #   Needed since input arguments >1
            experiments = []
            files = list(results.keys())
            assert len(files) == 1
            assert sorted(results[files[0]].keys()) == \
                sorted(compute_data.compute_data_feature_names)
            plot_dir = f"test_{module}_"
            for i, ftype in enumerate(results):
                data_binned = {k: results[ftype][k][:]
                               for k in results[ftype].keys()}
                experiments.append({"i": i,
                                    "ftype": ftype,
                                    "files": files,
                                    "data_binned": data_binned,
                                    "low_burn": 250,
                                    "high_burn": 250,
                                    "exp_path": plot_dir.encode('utf-8'),
                                    "datadir": f"{str(shared_datadir)}/",
                                    "plot": True,
                                    'module': module,
                                    'features': all_features_list})

            for exp in experiments:
                compute_features.make_features(exp)
                plotdir = f"{str(shared_datadir)}/{plot_dir}plots/features"
                assert os.path.exists(plotdir)
                plot_regex = "[run_mean_contact_|histogram_contact_|imurows_|lowpass_]*.png"
                assert len(glob.glob(f"{plotdir}/{plot_regex}")) == 4


def test_generate_features_exits_with_no_data_file():
    with pytest.raises(SystemExit) as excinfo:
        compute_features.generate_features(
            "", "", "random/path", "", [], False, "", False, "")
    assert "Data file does not exist." in str(excinfo.value)


def test_generate_features(shared_datadir, all_features_list):
    multithreading = False
    featuresFile = f"{str(shared_datadir)}/full_features_2.txt"
    plot = False
    for exp_type, nt in {'rock':482, 'material':475, 'slip':356,
                        'patterns':491}.items():
        data_file = f"{str(shared_datadir)}/h5/compute_data/{exp_type}_data.h5"
        outfile_name = f"{exp_type}/out"
        module = exp_type
        with open(featuresFile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_features_list))

        input_burnin = [nt, 197970]
        sparse_dict, features_dense, nan_mask = compute_features.\
        generate_features(f'{str(shared_datadir)}/{exp_type}', str(shared_datadir),
                        data_file, outfile_name, input_burnin, plot, module,
                        multithreading, featuresFile)
        for v in sparse_dict.values():
            assert sorted(v.keys()) == \
                sorted(compute_features.computed_sparse_feature_names)
        assert sorted(features_dense.keys()) == \
           sorted(compute_features.computed_dense_feature_names)
        multithreading = not multithreading
