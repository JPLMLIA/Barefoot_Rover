import yaml
import os
from pprint import pprint as pp


def translate_to_yaml(features_file):
    with open(features_file, 'r') as f:
        features = f.read().splitlines()

    _output_dictionary = {}
    for index, value in enumerate(features):
        _output_dictionary.setdefault('features', {})
        _output_dictionary['features'].setdefault(value, 1)

    features_dir = os.path.dirname(features_file)
    features_fname = os.path.basename(features_file).split('.')[0]
    with open(features_dir + '/' + features_fname + '.yaml', 'w') as f:
        print("Dumping to yaml file in " + features_dir)
        yaml.dump(_output_dictionary, f)

    return features_dir + '/' + features_fname + '.yaml'


def parseYamlFile(features_file):
    features = []
    with open(features_file, 'r') as f:
        features_dict = yaml.safe_load(f)

    features_dict = features_dict.get('features')
    for key, value in features_dict.items():
        if value == 1:
            features.append(key)

    return features


def categorizeFeatures():
    _output_dictionary = {}
    dictionary = {}
    featureNames = ['contactAreaFeatures', 'contactAreaLaggedFeatures', 'IMUunwrapFeatures',
                    'IMUunwrapImageGeoFeatures', 'sinkFeatures', 'collectHydFeatures']

    contactAreaFeatures = ['area_ratio', 'maxcol_imu_diff', 'row_wheel_ratio',
                           'contact_mean', 'contact_max',
                           'contact_min', 'contact_std', 'contact_skew',
                           'contact_kurtosis']

    contactAreaLaggedFeatures = ['peak_max_std', 'peak_min_std', 'smness',
                                 'peak_max_std_cwt', 'wt_jump_max', 'wt_jump_mean', 'wt_min', 'wt_max', 'len_max_flat',
                                 'n_flat']

    noKeyContactAreaLaggedFeatures = ['npix_max_diff', 'npix_min_diff']

    IMUunwrapFeatures = ['unwrap_mean', 'unwrap_std']

    IMUunwrapImageGeoFeatures = ['area', 'max_intensity', 'solidity', 'perimeter',
                                 'convex_area', 'eccentricity', 'orientation', 'min_intensity',
                                 'major_axis_length', 'minor_axis_length', 'extent', 'filled_area',
                                 'equivalent_diameter', 'mean_intensity', 'euler_number']

    sinkFeatures = ['sink_mean', 'sink_std', 'sink_slope', 'sink_diff_angle_mean', 'sink_theta_m',
                    'sink_diff_angle_std']

    collectHydFeatures = ['amplitude', 'phase']

    collectHydFeaturesRot = ['rot_angle']

    IMUunwrapImageGeoFeaturesSubnames = ['mean']

    keys = ['all', 'grouser', 'nongrouser', 'ambiguous']

    pglag = [4, 3, 2, 1, 0, -1, -2, -3, -4]

    shapes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    freqs_new = [10550.1, 947.86, 111.304, 10.0]

    for featureName in featureNames:
        dictionary.setdefault(featureName, {})

    with open("full_feature_list.yaml", 'w') as file:
        for feature in contactAreaFeatures:
            for key in keys[:-1]:
                featureString = feature + "_" + key
                dictionary['contactAreaFeatures'].setdefault('keys', []).append(featureString)

        for feature in contactAreaLaggedFeatures:
            for key in keys[:-1]:
                featureString = feature + "_" + key
                dictionary['contactAreaLaggedFeatures'].setdefault('keys', []).append(featureString)

        for feature in noKeyContactAreaLaggedFeatures:
            dictionary['contactAreaLaggedFeatures'].setdefault('keys', []).append(feature)

        for feature in IMUunwrapFeatures:
            for key in keys[1:-1]:
                for lag in pglag:
                    featureString = feature + str(lag) + "_" + key
                    dictionary['IMUunwrapFeatures'].setdefault('keys', []).append(featureString)

        for feature, shape in zip(IMUunwrapImageGeoFeatures, shapes):
            for key in keys[1:-1]:
                for lag in pglag:
                    for subname in IMUunwrapImageGeoFeaturesSubnames:
                        featureString = feature + str(shape) + "_lag" + str(lag) + "_" + subname + '_' + key
                        dictionary['IMUunwrapImageGeoFeatures'].setdefault('keys', []).append(featureString)

        for feature in sinkFeatures:
            dictionary['sinkFeatures'].setdefault('keys', []).append(feature)

        for feature in collectHydFeatures:
            for freq in freqs_new:
                featureString = feature + "_" + str(freq)
                dictionary['collectHydFeatures'].setdefault('keys', []).append(featureString)

        for feature in collectHydFeaturesRot:
            dictionary['collectHydFeatures'].setdefault('keys', []).append(feature)

        # Necessary for TOGA 3 level dictionary requirement
        _output_dictionary.setdefault('param1', {})
        _output_dictionary['param1'] = dictionary

        yaml.dump(_output_dictionary, file)


if __name__ == "__main__":

    features_file = './full_features.txt'
    features_file = './features_file.yaml'

    if os.path.exists(features_file):
        print("Found file")
    else:
        print("No file found")
        exit(1)

    # translateFeaturesToYaml(features_file)
    parseYamlFile(features_file)
