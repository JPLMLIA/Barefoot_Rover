import argparse
import timeit
import csv
import os
from translateToYaml import parseYamlFile
import logging
from pprint import pprint as pp
from statistics import mean


def make_log_dir(outdir: str):
    log_dir = os.path.join(outdir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


def saveTogaOutput(accuracy, time_to_compute, num_features, output_folder):
    average_time_to_compute = mean(time_to_compute)
    full_file_path = os.path.join(output_folder, 'toga_output.csv')

    with open(full_file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["accuracy", "average_time_to_compute", "num_features"])
        writer.writerow([accuracy, average_time_to_compute, num_features])


def main():
    # python toga_wrapper.py --datadir '/home/ppascual/Docments/Dataset/' --features 'full_features.txt' -fl './ap_experiments/full_feature_list.yaml' --datafile 'composition_UnifiedData.h5' --outfolder './test_output/' --burnin 50 'None' --module 'material' --model_type 'classifier' --version 'v1' --date '2019-11-12'
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', default="/Volumes/MLIA_active_data/data_barefoot/",
                        help="Path to data repository. Defaults to /Volumes/MLIA_active_data/data_barefoot/")

    parser.add_argument('--features', default="full_features.txt",
                        help="List of features to compute.  One per line.")

    parser.add_argument('-fl', '--full_feature_list', help="YAML file of the clustered list of features")

    parser.add_argument('--datafile', help="Full path to unified data H5 created using compute_data.py")

    parser.add_argument('--outfolder', default=os.getcwd(),
                        help="Full path, and name, of the resulting sparse feature data file")

    parser.add_argument('--outfile_name', default='barefoot_out',
                        help="Full path, and name, of the resulting sparse feature data file")  # Not needed

    parser.add_argument('--burnin', nargs='+',
                        help="Two values indicating low-end burn in and high-end burn in.")

    parser.add_argument('--module', default="rock",
                        help="Barefoot classifier type.  Ex: rock, patterns, material, hydrations")

    parser.add_argument('--model_type', help="classifier or regressor")

    parser.add_argument('--version', help="Version string.  Example: V1")

    parser.add_argument('--date', help="Date string. Example: 01242019 for January 24th, 2019")

    parser.add_argument('--data_save_path', default=None,
                        help="Optional file path to save de-NaN'd data which was used in model creation.")

    parser.add_argument('--save', default=False,
                        help="Flag for saving h5 feature files. For running TOGA wrapper, default = False")

    args = parser.parse_args()

    make_log_dir(args.outfolder)

    import compute_features
    from bf_logging import bf_log
    from train_model import train_model
    bf_log.setup_logger()
    logging.getLogger(__name__)

    togaRunTime = 0
    togaStartTime = timeit.default_timer()

    full_file_path = os.path.join(args.outfolder, 'features.txt')
    with open(full_file_path, 'w') as translated_features_file:
        _features = parseYamlFile(args.features)
        for _ in _features:
            translated_features_file.writelines(_ + "\n")

    # Get dense feature dictionary and the number of features
    _, dense, _ = compute_features.generate_features(args.datadir,  # datadir
                                                     args.outfolder,  # classdir
                                                     args.datafile,  # datafile
                                                     args.outfile_name,  # outfile_name
                                                     args.burnin,  # input burnin
                                                     False,  # plot = False
                                                     args.module,  # module
                                                     False,  # multithreading = False
                                                     full_file_path)  #

    # Train model with dense features
    # feature_file, classdir, model_type, module, version, date, data_save_path
    accuracy, time_to_compute = train_model(dense,  # feature_file
                                            args.outfolder,  # classdir
                                            args.model_type,  # model_type
                                            args.module,  # module
                                            args.version,  # version
                                            args.date,  # date
                                            args.data_save_path)  # data_save_path

    saveTogaOutput(accuracy, time_to_compute, len(dense["feature_names"]), args.outfolder)

    togaRunTime += timeit.default_timer() - togaStartTime
    logging.info("Iteration run time: {time:.1f}".format(time=togaRunTime))

    print("=======Done=======")


if __name__ == "__main__":
    main()
