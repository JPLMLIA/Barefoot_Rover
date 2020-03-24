#!/bin/sh
LOG2_CORES=3
PROGRAM=random_forest_classifier

INPUT_FOLDER="."
INPUT_NAME="iris.csv"; SELECTION=4; NUM_TREES=10; RATIO=0.9
#INPUT_NAME="iris.csv"; SELECTION=4; NUM_TREES=10; RATIO=0.5
#INPUT_NAME="wine_data.csv"; SELECTION=4; NUM_TREES=20; RATIO=0.9
#INPUT_NAME="breast_cancer.csv"; SELECTION=6; NUM_TREES=100; RATIO=0.9
#INPUT_NAME="breast_cancer.csv"; SELECTION=6; NUM_TREES=10; RATIO=0.9

echo "Running simulation for $INPUT_NAME"

OUTPUT_FOLDER="results/$INPUT_NAME/$LOG2_CORES"

mkdir -p $OUTPUT_FOLDER

#emusim.x --ignore_starttiming --gcs_per_nodelet 1 --core_clk_mhz 160 --ddr_speed 3 -n $LOG2_CORES -o $OUTPUT_FOLDER/toto $PROGRAM.mwx $INPUT_FOLDER/$INPUT_NAME $NUM_TREES $SELECTION $RATIO
emusim.x --gcs_per_nodelet 1 --core_clk_mhz 160 --ddr_speed 3 -n $LOG2_CORES -o $OUTPUT_FOLDER/$PROGRAM $PROGRAM.mwx $INPUT_FOLDER/$INPUT_NAME $NUM_TREES $SELECTION $RATIO

echo '\007'
