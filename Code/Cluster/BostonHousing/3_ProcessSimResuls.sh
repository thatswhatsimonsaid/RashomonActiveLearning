#!/bin/bash

### Current Directory Name ###
CURRENT_DIR=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DIR"

### Extract Random Forests Results ###
cd ~/RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "RandomForestClassification" \
    --Categories "MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.025.pkl"

### Extract Duplicate TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "DPL0.01.pkl"

### Extract Unique TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "UNQ0.01.pkl"