### Summary: The following script creates an sbatch file to run ExtractError to extract the simulation results for each model method.

### Import packages ###
import os
import itertools
import numpy as np
import pandas as pd
import argparse


### Directory ###
cwd = os.getcwd()
ParentDirectory = os.path.abspath(os.path.join(cwd, "../../.."))

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--DataType", type=str, default="-1", help="Simulation case number.")
args = parser.parse_args()

### Rashomon Inputs ###
DataType = ["COMPAS"]
SelectorType = ["PassiveLearning", "RashomonQBC"]
ModelType = ["TreeFarms"]
RashomonModelsNumber = ["1", "10", "100"]

### Combinations ###
Combinations = list(itertools.product(DataType, SelectorType, ModelType, RashomonModelsNumber))

### Filter Combinations ###
Combinations = [
    (DataType, SelectorType, ModelType, RashomonModelsNumber)
    for DataType, SelectorType, ModelType, RashomonModelsNumber in Combinations
    if not (SelectorType == "PassiveLearning" and RashomonModelsNumber != "1")
]

### Data Frame ###
ParameterVector = pd.DataFrame(Combinations, columns = ["DataType", "SelectorType", "ModelType", "RashomonModelsNumber"])

### Job Name ###
ParameterVector["JobName"] = ("ExtractResults_" + 
                              "Data" + ParameterVector["DataType"].astype(str) +
                              "_ST" + ParameterVector["SelectorType"].astype(str) +
                              "_MT" + ParameterVector["ModelType"].astype(str) + 
                              "_RMN" + ParameterVector["RashomonModelsNumber"].astype(str)
                              )

### Mutate Parameter Vector ###
ParameterVector["SelectorType"] = "'" + ParameterVector["SelectorType"] + "'"
ParameterVector["ModelType"] = ParameterVector["ModelType"] + "RashomonNum" + ParameterVector["RashomonModelsNumber"]
ParameterVector.drop("RashomonModelsNumber", inplace= True, axis = 1)

print(ParameterVector)

# Loop through each row in the DataFrame
for i, row in ParameterVector.iterrows():
    # Extract parameters for the current row
    JobName = row["JobName"]
    DataType = row["DataType"]
    SelectorType = row["SelectorType"]
    ModelType = row["ModelType"]

    # Define the path for the .sbatch file
    sbatch_file_path = os.path.join(ParentDirectory, "Code", "Cluster", DataType, "ExtractResults", f"{JobName}_extract.sbatch")
    
    # Create the .sbatch file content
    sbatch_content = [
        "#!/bin/bash",
        f"#SBATCH --job-name={JobName}_extract",
        "#SBATCH --partition=short",
        "#SBATCH --ntasks=1",
        "#SBATCH --time=11:59:00",
        "#SBATCH --mem-per-cpu=30000",
        f"#SBATCH -o ClusterMessages/out/extract_{JobName}_%j.out",
        f"#SBATCH -e ClusterMessages/error/extract_{JobName}_%j.err",
        "#SBATCH --mail-type=FAIL",
        "#SBATCH --mail-user=simondn@uw.edu",
        "",
        "cd ~/RashomonActiveLearning",
        "module load Python",
        "python Code/utils/Auxiliary/ExtractError.py \\",
        f"    --DataType {DataType} \\",
        f"    --SelectorType {SelectorType} \\",
        f"    --ModelType {ModelType} \\"
    ]
    
    # Ensure directory exists for SBATCH file
    os.makedirs(os.path.dirname(sbatch_file_path), exist_ok=True)
    
    # Write content to .sbatch file
    with open(sbatch_file_path, "w") as sbatch_file:
        sbatch_file.write("\n".join(sbatch_content))

print("Extract Sbatch files for extraction generated successfully.")

