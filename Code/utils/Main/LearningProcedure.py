# Summary: Runs active learning procedure by querying candidate observations from df_Candidate and adding them to the training set df_Train.
# Input: A dictionary SimulationConfigInputUpdated with the following keys and values:
#   DataFileInput: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder.
#   df_Train: The given train dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Test: The given test dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Candidate: The given candidate dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   Seed: Seed for reproducability.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially "unseen" and later added to the training set.
#   SelectorType: Selector type. Examples can be GSx, GSy, or PassiveLearning.
#   ModelType: Predictive model. Examples can be LinearRegression or RandomForestRegresso.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFarms.
#   n_estimators: The number of trees for a random forest.
#   regularization: Penalty on the number of splits in a tree.
#   rashomon_bound_adder: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output:
#   ErrorVec: A 1xM vector of errors with M being the number of observations in df_Candidate. 
#   SelectedObservationHistory: The index of the queried candidate observation at each iteration
#   TreeCount: A dictionary that contains two keys: {AllTreeCount, UniqueTreeCount} indicating
#                          the number of trees in the Rashomon set from TreeFarms and the number of unique classification patterns.

### Import packages ###
import gc
import pandas as pd

### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

### Function ###
def LearningProcedure(SimulationConfigInputUpdated):

    ### Set Up ###
    ErrorVec = []
    SelectedObservationHistory = []
    TreeCount = {"AllTreeCount": [], "UniqueTreeCount": []}
    TotalIterations = len(SimulationConfigInputUpdated["df_Candidate"])

    ### Algorithm ###
    for i in range(len(SimulationConfigInputUpdated["df_Candidate"])):

        ### Set Up ###
        gc.collect()
        print(f"Iteration: {i}/{TotalIterations}")

        ### Prediction Model ###
        ModelType = globals().get(SimulationConfigInputUpdated["ModelType"], None)
        ModelArgsFiltered = FilterArguments(ModelType, SimulationConfigInputUpdated)
        Model = ModelType(**ModelArgsFiltered)
        SimulationConfigInputUpdated['Model'] = Model

        ### Test Error ###
        TestErrorOutput = TestErrorFunction(InputModel = Model, 
                                            df_Test = SimulationConfigInputUpdated["df_Test"], 
                                            Type = SimulationConfigInputUpdated["Type"])
        
        ### Store error ###
        CurrentError = (TestErrorOutput["Error_Duplicate"] 
                       if 'TREEFARMS' in str(type(Model)) 
                       else TestErrorOutput["ErrorVal"])
        ErrorVec.append(CurrentError)

        ### Sampling Procedure ###
        SelectorType = globals().get(SimulationConfigInputUpdated["SelectorType"], None)
        SelectorArgsFiltered = FilterArguments(SelectorType, SimulationConfigInputUpdated)
        SelectorFuncOutput = SelectorType(**SelectorArgsFiltered)

        ### Get Query Observation ###
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[[QueryObservationIndex]]
        SelectedObservationHistory.append(QueryObservationIndex)
        
        ### Update Train and Candidate Sets ###
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex) 

        ### Store Number of (Unique) Trees ###
        if('TREEFARMS' in str(type(Model))):
            TreeCount["AllTreeCount"].append(SelectorFuncOutput["AllTreeCount"])          # Store number of trees
            TreeCount["UniqueTreeCount"].append(SelectorFuncOutput["UniqueTreeCount"])    # Store number of unique/duplicate trees

        ### Clean ###
        del Model
        gc.collect()

    ### Return ###          
    return {"ErrorVec": ErrorVec,
            "TreeCount": TreeCount,
            "SelectedObservationHistory": SelectedObservationHistory}