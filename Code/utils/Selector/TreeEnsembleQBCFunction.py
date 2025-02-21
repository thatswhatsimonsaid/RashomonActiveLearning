# Summary: Query-by-committee function for either random forest or Rashomon's TreeFarms that 
#          recommends an observation from the candidate set to be queried.
# Input:
#   Model: The predictive model.
#   df_Candidate: The candidate set.
#   df_Train: The training set.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFarms.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

# NOTE: Incorporate covariate GSx in selection criteria? Good for tie breakers.

### Libraries ###
import gc
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

### Function ###
def TreeEnsembleQBCFunction(Model, df_Candidate, df_Train, UniqueErrorsInput):

    ### Ignore warning (taken care of) ###
    np.seterr(all = 'ignore') 
    warnings.filterwarnings("ignore", category=UserWarning)

    ### Set Up ###
    PredictionList = []
    BatchSize = 1000

    ### Predicted Values ###
    ## Rashomon Classification ##
    if 'TREEFARMS' in str(type(Model)):

        # Batch Prediction #
        TreeCounts = Model.get_tree_count()
        for i in range(0, TreeCounts, BatchSize):
            batch_end = min(i + BatchSize, TreeCounts)
            batch_predictions = []
            
            for tree_idx in range(i, batch_end):
                pred = Model[tree_idx].predict(
                    df_Candidate.loc[:, df_Candidate.columns != "Y"]
                )
                batch_predictions.append(pred)
            
            PredictionList.extend(batch_predictions)
            gc.collect()
        
        # Prediction Array #
        PredictionArray_Duplicate = pd.DataFrame(
            np.array(PredictionList),
            columns=df_Candidate.index.astype(str)
        )
        AllTreeCount = len(PredictionList)
        
        # Unique vs. Duplicate #
        if UniqueErrorsInput:
            PredictedValues = PredictionArray_Duplicate.drop_duplicates()
            UniqueTreeCount = len(PredictedValues)
        else:
            PredictedValues = PredictionArray_Duplicate
            UniqueTreeCount = len(PredictedValues.drop_duplicates())
        
        Output = {
            "AllTreeCount": AllTreeCount,
            "UniqueTreeCount": UniqueTreeCount
        }

    ## Random Forest Classification ###
    elif 'RandomForestClassifier' in str(type(Model)):
        
        # Batch Prediction #
        for i in range(0, Model.n_estimators, BatchSize):
            batch_end = min(i + BatchSize, Model.n_estimators)
            batch_predictions = []
            
            for tree_idx in range(i, batch_end):
                pred = Model.estimators_[tree_idx].predict(
                    df_Candidate.loc[:, df_Candidate.columns != "Y"]
                )
                batch_predictions.append(pred)
            
            PredictionList.extend(batch_predictions)
            gc.collect()
        
        # Prediction Array #
        PredictedValues = np.vstack(PredictionList)
        Output = {}

    ### Calculate vote entropy ###
    UniqueClasses = set(df_Train["Y"])
    VoteEntropy = np.zeros(len(df_Candidate))

    ## Vote entropy per class ##
    for classes in UniqueClasses:
        VoteC = np.mean(PredictedValues == classes, axis=0)
        LogVoteC = np.log(VoteC)
        CurrentEntropy = -VoteC * LogVoteC
        VoteEntropy += np.nan_to_num(CurrentEntropy, nan=0)

    ## Query Recommendation  ##
    df_Candidate["UncertaintyMetric"] = VoteEntropy
    IndexRecommendation = int(df_Candidate.sort_values(by="UncertaintyMetric", ascending=False).index[0])
    df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)
    Output["IndexRecommendation"] = IndexRecommendation
    
    ### Output ###
    return Output