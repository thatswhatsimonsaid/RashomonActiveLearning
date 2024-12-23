# Summary: Query-by-committee function for either random forest or Rashomon's TreeFarms that 
#          recommends an observation from the candidate set to be queried.
# Input:
#   Model: The predictive model.
#   df_Candidate: The candidate set.
#   df_Train: The training set.
#   AllErrors: The test errors from each model in the Rashomon set.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFarms.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

# NOTE: Incorporate covariate GSx in selection criteria? Good for tie breakers.

### Libraries ###
import warnings
import numpy as np
from scipy.spatial.distance import cdist

### Function ###
def TreeEnsembleQBCFunction(Model, df_Candidate, df_Train, AllErrors, UniqueErrorsInput):

    ### Ignore warning (taken care of) ###
    np.seterr(all = 'ignore') 
    warnings.filterwarnings("ignore", category=UserWarning)

    # ### GSx: Incorporate? Good for tie breakers. Could also do GSx on the data set as opposed to the one-hot set
    # d_nmX = cdist(df_Candidate.loc[:,df_Candidate.columns!= "Y"], df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)
    # d_nX = d_nmX.min(axis=1)

    ### Predicted Values ###
    if 'TREEFARMS' in str(type(Model)):                                                                         # TreeFarms

        ## Unique Errors ##
        if UniqueErrorsInput:
            AllErrorsArray = np.array(AllErrors)
            UniqueErrors = sorted(set(AllErrors))                                                               # NOTE: NEED TO CORRECTLY CATEGORIZE CLASSIFICATION PATTERNS HERE
            LowestErrorIndices = [int(np.where(AllErrorsArray == error)[0][0]) for error in UniqueErrors]
        else:
            LowestErrorIndices = np.argsort(AllErrors)

        ## Prediction ##
        PredictedValues = [Model[i].predict(df_Candidate) for i in LowestErrorIndices]

    elif 'RandomForestClassifier' in str(type(Model)):                                                          # RandomForest
        PredictedValues = [Model.estimators_[tree].predict(df_Candidate.loc[:, df_Candidate.columns != "Y"]) for tree in range(Model.n_estimators)] 
    
    ### Stack values ###
    PredictedValues = np.vstack(PredictedValues)

    ### Vote Entropy ###
    VoteC = {}
    LogVoteC = {}
    VoteEntropy = {}
    UniqueClasses = set(df_Train["Y"])

    # Vote entropy per class #
    for classes in UniqueClasses:
        VoteC[classes] = np.mean(PredictedValues == classes, axis=0)
        LogVoteC[classes] = np.log(VoteC[classes])
        VoteEntropy[classes] =  - VoteC[classes] * LogVoteC[classes]
        VoteEntropy[classes] = np.nan_to_num(VoteEntropy[classes], nan=0)

    # Vote Entropy #
    VoteEntropyMatrix = np.stack(list(VoteEntropy.values()), axis=1)
    VoteEntropyFinal = np.sum(VoteEntropyMatrix, axis=1)

    ### Uncertainty Metric ###
    df_Candidate["UncertaintyMetric"] = VoteEntropyFinal
    IndexRecommendation = int(df_Candidate.sort_values(by = "UncertaintyMetric", ascending = False).index[0])
    df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)
    
    return IndexRecommendation

