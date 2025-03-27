# Summary: Meta learning query function for either random forest or Rashomon's TreeFarms that 
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
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

### Function ###
def MetaLearnerFunction(Model, df_Candidate, df_Train, UniqueErrorsInput):

    ### Ignore warning (taken care of) ###
    np.seterr(all = 'ignore') 
    warnings.filterwarnings("ignore", category=UserWarning)

    ### Predicted Values ###

    ## Rashomon Classification ##
    if 'TREEFARMS' in str(type(Model)):

        ## Tree Counts ##
        TreeCounts = Model.get_tree_count()

        ## TRAINING SET ##
        # Duplicate #
        PredictionArray_Duplicate_Training = pd.DataFrame(np.array([Model[i].predict(df_Train.loc[:, df_Train.columns != "Y"]) for i in range(TreeCounts)]))
        PredictionArray_Duplicate_Training.columns = df_Train.index.astype(str)
        EnsemblePrediction_Duplicate_Training = pd.Series(stats.mode(PredictionArray_Duplicate_Training)[0])
        EnsemblePrediction_Duplicate_Training.index = df_Train["Y"].index
        AllTreeCount_Duplicate__Training = PredictionArray_Duplicate_Training.shape[0]

        # Unique #
        PredictionArray_Unique_Training = pd.DataFrame(PredictionArray_Duplicate_Training).drop_duplicates()
        EnsemblePrediction_Unique_Training = pd.Series(stats.mode(PredictionArray_Unique_Training)[0])
        EnsemblePrediction_Unique_Training.index = df_Train["Y"].index
        UniqueTreeCount_Training = PredictionArray_Unique_Training.shape[0]


        ## CANDIDATE SET ##
        # Duplicate #
        PredictionArray_Duplicate_Candidate = pd.DataFrame(np.array([Model[i].predict(df_Candidate.loc[:, df_Candidate.columns != "Y"]) for i in range(TreeCounts)]))
        PredictionArray_Duplicate_Candidate.columns = df_Candidate.index.astype(str)
        EnsemblePrediction_Duplicate_Candidate = pd.Series(stats.mode(PredictionArray_Duplicate_Candidate)[0])
        EnsemblePrediction_Duplicate_Candidate.index = df_Candidate["Y"].index
        AllTreeCount_Candidate = PredictionArray_Duplicate_Candidate.shape[0]

        # Unique #
        PredictionArray_Unique_Candidate = pd.DataFrame(PredictionArray_Duplicate_Candidate).drop_duplicates()
        EnsemblePrediction_Unique_Candidate = pd.Series(stats.mode(PredictionArray_Unique_Candidate)[0])
        EnsemblePrediction_Unique_Candidate.index = df_Candidate["Y"].index
        UniqueTreeCount_Candidate = PredictionArray_Unique_Candidate.shape[0]

        ### Duplicate or Unique
        if UniqueErrorsInput:
            PredictedValues_Training = PredictionArray_Unique_Training
            PredictedValues_Candidate = PredictionArray_Unique_Candidate
        else:
            PredictedValues_Training = PredictionArray_Duplicate_Training
            PredictedValues_Candidate = PredictionArray_Duplicate_Candidate

        Output = {"AllTreeCount": AllTreeCount_Candidate,
                  "UniqueTreeCount": UniqueTreeCount_Candidate}

    ### Random Forest Classification ###
    elif 'RandomForestClassifier' in str(type(Model)):
        PredictedValues = [Model.estimators_[tree].predict(df_Candidate.loc[:, df_Candidate.columns != "Y"]) for tree in range(Model.n_estimators)] 
        PredictedValues = np.vstack(PredictedValues)
        Output = {}


    ### META LEARNER ###
    ### Meta Learner (Logistic Regression) ###
    LogisticModel = LogisticRegression()
    LogisticModel.fit(X = PredictedValues_Training.T,
                    y = df_Train["Y"])
    # CandidatePrediction = LogisticModel.predict(X = PredictedValues_Candidate.T)
    # np.mean(CandidatePrediction == df_Candidate["Y"])

    ### Confidence Score ###

    # Probability #
    Probabilities = LogisticModel.predict_proba(PredictionArray_Duplicate_Candidate.T)
    UncertaintyMetric1 = 1 - np.max(Probabilities, axis=1)           # Higher means more uncertain

    # Entropy #
    UncertaintyMetric2 = entropy(Probabilities, axis=1)

    # Min-Margin #
    SortedProbabilities = np.sort(Probabilities, axis=1)
    UncertaintyMetric3 = - (SortedProbabilities[:, -1] - SortedProbabilities[:, -2]) # Difference between top two probabilities (multiplied by 1 for IndexRecommendation  ascending = False below)

    ### Uncertainty Metric ###
    df_Candidate["UncertaintyMetric"] = UncertaintyMetric3
    IndexRecommendation = int(df_Candidate.sort_values(by = "UncertaintyMetric", ascending = False).index[0])
    df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)

    # Output #
    Output["IndexRecommendation"] = IndexRecommendation

    return Output