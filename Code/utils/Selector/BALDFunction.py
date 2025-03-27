### Libraries ###
import gc
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

### Function ###
def TreeEnsembleBALDFunction(Model, df_Candidate, df_Train, UniqueErrorsInput):

    ### Ignore warning (taken care of) ###
    np.seterr(all = 'ignore') 
    warnings.filterwarnings("ignore", category=UserWarning)

    ### Set Up ###
    PredictionList = []
    BatchSize = 1000
    UniqueClasses = sorted(list(set(df_Train["Y"])))
    ClassN = len(UniqueClasses)
    ClassMapping = {cls: idx for idx, cls in enumerate(UniqueClasses)}

    ### Predicted Values ###
    ## Random Forest Classification ###
    if 'RandomForestClassifier' in str(type(Model)):
        
        # Batch Prediction #
        for i in range(0, Model.n_estimators, BatchSize):
            BatchEnd = min(i + BatchSize, Model.n_estimators)
            batch_predictions = []
            
            for TreeIndex in range(i, BatchEnd):
                # Get the actual tree
                tree = Model.estimators_[TreeIndex]
                
                # For each class, get the probability
                if hasattr(tree, "predict_proba"):
                    # If tree can predict probabilities directly
                    probs = tree.predict_proba(df_Candidate.loc[:, df_Candidate.columns != "Y"])
                else:
                    # Otherwise, use one-hot encoding of predictions
                    preds = tree.predict(df_Candidate.loc[:, df_Candidate.columns != "Y"])
                    probs = np.zeros((len(preds), ClassN))
                    for j, pred in enumerate(preds):
                        probs[j, ClassMapping[pred]] = 1.0
                
                batch_predictions.append(probs)
            
            PredictionList.extend(batch_predictions)
            gc.collect()
        
        # Stack all predictions
        tree_probas = np.stack(PredictionList)  # Shape: (n_trees, n_samples, n_classes)
        
    # ## Rashomon Tree Ensemble ##    
    # elif 'TREEFARMS' in str(type(Model)):
        
    #     # Batch Prediction #
    #     TreeCounts = Model.get_tree_count()
    #     for i in range(0, TreeCounts, BatchSize):
    #         BatchEnd = min(i + BatchSize, TreeCounts)
    #         batch_predictions = []
            
    #         for TreeIndex in range(i, BatchEnd):
    #             # For each class, check if the prediction matches
    #             raw_preds = Model[TreeIndex].predict(df_Candidate.loc[:, df_Candidate.columns != "Y"])
                
    #             # Convert to one-hot probabilities
    #             probs = np.zeros((len(raw_preds), ClassN))
    #             for j, pred in enumerate(raw_preds):
    #                 probs[j, ClassMapping[pred]] = 1.0
                
    #             batch_predictions.append(probs)
            
    #         PredictionList.extend(batch_predictions)
    #         gc.collect()
        
    #     # Stack all predictions
    #     tree_probas_full = np.stack(PredictionList)  # Shape: (n_trees, n_samples, n_classes)
        
    #     # Handle unique trees if requested
    #     if UniqueErrorsInput:
    #         # Reshape to 2D for duplicate checking
    #         tree_probas_reshaped = tree_probas_full.reshape(tree_probas_full.shape[0], -1)
            
    #         # Find unique rows
    #         unique_indices = []
    #         seen = set()
    #         for i, row_tuple in enumerate(map(tuple, tree_probas_reshaped)):
    #             if row_tuple not in seen:
    #                 seen.add(row_tuple)
    #                 unique_indices.append(i)
            
    #         tree_probas = tree_probas_full[unique_indices]
    #         UniqueTreeCount = len(unique_indices)
    #     else:
    #         tree_probas = tree_probas_full
    #         # Count unique trees for reporting
    #         tree_probas_reshaped = tree_probas.reshape(tree_probas.shape[0], -1)
    #         UniqueTreeCount = len(set(map(tuple, tree_probas_reshaped)))
        
    #     AllTreeCount = len(tree_probas)
        
    #     Output = {
    #         "AllTreeCount": AllTreeCount,
    #         "UniqueTreeCount": UniqueTreeCount
    #     }
    
    ### Calculate BALD score ###
    MeanPrediction = np.mean(tree_probas, axis=0)  # Calculate mean prediction across trees
    EntropyMean = -np.sum(MeanPrediction * np.log(MeanPrediction + 1e-10), axis=1) # Calculate entropy of mean prediction
    entropies = -np.sum(tree_probas * np.log(tree_probas + 1e-10), axis=2)  # Calculate entropy of each individual prediction (Add small epsilon to avoid log(0))
    MeanEntropy = np.mean(entropies, axis=0)  # Calculate mean of entropies
    BaldScores = EntropyMean - MeanEntropy #BALD score = entropy of mean - mean of entropies

    ## Query Recommendation  ###
    df_Candidate["UncertaintyMetric"] = BaldScores
    IndexRecommendation = int(df_Candidate.sort_values(by="UncertaintyMetric", ascending=False).index[0])
    df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)

    ### Output ###
    Output = {}
    Output["IndexRecommendation"] = IndexRecommendation
    return Output