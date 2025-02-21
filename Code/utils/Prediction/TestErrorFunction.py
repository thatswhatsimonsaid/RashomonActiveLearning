# Summary: Calculates the loss (RMSE for regression and classification error for classification) of the test set.
# Input:
#   InputModel: The prediction model used.
#   df_Test: The test data.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output:
# RMSE: The residual mean squared error of the predicted values and their true values in the test set. 

### Libraries ###
import gc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from .BatchPrediction import *

### Function ###
def TestErrorFunction(InputModel, df_Test, Type):

    ### RMSE ###
    if(Type == "Regression"):
        Prediction = BatchPredictFunction(InputModel,df_Test.loc[:, df_Test.columns != "Y"])
        ErrorVal = np.mean((Prediction - df_Test["Y"])**2)
        ErrorVal = [ErrorVal.tolist()]
        return {"ErrorVal": ErrorVal}

    ### Classification Error ###
    if(Type == "Classification"):
        
        ## Rashomon Classification ##
        if 'TREEFARMS' in str(type(InputModel)):

            # Set Up #
            TreeCounts = InputModel.get_tree_count()
            PredictionList = []
            BatchSize = 100
            
            # Batch Predictions #
            for i in range(0, TreeCounts, BatchSize):
                BatchEnd = min(i + BatchSize, TreeCounts)
                BatchPredictions = []
                
                for tree_idx in range(i, BatchEnd):
                    pred = InputModel[tree_idx].predict(
                        df_Test.loc[:, df_Test.columns != "Y"]
                    )
                    BatchPredictions.append(pred)
                PredictionList.extend(BatchPredictions)
                gc.collect()

            # Convert to DataFrame #
            PredictionArray_Duplicate = pd.DataFrame(
                np.array(PredictionList),
                columns=df_Test.index.astype(str))
            
            # Calculate ensemble prediction #
            EnsemblePrediction_Duplicate = pd.Series(
                stats.mode(PredictionArray_Duplicate)[0],
                index=df_Test["Y"].index)
            
            # Calculate error
            Error_Duplicate = float(
                f1_score(df_Test["Y"], EnsemblePrediction_Duplicate, average='micro'))

            # Output #
            return {"Error_Duplicate": Error_Duplicate}

        else:
            Prediction = BatchPredictFunction(InputModel, df_Test.loc[:, df_Test.columns != "Y"])
            ErrorVal = float(f1_score(df_Test["Y"], Prediction, average='micro'))
            return {"ErrorVal": ErrorVal}

            