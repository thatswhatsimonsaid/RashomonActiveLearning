# Summary: Initializes and fits a treefarms model.
# Input:
#   df_Train: The training data.
#   TopCModels: TopCModels top models
#   config:
# Output:
# treeFarmsModel: A treefarms model.

### Libraries ###
# from treeFarms.treefarms import TREEFARMS

import sys
# sys.path.append('/Users/simondn/Documents/RashomonActiveLearning/Code')
# from treeFarms.treefarms.model.threshold_guess import compute_thresholds
# from utils.Prediction.TreeFARMS import <function_or_class_name>
from treeFarms.treefarms.model.treefarms import TREEFARMS


### Function ###
def TreeFarmsFunction(df_Train, regularization, rashomon_bound_adder):
    ### Train TreeFarms Model ###
    config = {"regularization": regularization, "rashomon_bound_adder": rashomon_bound_adder}
    TreeFarmsModel = TREEFARMS(config)
    TreeFarmsModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    
    ### Return ###
    return TreeFarmsModel


# NOTE: Is there a way to prune the tree such that only the top models are given back? Look into this
