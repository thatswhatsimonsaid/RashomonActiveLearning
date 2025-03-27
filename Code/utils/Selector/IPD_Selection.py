# Summary: 
# Input:
#   df_Train: The training set.
#   df_Candidate: The candidate set.
#   Model: The predictive model.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.


### Libraries ###
import numpy as np
from sklearn.linear_model import LinearRegression

def IPDSelection(Model, df_Train, df_Candidate):
    
    ### Data Prediction ###
    True_Train = np.array(df_Train["Y"]).reshape(-1, 1)
    Predict_Train = Model.predict(df_Train.loc[:,df_Train.columns != "Y"]).reshape(-1,1)
    Predict_Candidate = Model.predict(df_Candidate.loc[:,df_Candidate.columns != "Y"]).reshape(-1, 1)

    ### Relationship ###
    LinearRegressionModel_Relationship = LinearRegression()
    LinearRegressionModel_Relationship.fit(True_Train,Predict_Train)
    Predict_Relationship = LinearRegressionModel_Relationship.predict(Predict_Candidate) # Wait isn't doing this selection method literally just residual regression in AL

    ### Observation Recommended ###
    np.where(Predict_Relationship == np.max(Predict_Relationship))
    ArgMaxID = int(np.argmax(Predict_Relationship))
    IndexRecommendation = int(df_Candidate.iloc[ArgMaxID].name)

    ### Output ###
    Output = {"IndexRecommendation": IndexRecommendation}
    return(Output)
