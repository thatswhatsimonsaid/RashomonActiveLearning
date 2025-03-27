# Summary: Chooses an index at random from the candidate set to be queried.
# Input:
#   df_Candidate: The candidate set.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

### Libraries ###
# import pandas as pd

def PassiveLearning(df_Candidate):
    ### Passive Sampling ###
    QueryObservation = df_Candidate.sample(n=1)
    IndexRecommendation = QueryObservation.index[0]

    Output = {"IndexRecommendation": int(IndexRecommendation)}
    return(Output)