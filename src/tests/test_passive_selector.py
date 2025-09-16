### LIBRARIES ###
import pandas as pd
import numpy as np

### TEST THIS CLASS ###
from src.utils.selectors import PassiveSelector

### TEST FUNCTION ###
def test_passive_selector_selects_a_valid_index():
    """
    Tests that PassiveSelector returns a single, valid index from the candidate set.
    """
    ## 1. ARRANGE: Simulate a dataset 

    # Configuration for the mock dataset
    NUM_SAMPLES = 100
    NUM_FEATURES = 4
    RANDOM_SEED = 42 
    np.random.seed(RANDOM_SEED)

    # Generate data
    feature_data = {f'feature{i+1}': np.random.randn(NUM_SAMPLES) for i in range(NUM_FEATURES)}
    feature_data['Y'] = np.random.randint(0, 2, size=NUM_SAMPLES)
    realistic_index = range(1000, 1000 + NUM_SAMPLES)
    df_candidate = pd.DataFrame(feature_data, index=realistic_index)
    
    ## Selector instance
    selector = PassiveSelector(random_state=42)

    ## 2. ACT: Call the method
    result = selector.select(
        model=None,
        df_train=None,
        df_candidate=df_candidate
    )
    queried_index = result["IndexRecommendation"]

    ## 3. ASSERT: Check the output
    assert isinstance(queried_index, int)
    assert queried_index in df_candidate.index