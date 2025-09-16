"""
Contains the main engine for running an active learning simulation.
"""

### LIBRARIES ###
from dataclasses import dataclass, field
import time
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from tqdm import tqdm 
from src.utils.models import ModelWrapper, evaluate_model
from src.utils.selectors import Selector

### SIMULATION CONFIGURATION AND RESULTS ###
@dataclass
class SimulationConfig:
    """A structured class to hold all simulation parameters."""
    model: ModelWrapper
    selector: Selector
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_candidate: pd.DataFrame

@dataclass
class SimulationResult:
    """A structured class to hold the results of a simulation."""
    accuracy_history: List[float] = field(default_factory=list)
    f1_history: List[float] = field(default_factory=list)
    elapsed_time: Optional[float] = None
    selection_history: List[int] = field(default_factory=list)

### RUN LEARNING PROCEDURE ###
def run_learning_procedure(config: SimulationConfig) -> SimulationResult:
    """
    Runs the full active learning loop.

    Args:
        config: A SimulationConfig object containing all necessary components
                (model, selector, data).

    Returns:
        A SimulationResult object containing the history of the run.
    """
    ## Time ##
    start_time = time.time()

    ## Copy data ##
    df_train = config.df_train.copy()
    df_candidate = config.df_candidate.copy()
    df_test = config.df_test.copy()
    
    ## Initialize results ##
    results = SimulationResult()
    
    ## Active Learning Loop ##
    num_iterations = len(df_candidate)
    for _ in tqdm(range(num_iterations), desc="Active Learning Iterations"):
        
        # 1. Separate features and target for the current training set
        X_train = df_train.drop(columns="Y")
        y_train = df_train["Y"]

        # 2. Fit the model on the current training data
        config.model.fit(X_train, y_train)

        # 3. Evaluate the model's performance on the test set
        eval_metrics = evaluate_model(config.model, df_test)
        results.accuracy_history.append(eval_metrics["accuracy"])
        results.f1_history.append(eval_metrics["f1_micro"])

        # 4. Use the selector to choose the next sample
        selection_output = config.selector.select(
            model=config.model,
            df_train=df_train,
            df_candidate=df_candidate
        )
        queried_index = selection_output["IndexRecommendation"]
        
        if queried_index is None:
            print("No more candidates to select. Ending simulation.")
            break
        
        results.selection_history.append(queried_index)

        # 5. Update the training and candidate sets
        queried_observation = df_candidate.loc[[queried_index]]
        df_train = pd.concat([df_train, queried_observation], ignore_index=True)
        df_candidate = df_candidate.drop(queried_index)

    ## Time ##
    results.elapsed_time = time.time() - start_time
    
    ## Return ##
    return results