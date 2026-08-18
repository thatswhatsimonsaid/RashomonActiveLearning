### Summary ###
"""
Contains the main engine for running an active learning simulation.
"""

### Libraries ###
import sys
import time
import gc
import pandas as pd
from tqdm import tqdm 
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from src.utils.query_strategies import Selector
from src.utils.models import ModelWrapper, evaluate_models

### Configurations ###
@dataclass
class SimulationConfig:
    """A structured class to hold all simulation parameters."""
    selector_model: ModelWrapper  
    predictor_model: ModelWrapper 
    oracle_model: ModelWrapper
    selector: Selector
    df_train: pd.DataFrame
    df_candidate: pd.DataFrame
    df_test: pd.DataFrame 

@dataclass
class SimulationResult:
    """A structured class to hold the metrics of a simulation."""
    accuracy_history: List[float] = field(default_factory=list)
    f1_history: List[float] = field(default_factory=list)
    oracle_agreement_history: List[float] = field(default_factory=list)
    tree_edit_distance_history: List[float] = field(default_factory=list)
    calibrated_params: Dict[str, Any] = field(default_factory=dict)
    elapsed_time: Optional[float] = None    
    selection_history: List[Optional[int]] = field(default_factory=list)    
    epsilon_history: List[float] = field(default_factory=list)
    rashomon_size_history: List[int] = field(default_factory=list)
    committee_size_history: List[float] = field(default_factory=list)
    entropy_history: List[Dict[int, float]] = field(default_factory=list)

### Learning Procedure Function ###
def run_learning_procedure(config: SimulationConfig, calibrated_params: Dict[str, Any] = None) -> SimulationResult:
    """
    Runs the full active learning loop, including the final evaluation.
    """

    ## Initialize ##
    start_time = time.time()
    df_train = config.df_train.copy()
    df_candidate = config.df_candidate.copy()    
    results = SimulationResult()
    if calibrated_params:
        results.calibrated_params = calibrated_params
    total_iterations = len(df_candidate) + 1
    
    ## Active Learning Loop ##
    for i in tqdm(range(total_iterations), desc="Active Learning Iterations", file=sys.stdout):
        
        # 1. Initialize Step #
        X_train = df_train.drop(columns="Y")
        y_train = df_train["Y"]
        
        # 2. Fit Models #
        config.predictor_model.fit(X_train, y_train)
        config.selector_model.fit(X_train, y_train)
        
        # 3. Evaluate Everything #
        metrics = evaluate_models(
            predictor_model=config.predictor_model,
            oracle_model=config.oracle_model,
            df_test=config.df_test
        )
        
        # 4. Save Metrics Immediately #
        results.accuracy_history.append(metrics["accuracy"])
        results.f1_history.append(metrics["f1_micro"])
        results.oracle_agreement_history.append(metrics["oracle_agreement"])
        results.tree_edit_distance_history.append(metrics["tree_edit_distance"])
        if hasattr(config.selector_model, "epsilon"):
            results.epsilon_history.append(config.selector_model.epsilon)
        if hasattr(config.selector_model, "get_rashomon_size"):
             results.rashomon_size_history.append(config.selector_model.get_rashomon_size())
             
        # 5. Check Termination Condition (iie. Full Dataset Reached)
        if df_candidate.empty:
            results.selection_history.append(None)
            results.entropy_history.append({})
            break 

       # 6. Select Next Sample 
        selection_output = config.selector.select(
            model=config.selector_model,
            df_train=df_train,
            df_candidate=df_candidate
        )
        queried_index = selection_output["IndexRecommendation"]   
        
        if hasattr(config.selector, "effective_committee_size_"):
             results.committee_size_history.append(config.selector.effective_committee_size_)
        elif hasattr(config.selector, "committee_size_"):
             results.committee_size_history.append(float(config.selector.committee_size_))
             
        # 7. Update Datasets
        queried_observation = df_candidate.loc[[queried_index]]
        df_train = pd.concat([df_train, queried_observation])
        df_candidate = df_candidate.drop(queried_index)

        # 8. Save Selection Metadata
        results.selection_history.append(queried_index)
        if "AllEntropies" in selection_output:
            results.entropy_history.append(selection_output["AllEntropies"].to_dict())
        else:
            results.entropy_history.append({})
        
        # Cleanup
        gc.collect()

    ## Finalize ##
    results.elapsed_time = time.time() - start_time
    
    ## Return ##
    return results