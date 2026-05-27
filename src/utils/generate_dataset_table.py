### Libraries ###
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

### Path Setup ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "study1_active_learning"
FRAGMENTS_DIR = OUTPUT_DIR / "Tables" / "table_fragments"
OUTPUT_FILENAME = "DatasetTable.tex"
sys.path.append(str(PROJECT_ROOT))
from src.utils.models import PySORTDWrapper

### Configuration ###
PYSORTD_CONFIG = {
    "regularization": 0.001,
    "rashomon_multiplier": 0.1,  
    "max_num_trees": 100_000,     
    "max_depth": 3,
    "time_limit": 20000,
}
BETA = 10.0

META_INFO = {
    "anneal":                       ("Anneal",                   "UCI"),    # Dataset 1
    "bank_marketing":               ("Bank Marketing",           "UCI"),    # Dataset 2
    "banknote_authentication":      ("Banknote Auth.",            "UCI"),   # Dataset 3 
    "bar-7":                        ("Bar-7",                    "SORTD"),  # Dataset 4
    "breast_cancer_wisconsin":      ("Breast Cancer WI",         "UCI"),    # Dataset 5
    "car_evaluation":               ("Car Evaluation",           "UCI"),    # Dataset 6 
    "cheap_restaurant":             ("Cheap Restaurant",         "SORTD"),  # Dataset 7
    "coffee_house":                 ("Coffee House",             "SORTD"),  # Dataset 8
    "compas":                       ("COMPAS",                   "ProPublica"), # Dataset 9
    "expensive_restaurant":         ("Expensive Restaurant",     "SORTD"),  # Dataset 10
    "fico":                         ("FICO (HELOC)",             "FICO"),   # Dataset 11
    "haberman":                     ("Haberman",                 "UCI"),    # Dataset 12
    "hepatitis":                    ("Hepatitis",                "UCI"),    # Dataset 13
    "hypothyroid":                  ("Hypothyroid",              "UCI"),    # Dataset 14
    "lymph":                        ("Lymphography",             "UCI"),    # Dataset 15
    "monk2":                        ("MONK-2",                   "UCI"),    # Dataset 16
    "primary-tumor":                ("Primary Tumor",            "UCI"),    # Dataset 17
    "tic-tac-toe":                  ("Tic-Tac-Toe",              "UCI"),    # Dataset 18
    "vote":                         ("Congressional Vote",       "UCI"),    # Dataset 19
    "yeast":                        ("Yeast",                    "UCI"),    # Dataset 20
    "Synthetic_XOR_Baseline":       (r"Synth.\ XOR ($\alpha$=0, $\phi$=0)",        "Synthetic"),
    "Synthetic_XOR_Alpha_25":       (r"Synth.\ XOR ($\alpha$=0.25)",               "Synthetic"),
    "Synthetic_XOR_Alpha_50":       (r"Synth.\ XOR ($\alpha$=0.50)",               "Synthetic"),
    "Synthetic_XOR_Alpha_75":       (r"Synth.\ XOR ($\alpha$=0.75)",               "Synthetic"),
    "Synthetic_XOR_Alpha_100":      (r"Synth.\ XOR ($\alpha$=1.00)",               "Synthetic"),
    "Synthetic_XOR_Phi_05":         (r"Synth.\ XOR ($\phi$=0.05)",                 "Synthetic"),
    "Synthetic_XOR_Phi_10":         (r"Synth.\ XOR ($\phi$=0.10)",                 "Synthetic"),
    "Synthetic_XOR_Phi_25":         (r"Synth.\ XOR ($\phi$=0.25)",                 "Synthetic"),
    "Synthetic_XOR_Phi_45":         (r"Synth.\ XOR ($\phi$=0.45)",                 "Synthetic"),
}

def load_dataset(file_key: str) -> pd.DataFrame:
    path = DATA_DIR / f"{file_key}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
    
def compute_gibbs_ecs(wrapper, X, y) -> float:
    """Computes the Effective Committee Size (ECS) using Gibbs weights."""
    losses = wrapper.get_ensemble_losses(X, y)
    if len(losses) == 0: return 0
    
    adj_losses = losses - np.min(losses)
    unnormalized_weights = np.exp(-BETA * adj_losses)
    weights = unnormalized_weights / np.sum(unnormalized_weights)
    
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    ecs = np.exp(entropy)
    return float(ecs)

def compute_dataset_stats(file_key: str):
    df = load_dataset(file_key)
    X, y = df.drop(columns="Y"), df["Y"]
    
    wrapper = PySORTDWrapper(**PYSORTD_CONFIG)
    wrapper.fit(X, y)
    
    return {
        "n_samples": len(df),
        "n_features": X.shape[1],
        "oracle_acc": float(np.mean(wrapper.predict(X) == y.values)) * 100,
        "rashomon_size": wrapper.get_rashomon_size(),
        "ecs": compute_gibbs_ecs(wrapper, X, y)
    }

def merge_and_generate_latex():
    rows = []
    for i, (file_key, (display_name, source)) in enumerate(META_INFO.items(), 1):
        frag_path = FRAGMENTS_DIR / f"{file_key}.pkl"
        if frag_path.exists():
            with open(frag_path, "rb") as f:
                stats = pickle.load(f)
            rows.append({"no": i, "name": display_name, "source": source, **stats})
    
    if not rows: return
    
    # Updated table structure with 3 fewer columns (8 total r/l/r parameters)
    latex = r"""\begin{table*}[htbp]
    \centering
    \scriptsize
    \begin{tabular}{rllrrrrr}
        \toprule
        \textbf{No.} & \textbf{Dataset} & \textbf{Src} & $N$ & $d$ & \textbf{Orc\%} & $|\hat{\mathcal{R}}|$ & \textbf{ECS} \\ 
        \midrule
"""
    prev_source = None
    for row in rows:
        if prev_source and row["source"] == "Synthetic" and prev_source != "Synthetic":
            latex += r"        \midrule" + "\n"
        prev_source = row["source"]        
        latex += (f"        {row['no']} & {row['name']} & {row['source'][:3]} & "
              f"{row['n_samples']:,} & {row['n_features']} & "
              f"{row['oracle_acc']:.1f} & "
              f"{row['rashomon_size']:,} & {row['ecs']:,.1f} \\\\\n")
              
    latex += r"""        \bottomrule
    \end{tabular}
\end{table*}
"""
    with open(OUTPUT_DIR / "Tables" / OUTPUT_FILENAME, "w") as f:
        f.write(latex)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_and_generate_latex()
    elif args.dataset:
        FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)
        stats = compute_dataset_stats(args.dataset)
        with open(FRAGMENTS_DIR / f"{args.dataset}.pkl", "wb") as f:
            pickle.dump(stats, f)