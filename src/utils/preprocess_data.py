### Libraries ###
import pandas as pd
import numpy as np
import io
import requests
import pickle
from pathlib import Path

### Paths ###
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "src/data" 
RAW_URL_BASE = "https://raw.githubusercontent.com/ConSol-Lab/pysortd/main/data/accuracy"

### Import datasets ###
DATASET_FILES = [
    "HTRU2.csv" # Leave out - too big/long?
    "anneal.csv",
    "bank_marketing.csv",
    "banknote_authentication.csv", # Leave out - over 20 datasets.
    "bar-7.csv",
    "biodeg.csv", # Leave out - too big/long?
    "breast_cancer_wisconsin.csv",
    "car_evaluation.csv", 
    "cheap_restaurant.csv", 
    "coffee_house.csv", 
    "expensive_restaurant.csv",
    "haberman.csv",
    "hepatitis.csv",
    "hypothyroid.csv",
    "kr-vs-kp.csv", # Leave out - too big (but want to include)?
    "lymph.csv",
    "monk1.csv",
    "monk2.csv",
    "monk3.csv",
    "primary-tumor.csv",
    "spect.csv",
    "tic-tac-toe.csv",
    "vote.csv",
    "yeast.csv"
]

### Additional Datasets (TreeFarms Repository) ###
TREEFARMS_DATASETS = {
    "fico": "https://raw.githubusercontent.com/ubc-systopia/treeFarms/main/experiments/datasets/fico/fico-binary.csv",
    "compas": "https://raw.githubusercontent.com/ubc-systopia/treeFarms/main/experiments/datasets/compas/binned.csv",
}

def process_treefarms_dataset(name: str, url: str):
    print(f"\n[{name}] Fetching from TreeFarms...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # 1. Parse (standard CSV with headers)
        df = pd.read_csv(io.StringIO(response.text))
        
        # 2. Rename last column to 'Y'
        cols = list(df.columns)
        cols[-1] = "Y"
        df.columns = cols
        
        # 3. Convert to int
        df = df.astype(int)
        
        # 4. Save
        _save_pickle(df, DATA_DIR / f"{name}.pkl")
        
    except Exception as e:
        print(f"  [ERROR] {e}")


def generate_synthetic_study(
    n_samples=500, 
    n_features=20, 
    alpha=0.0, 
    phi=0.0, 
    random_state=42
):
    """
    Unified Data Generating Process for UNREAL stress tests.
    All features are binary {0, 1}.
    
    Args:
        n_samples: Number of observations.
        n_features: Total covariates (only first two are signal).
        alpha: Misspecification parameter [0, 1]. 
               0.0 = Pure XOR (Tree-based)
               1.0 = Pure Linear Threshold (Non-tree-based)
        phi: Label noise parameter [0, 0.5].
             Fraction of Class-1 labels flipped to Class-0.
        random_state: Seed for reproducibility.
    """
    np.random.seed(random_state)
    
    # 1. Generate Data
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    y_tree = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    
    # 2. Linear part
    weights = np.random.randn(n_features)
    y_linear = (X @ weights > 0).astype(int)
    
    # 3. Blend labels based on alpha
    selector = np.random.rand(n_samples) < alpha
    Y = np.where(selector, y_linear, y_tree)
    
    # 4. Apply symmetric label noise
    if phi > 0:
        flip_mask = np.random.rand(n_samples) < phi        
        Y = np.logical_xor(Y, flip_mask).astype(int)
            
    # 6. Final DataFrame
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(n_features)])
    df["Y"] = Y
    
    return df

def generate_parity_study(
    n_samples=1000, 
    n_bits=3, 
    n_noise=0, 
    random_state=42
):
    """
    Generates an n-bit parity dataset. 
    Target Y = 1 if the sum of the first n_bits features is even, else 0.
    The remaining n_noise features are purely random distractors.
    """
    np.random.seed(random_state)
    n_total = n_bits + n_noise
    
    X = np.random.randint(0, 2, size=(n_samples, n_total))    
    Y = (np.sum(X[:, :n_bits], axis=1) % 2 == 0).astype(int)
            
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(n_total)])
    df["Y"] = Y
    
    return df

### Auxiliary functions ###
def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _save_pickle(df: pd.DataFrame, path: Path):
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    print(f"    > Saved to {path.name} | Shape: {df.shape}")

### Download and process datasets ###
def process_dataset_url(filename: str):
    name = filename.replace(".csv", "")
    url = f"{RAW_URL_BASE}/{filename}"
    print(f"\n[{name}] Fetching...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text), sep=r'\s+', header=None, engine='python')
        if df.shape[1] < 2:
            print(f"  [ERROR] Parsed only {df.shape[1]} column. Skipping.")
            return

        df.columns = [f"V{i}" for i in range(df.shape[1] - 1)] + ["Y"]        
        df = df.astype(int)

        _save_pickle(df, DATA_DIR / f"{name}.pkl")
        
    except Exception as e:
        print(f"  [ERROR] {e}")

### Main ###
def main():
    print(f"--- PREPROCESSING DATASETS TO {DATA_DIR} ---")
    _ensure_dir(DATA_DIR)
    
    ## 1. Process Repository Datasets ##
    for filename in DATASET_FILES:
        process_dataset_url(filename)

    ## 1b. Process TreeFarms Datasets ##
    for name, url in TREEFARMS_DATASETS.items():
        process_treefarms_dataset(name, url)
        
    ## 2. Generate Synthetic Datasets ##
    alpha=0.0, 
    phi=0.0, 
    random_state=42

    # 2a. Baseline (Standard XOR - Alpha=0, Phi=0)
    print("\nGenerating Baseline XOR...")
    df_base = generate_synthetic_study(alpha=0.0, phi=0.0, n_samples=500, n_features=20)
    _save_pickle(df_base, DATA_DIR / "Synthetic_XOR_Baseline.pkl")

    # 2b. Misspecification Study (Varying Alpha)
    alphas = [0.25, 0.50, 0.75, 1.0]
    for a in alphas:
        print(f"Generating Misspecification Study (Alpha={a})...")
        df_alpha = generate_synthetic_study(alpha=a, phi=0.0, n_samples=500, n_features=20)
        dataset_name = f"Synthetic_XOR_Alpha_{int(a*100):02d}"
        _save_pickle(df_alpha, DATA_DIR / f"{dataset_name}.pkl")
        
    # 2c. Label Noise Study (Varying Phi)
    phis = [0.05, 0.10, 0.25, 0.45]
    for p in phis:
        print(f"Generating Label Noise Study (Phi={p})...")
        df_phi = generate_synthetic_study(alpha=0.0, phi=p, n_samples=500, n_features=20)
        dataset_name = f"Synthetic_XOR_Phi_{int(p*100):02d}"
        _save_pickle(df_phi, DATA_DIR / f"{dataset_name}.pkl")

    ## 3. Generate Parity Study (Varying Covariates) ##
    noise_levels = [0, 6, 16, 26]
    for n_noise in noise_levels:
        dataset_name = f"Parity_8bit_Noise_{n_noise:02d}"
        print(f"\n--- Generating: {dataset_name} ---")
        
        df_parity = generate_parity_study(
            n_samples=100, 
            n_bits=3, 
            n_noise=n_noise
        )
        _save_pickle(df_parity, DATA_DIR / f"{dataset_name}.pkl")

        
if __name__ == "__main__":
    main()