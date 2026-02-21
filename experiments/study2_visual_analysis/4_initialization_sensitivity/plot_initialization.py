### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
from scipy import stats

### Analysis Parameters ###
BOUNDARY_ZONE = 0.2
DISTANCE_METRIC = 'euclidean'

### PATHS ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
BASE_STUDY_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "4_initialization_sensitivity"
INPUT_DIR = BASE_STUDY_DIR / "raw"
PLOT_ROOT = BASE_STUDY_DIR / "plots"


from experiments.study2_visual_analysis.utils import (
    true_dgp, distance_to_boundary, GRID_RESOLUTION, 
    run_corrected_t_test
)

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

# ==============================================================================
# --- DATA LOADING ---
# ==============================================================================

def load_and_aggregate():
    files = list(INPUT_DIR.glob("init_task_*.pkl"))
    if not files:
        print(f"No files found in {INPUT_DIR}")
        return None

    print(f"Found {len(files)} files. Aggregating...")
    aggregated = {} 

    for f in files:
        try:
            with open(f, 'rb') as file:
                res = pickle.load(file)
            scen = res['scenario']
            if scen not in aggregated:
                aggregated[scen] = {'QBC-RF': [], 'UNREAL': []}
            aggregated[scen]['QBC-RF'].append(res['QBC-RF'])
            aggregated[scen]['UNREAL'].append(res['UNREAL'])
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return aggregated

# ==============================================================================
# --- PLOTTING FUNCTIONS ---
# ==============================================================================

def plot_heatmap(avg_rf, avg_unreal, output_dir, scenario_name, n_runs):
    """Generates 2D Heatmap comparison."""
    xx, yy = np.meshgrid(np.linspace(0, 1, GRID_RESOLUTION), np.linspace(0, 1, GRID_RESOLUTION))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_true = np.array([true_dgp(x) for x in grid_points]).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    configs = [(axes[0], avg_rf, "QBC-RF"), (axes[1], avg_unreal, "UNREAL")]
    
    array_indices = [(8,8), (8,25), (8,41), (25,8), (25,25), (25,41), (41,8), (41,25), (41,41)]
    x_coords = [0.165, 0.50, 0.835]
    y_coords = [0.165, 0.50, 0.835]

    for ax, data, title in configs:
        im = ax.contourf(xx, yy, data, levels=np.linspace(0, 1.01, 100), cmap='viridis', vmin=0, vmax=1)
        ax.contour(xx, yy, Z_true, colors='white', linestyles='--', levels=[0.5], linewidths=2)
        
        # Numerical Labels
        for i, (r, c) in enumerate(array_indices):
            try:
                val = data[r, c]
                ax.text(x_coords[i%3], y_coords[i//3], f"{val:.2f}", 
                        color='black' if val < 0.5 else 'white', 
                        ha='center', va='center', fontsize=12, weight='bold')
            except: pass
            
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel("X1"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        
    axes[0].set_ylabel("X2")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Average Vote Entropy", rotation=270, labelpad=20)
    fig.suptitle(f"{scenario_name}\n(Average over {n_runs} runs)", fontsize=18, y=1.05)
    
    plt.savefig(output_dir / f"heatmap_{scenario_name}.png", bbox_inches='tight')
    plt.close()

def plot_boundary_line(df_data, output_dir, scenario_name):
    """Generates Line Plot (Entropy vs Distance)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Binning
    df_binned = df_data.assign(Bin=pd.cut(df_data['Distance'], 50)).groupby('Bin').mean(numeric_only=True).reset_index()
    
    ax.plot(df_binned['Distance'], df_binned['VE_UNREAL'], label='UNREAL', color='darkblue', linewidth=3)
    ax.plot(df_binned['Distance'], df_binned['VE_RF'], label='QBC-RF', color='darkred', linewidth=3)
    
    ax.axvline(0.0, color='gray', linestyle='--')
    ax.axvspan(-BOUNDARY_ZONE, BOUNDARY_ZONE, color='lightgray', alpha=0.3)
    
    ax.set_title(f"Boundary Sensitivity: {scenario_name}")
    ax.set_xlabel(f"Distance from Boundary ({DISTANCE_METRIC})")
    ax.set_ylabel("Average Vote Entropy")
    ax.legend()
    
    plt.savefig(output_dir / f"boundary_sensitivity_{scenario_name}.png", bbox_inches='tight')
    plt.close()

# ==============================================================================
# --- STATISTICAL COMPARISON ---
# ==============================================================================

def save_statistical_comparison(df_zone, output_dir, scenario_name):
    """
    Runs both Naive and Corrected T-Tests and saves a comparison CSV.
    """
    ve_unreal = df_zone['VE_UNREAL']
    ve_rf = df_zone['VE_RF']
    
    # 1. Uncorrected (Naive) Paired T-Test
    t_stat_naive, p_val_naive = stats.ttest_rel(ve_unreal, ve_rf)
    mean_diff = ve_unreal.mean() - ve_rf.mean()
    
    # 2. Corrected T-Test (Using utils logic)
    corr_results = run_corrected_t_test(ve_unreal, ve_rf)
    
    # 3. Construct DataFrame
    data = {
        'Test': ['Uncorrected Paired T-Test', 'N_eff Corrected T-Test'],
        'N_Pixels': [len(df_zone), len(df_zone)],
        'N_Effective': [len(df_zone), corr_results['n_eff']],
        'Mean_Diff': [mean_diff, corr_results['mean_diff']],
        'T_Statistic': [t_stat_naive, corr_results['t_stat']],
        'P_Value': [p_val_naive, corr_results['p_value']]
    }
    
    df_stats = pd.DataFrame(data)
    df_stats.to_csv(output_dir / f"stats_report_{scenario_name}.csv", index=False)

# ==============================================================================
# --- MAIN PROCESSOR ---
# ==============================================================================
def plot_diagonal_profile(df_data, output_dir, scenario_name):
    """
    Extracts the diagonal (X1 approx X2) and plots the Entropy Profile.
    This visualizes the 'Signal-to-Noise' ratio perfectly.
    """
    # Filter for points along the diagonal (within 1% tolerance)
    # This creates a slice through the center of the 'L'
    df_diag = df_data[abs(df_data['X1'] - df_data['X2']) < 0.02].copy()
    
    # Sort by position along the diagonal (using X1 as proxy)
    df_diag = df_diag.sort_values(by='X1')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the "Mountain Profiles"
    ax.plot(df_diag['X1'], df_diag['VE_UNREAL'], label='UNREAL', color='gold', linewidth=3, linestyle='-')
    ax.plot(df_diag['X1'], df_diag['VE_RF'], label='QBC-RF', color='green', linewidth=3, linestyle='--')
    
    # Mark the True Boundary location (0.5, 0.5)
    ax.axvline(0.5, color='gray', linestyle=':', label='True Boundary')
    
    ax.set_title(f"Diagonal Cross-Section: {scenario_name}\n(Uncertainty Profile from 0,0 to 1,1)", fontsize=14)
    ax.set_xlabel("Position along Diagonal (0.0 = Bottom Left, 1.0 = Top Right)")
    ax.set_ylabel("Average Vote Entropy")
    ax.set_ylim(0, 1.05) # Keep scale fixed to compare A vs B easily
    ax.legend()
    
    plt.savefig(output_dir / f"diagonal_profile_{scenario_name}.png", bbox_inches='tight')
    plt.close()


def process_scenario(scenario, rf_list, unreal_list):
    print(f"Processing {scenario} ({len(rf_list)} runs)...")
    
    # 1. Determine Output Folder Logic
    if "A_Random" in scenario: sub = "ScenarioA_Corners"
    elif "B_Random" in scenario: sub = "ScenarioB_Boundary"
    elif "C_Margin" in scenario:
        margin = scenario.split('_')[-1]
        sub = f"ScenarioC_Margins/Margin_{margin}"
    else: sub = "Other"
    
    out_dir = PLOT_ROOT / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Averages
    avg_rf = np.mean(rf_list, axis=0)
    avg_unreal = np.mean(unreal_list, axis=0)
    
    # 3. Prepare DataFrame for Stats
    xx, yy = np.meshgrid(np.linspace(0, 1, GRID_RESOLUTION), np.linspace(0, 1, GRID_RESOLUTION))
    df = pd.DataFrame({
        'X1': xx.ravel(), 'X2': yy.ravel(),
        'VE_RF': avg_rf.ravel(), 'VE_UNREAL': avg_unreal.ravel()
    })
    df['Distance'] = df[['X1', 'X2']].apply(lambda x: distance_to_boundary(x, metric=DISTANCE_METRIC), axis=1)
    
    # 4. Generate Outputs
    plot_heatmap(avg_rf, avg_unreal, out_dir, scenario, len(rf_list))
    plot_boundary_line(df, out_dir, scenario)
    plot_diagonal_profile(df, out_dir, scenario)
    
    # 5. Statistical Comparison
    df_zone = df[(df['Distance'] >= -BOUNDARY_ZONE) & (df['Distance'] <= BOUNDARY_ZONE)]
    if not df_zone.empty:
        save_statistical_comparison(df_zone, out_dir, scenario)

def main():
    data = load_and_aggregate()
    if not data: return
    
    for scen, models in data.items():
        process_scenario(scen, models['QBC-RF'], models['UNREAL'])
        
    print(f"\nFull analysis complete. Check folder: {PLOT_ROOT}")

if __name__ == "__main__":
    main()