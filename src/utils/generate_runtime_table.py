import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

### CONFIGURATION ###
# Identify project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "study1_active_learning" / "tree_predictor"
OUTPUT_DIR = PROJECT_ROOT / "results" / "study1_active_learning" / "Tables"
OUTPUT_FILENAME = "RuntimeTable.tex"

# Method mapping for lookup and display
NAME_MAPPING = {
    "M1": "Random",
    "M2": "RF (3)",
    "M3": "RF (Sqrt)",
    "M4": "RF (All)",
    "M5": "UNR (0.05)",
    "M6": "UNR (0.10)",
    "M8": "Uncert.",
    "M9": "Coreset",
    "M10": "R-EMC"
}

# Define the grouping structure for the LaTeX table
COLUMN_GROUPS = [
    {
        "group_name": "Baselines",
        "columns": [("M1", "Random"), ("M8", "Uncert."), ("M9", "Coreset")]
    },
    {
        "group_name": "QBC-RF (Artif. Div.)",
        "columns": [("M2", "F=3"), ("M3", "F=Sqrt"), ("M4", "F=All")]
    },
    {
        "group_name": "Structural (Proposed)",
        "columns": [("M5", "UNR .05"), ("M6", "UNR .10"), ("M10", "R-EMC")]
    }
]

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        return

    dataset_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "aggregated").exists()])
    results = []

    print(f"Generating Runtime Table for {len(dataset_dirs)} datasets...")

    for ds_dir in dataset_dirs:
        ds_name_raw = ds_dir.name
        # Format Dataset Name for LaTeX
        ds_display = ds_name_raw.replace("_", " ").title().replace("Synthetic", "Synth.")
        row = {"Dataset": ds_display}
        
        agg_path = ds_dir / "aggregated"
        
        for group in COLUMN_GROUPS:
            for m_id, _ in group["columns"]:
                pkl_file = agg_path / f"{m_id}_results.pkl"
                if pkl_file.exists():
                    try:
                        with open(pkl_file, "rb") as f:
                            data = pickle.load(f)
                        
                        # Extract the elapsed_time array (seeds,)
                        times = data.get("elapsed_time")
                        if times is not None and len(times) > 0:
                            row[m_id] = np.median(times)
                        else:
                            row[m_id] = np.nan
                    except Exception:
                        row[m_id] = np.nan
                else:
                    row[m_id] = np.nan
        
        results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Global Median Row
    if not df.empty:
        avg_row = df.median(numeric_only=True)
        avg_row["Dataset"] = "\\textbf{MEDIAN}"
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    ### GENERATE LATEX ###
    latex = r"\begin{table*}[htbp]" + "\n"
    latex += r"    \centering" + "\n"
    latex += r"    \scriptsize" + "\n"
    latex += r"    \setlength{\tabcolsep}{4pt}" + "\n"
    
    # Build alignment string
    total_data_cols = sum(len(g["columns"]) for g in COLUMN_GROUPS)
    latex += f"    \\begin{{tabular}}{{l {'r' * total_data_cols}}}" + "\n"
    latex += r"        \toprule" + "\n"
    
    ## HEADER ROW 1: GROUPS ##
    header1 = "        " 
    cmidrules = ""
    curr_idx = 2 
    
    for group in COLUMN_GROUPS:
        n = len(group["columns"])
        header1 += f" & \\multicolumn{{{n}}}{{c}}{{\\textbf{{{group['group_name']}}}}}"
        cmidrules += f" \\cmidrule(lr){{{curr_idx}-{curr_idx + n - 1}}}"
        curr_idx += n
        
    latex += header1 + r" \\" + "\n"
    latex += "        " + cmidrules + "\n"
    
    ## HEADER ROW 2: SUB-HEADERS ##
    header2 = "        \\textbf{Dataset}"
    for group in COLUMN_GROUPS:
        for _, label in group["columns"]:
            header2 += f" & {label}"
    latex += header2 + r" \\ \midrule" + "\n"

    # DATA ROWS
    for i, row in df.iterrows():
        ds_name = row['Dataset']
        latex += f"        {ds_name}"
            
        for group in COLUMN_GROUPS:
            for m_id, _ in group["columns"]:
                val = row.get(m_id)
                if pd.isna(val):
                    latex += " & ---"
                else:
                    # Use bolding for the median summary row
                    if "MEDIAN" in ds_name:
                        latex += f" & \\textbf{{{val:.1f}}}"
                    else:
                        latex += f" & {val:.1f}"
        
        latex += r" \\" + "\n"
        if i == len(df) - 2: # Before median row
             latex += r"        \midrule" + "\n"

    latex += r"        \bottomrule" + "\n"
    latex += r"    \end{tabular}" + "\n"
    latex += r"    \caption{Median experiment runtime (seconds) across 25 simulation seeds. UNR (Proposed) denotes the UNREAL algorithm.}" + "\n"
    latex += r"    \label{tab:RuntimeComparison}" + "\n"
    latex += r"\end{table*}" + "\n"

    # Save to File
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    with open(output_path, "w") as f:
        f.write(latex)
        
    print(f"\nLaTeX Runtime Table saved to: {output_path}")

if __name__ == "__main__":
    main()