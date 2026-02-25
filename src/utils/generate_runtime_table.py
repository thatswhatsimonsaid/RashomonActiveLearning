### Libraries ###
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

### CONFIGURATION ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "study1_active_learning" / "tree_predictor"
OUTPUT_DIR = PROJECT_ROOT / "results" / "study1_active_learning" / "Tables"
OUTPUT_FILENAME = "RuntimeTable.tex"

NAME_MAPPING = {
    "M1": "Random",
    # "M2": "QBC-RF (p=3)",
    "M3": "QBC-RF (p=sqrt)",
    "M4": "QBC-RF (p=d)",
    "M5": "UNREAL",
    "M6": "Uncertainty",
    "M7": "Coreset",
    "M8": "BREAL",
    "M9": "QBC-RF (Weighted, p=sqrt)",
    "M10": "QBC-RF (Weighted, p=d)",
}

COLUMN_GROUPS = [
    {"group_name": "Baselines", "columns": [("M1", "Random"), ("M6", "Uncertainty"), ("M7", "Coreset")]},
    {"group_name": "QBC-RF","columns": [("M2", "F=3"), ("M3", "F=Sqrt"), ("M4", "F=All")]},
    {"group_name": "Weighted RF","columns": [("M9", "F=Sqrt"), ("M10", "F=All")]},
    {"group_name": "REAL (Proposed)","columns": [("M5", "UNREAL"), ("M8", "BREAL")]}
]

### DATASET ORDER ###
DATASET_ORDER = [
    "Synthetic_XOR_Baseline",
    "Synthetic_XOR_Phi_05",
    "Synthetic_XOR_Phi_10",
    "Synthetic_XOR_Phi_25",
    "Synthetic_XOR_Phi_45",
    "Synthetic_XOR_Alpha_50",
    "Synthetic_XOR_Alpha_25",
    "Synthetic_XOR_Alpha_75",
    "Synthetic_XOR_Alpha_100",
    "anneal",                               # Dataset 1
    "bank_marketing",                       # Dataset 2
    "bar-7",                                # Dataset 3
    "breast_cancer_wisconsin",              # Dataset 4
    "car_evaluation",                       # Dataset 5
    "cheap_restaurant",                     # Dataset 6
    "coffee_house",                         # Dataset 7
    "compas",
    "expensive_restaurant",                 # Dataset 8
    # "fico"
    "haberman",                             # Dataset 9
    "hepatitis",                            # Dataset 10
    "hypothyroid",                          # Dataset 11
    "kr-vs-kp",                             # Dataset 12
    "lymph",                                # Dataset 13
    # "monk1",                                # Dataset 14 (replace with FICO [or bank_note/spect] if finish on time)
    "monk2",                                # Dataset 15 
    # "monk3",                                # Dataset 16 (replace with COMPAS [or bank_note/spect] if finish on time)
    "primary-tumor",                        # Dataset 17
    "spect",
    "tic-tac-toe",                          # Dataset 18
    "vote",                                 # Dataset 19
    "yeast"                                 # Dataset 20
]

SYNTHETIC_DATASETS = {
    "Synthetic_XOR_Baseline",
    "Synthetic_XOR_Phi_05",
    "Synthetic_XOR_Phi_10",
    "Synthetic_XOR_Phi_25",
    "Synthetic_XOR_Phi_45",
    "Synthetic_XOR_Alpha_50",
    "Synthetic_XOR_Alpha_25",
    "Synthetic_XOR_Alpha_75",
    "Synthetic_XOR_Alpha_100",
}

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        return

    # Build a lookup of available dataset dirs
    available_dirs = {
        d.name: d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "aggregated").exists()
    }

    # Filter and order according to DATASET_ORDER, skip missing
    dataset_dirs = []
    for ds_name in DATASET_ORDER:
        if ds_name in available_dirs:
            dataset_dirs.append(available_dirs[ds_name])
        else:
            print(f"  [WARN] Dataset not found, skipping: {ds_name}")

    results = []
    print(f"Generating Runtime Table for {len(dataset_dirs)} datasets...")

    for ds_dir in dataset_dirs:
        ds_name_raw = ds_dir.name
        ds_display = ds_name_raw.replace("_", " ").title().replace("Synthetic", "Synth.")
        row = {"Dataset": ds_display, "_raw_name": ds_name_raw}
        agg_path = ds_dir / "aggregated"

        for group in COLUMN_GROUPS:
            for m_id, _ in group["columns"]:
                pkl_file = agg_path / f"{m_id}_results.pkl"
                if pkl_file.exists():
                    try:
                        with open(pkl_file, "rb") as f:
                            data = pickle.load(f)
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

    # Separate synthetic and real for median rows
    synth_mask = df["_raw_name"].isin(SYNTHETIC_DATASETS)
    real_mask  = ~synth_mask

    if not df.empty:
        # Synthetic median
        synth_median = df[synth_mask].median(numeric_only=True)
        synth_median["Dataset"] = "\\textbf{MEDIAN (Synth.)}"
        synth_median["_raw_name"] = "_synth_median"

        # Real-world median
        real_median = df[real_mask].median(numeric_only=True)
        real_median["Dataset"] = "\\textbf{MEDIAN (Real)}"
        real_median["_raw_name"] = "_real_median"

        # Insert synthetic median after last synthetic row, real median at end
        synth_df   = pd.concat([df[synth_mask], pd.DataFrame([synth_median])], ignore_index=True)
        real_df    = pd.concat([df[real_mask],  pd.DataFrame([real_median])],  ignore_index=True)
        df = pd.concat([synth_df, real_df], ignore_index=True)

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
    last_synth_idx = None
    for i, row in df.iterrows():
        raw = row.get("_raw_name", "")
        if raw in SYNTHETIC_DATASETS:
            last_synth_idx = i

    in_real_section = False

    for i, row in df.iterrows():
        raw = row.get("_raw_name", "")
        ds_name = row["Dataset"]
        is_median = "MEDIAN" in str(ds_name)

        if not in_real_section and raw not in SYNTHETIC_DATASETS and raw != "_synth_median":
            in_real_section = True
            latex += r"        \midrule" + "\n"

        if raw == "_synth_median":
            latex += r"        \midrule" + "\n"

        latex += f"        {ds_name}"

        for group in COLUMN_GROUPS:
            for m_id, _ in group["columns"]:
                val = row.get(m_id)
                if pd.isna(val):
                    latex += " & ---"
                else:
                    if is_median:
                        latex += f" & \\textbf{{{val:.1f}}}"
                    else:
                        latex += f" & {val:.1f}"

        latex += r" \\" + "\n"

        if raw == "_real_median" and i < len(df) - 1:
            pass  
        elif in_real_section and i == len(df) - 2:
            latex += r"        \midrule" + "\n"

    latex += r"        \bottomrule" + "\n"
    latex += r"    \end{tabular}" + "\n"
    latex += (
        r"    \caption{Median experiment runtime (seconds) across 25 simulation seeds. "
        r"UNR and UNR-B denote the standard and Bayesian variants of the UNREAL algorithm, respectively.}"
        + "\n"
    )
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