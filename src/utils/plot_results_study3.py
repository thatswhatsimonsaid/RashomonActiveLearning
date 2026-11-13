### Summary ###
"""
Trace / diagnostic plots for Study 3 (pattern disambiguation, synthetic XOR).

Method numbering differs from Study 1 — do NOT reuse plot_results.py labels.
"""

import argparse
import os
import pickle
from pathlib import Path

# Headless cluster / cloud VMs: never try a GUI backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONT_SIZE = 16

# Study 3 method legend (see experiments/study3_pattern_diversity/master_config.py)
METHODS_TO_PLOT = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
METHOD_LABELS = {
    "M1": "Random",
    "M2": "RF-QBC (√d)",
    "M3": "UNREAL (vote entropy)",
    "M4": "UNREAL (unique patterns)",
    "M5": "Pattern Δ (count)",
    "M6": "Pattern Δ (effective)",
    "M7": "Pattern Δ (Gibbs)",
}
METHOD_COLORS = {
    "M1": "#808080",
    "M2": "#ff7f0e",
    "M3": "#1f77b4",
    "M4": "#17becf",
    "M5": "#d62728",
    "M6": "#e377c2",
    "M7": "#2ca02c",
}
METHOD_STYLES = {
    "M1": "--",
    "M2": ":",
    "M3": "-",
    "M4": "-.",
    "M5": "-",
    "M6": "-",
    "M7": "-.",
}

# Rashomon / ECS diagnostics: PySORTD-backed selectors only
RASHOMON_METHODS = {"M3", "M4", "M5", "M6", "M7"}

METRICS_TO_PLOT = [
    ("accuracy_history", "Accuracy", "lower right"),
    ("f1_history", "F1 Score", "lower right"),
    ("tree_edit_distance_history", "Tree Edit Distance (TED)", "upper right"),
    ("rashomon_size_history", "Rashomon Set Size", "upper right"),
    ("committee_size_history", r"Effective Committee Size ($\exp(H)$)", "upper right"),
    ("oracle_agreement_history", "Oracle Agreement", "lower right"),
]

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
        "legend.fontsize": FONT_SIZE - 2,
        "figure.titlesize": FONT_SIZE + 2,
    }
)


def load_aggregated_data(aggregated_dir: Path):
    data = {}
    for pkl_file in aggregated_dir.glob("M*_results.pkl"):
        method_key = pkl_file.stem.split("_")[0]
        with open(pkl_file, "rb") as f:
            data[method_key] = pickle.load(f)
    return data


def plot_metric(data, metric_key, y_label, save_path, dataset_name, show_legend=True):
    plt.figure(figsize=(5, 5))
    has_data = False

    for method in METHODS_TO_PLOT:
        if method not in data:
            continue
        if metric_key in ("rashomon_size_history", "committee_size_history"):
            if method not in RASHOMON_METHODS:
                continue

        history = data[method].get(metric_key)
        if history is None or (isinstance(history, np.ndarray) and history.size == 0):
            continue
        has_data = True

        means = np.nanmean(history, axis=0)
        if metric_key == "tree_edit_distance_history" and np.all(means == -1.0):
            continue
        stds = np.nanstd(history, axis=0) / np.sqrt(history.shape[0])
        iterations = np.arange(len(means)) + 1

        plt.plot(
            iterations,
            means,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "black"),
            linestyle=METHOD_STYLES.get(method, "-"),
            linewidth=1.5,
        )
        plt.fill_between(
            iterations,
            means - stds,
            means + stds,
            color=METHOD_COLORS.get(method, "black"),
            alpha=0.1,
        )

    if not has_data:
        print(f"  -> Skipping {metric_key} (No data found)")
        plt.close()
        return

    plt.xlabel("Active Learning Iterations")
    plt.ylabel(y_label)
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {save_path.name}")


def plot_variance_metric(data, metric_key, y_label, save_path, dataset_name, show_legend=True):
    plt.figure(figsize=(5, 5))
    has_data = False
    for method in METHODS_TO_PLOT:
        if method not in data:
            continue
        history = data[method].get(metric_key)
        if history is None or (isinstance(history, np.ndarray) and history.size == 0):
            continue
        history = np.array(history)
        if history.ndim == 1:
            history = history.reshape(1, -1)
        has_data = True
        variances = np.nanvar(history, axis=0)
        iterations = np.arange(len(variances)) + 1
        plt.plot(
            iterations,
            variances,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "black"),
            linestyle=METHOD_STYLES.get(method, "-"),
            linewidth=2,
        )

    if not has_data:
        plt.close()
        return

    plt.xlabel("Active Learning Iterations")
    plt.ylabel(f"Variance of {y_label}")
    plt.title(f"{dataset_name}: {y_label} Stability (Variance)")
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Variance Plot: {save_path.name}")


def generate_legend(output_path: Path, ncol=4):
    handles, labels = [], []
    for method in METHODS_TO_PLOT:
        line = plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=2.5,
        )
        handles.append(line)
        labels.append(METHOD_LABELS[method])
    fig = plt.figure(figsize=(14, 1.6))
    fig_legend = fig.legend(
        handles, labels, loc="center", frameon=True, ncol=ncol, fontsize=11, handlelength=3
    )
    plt.gca().axis("off")
    fig.savefig(
        output_path,
        bbox_inches=fig_legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
        dpi=300,
        transparent=True,
    )
    plt.close(fig)
    print(f"Legend saved to: {output_path}")


def plot_dataset(project_root: Path, study_dir: str, dataset: str, show_legend: bool):
    dataset_dir = project_root / "results" / study_dir / dataset
    # Support both layouts: .../study_dir/dataset and .../study_dir/tree_predictor/dataset
    if not (dataset_dir / "aggregated").exists():
        alt = project_root / "results" / study_dir / "tree_predictor" / dataset
        if (alt / "aggregated").exists():
            dataset_dir = alt
    agg_dir = dataset_dir / "aggregated"
    img_dir = dataset_dir / "accuracy_images"
    if not agg_dir.exists():
        print(f"[ERROR] Aggregated results not found at: {agg_dir}")
        return False
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Plotting Study 3 results for {dataset} ---")
    data = load_aggregated_data(agg_dir)
    if not data:
        print("No data found to plot.")
        return False
    for metric_key, y_label, _ in METRICS_TO_PLOT:
        plot_metric(
            data,
            metric_key,
            y_label,
            img_dir / f"{metric_key}.png",
            dataset,
            show_legend=show_legend,
        )
    plot_variance_metric(
        data,
        "accuracy_history",
        "Accuracy",
        img_dir / "accuracy_variance.png",
        dataset,
        show_legend=show_legend,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument(
        "--study_dir",
        type=str,
        default="study3_pattern_diversity/tree_predictor",
    )
    parser.add_argument("--no-legend", dest="show_legend", action="store_false")
    parser.add_argument("--legend-only", action="store_true")
    parser.add_argument(
        "--all-complete",
        action="store_true",
        help="Plot every dataset under study_dir that has an aggregated/ folder.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    plots_root = project_root / "results" / "study3_pattern_diversity" / "PLOTS"
    plots_root.mkdir(parents=True, exist_ok=True)

    if args.legend_only:
        generate_legend(plots_root / "study3_legend.png")
        return

    if args.all_complete:
        base = project_root / "results" / args.study_dir
        # also allow study_dir without tree_predictor suffix
        candidates = []
        for root in {base, project_root / "results" / "study3_pattern_diversity" / "tree_predictor"}:
            if root.exists():
                for d in sorted(root.iterdir()):
                    if d.is_dir() and (d / "aggregated").exists():
                        candidates.append(d.name)
        datasets = sorted(set(candidates))
        print(f"Plotting {len(datasets)} aggregated datasets: {datasets}")
        ok = 0
        for ds in datasets:
            if plot_dataset(project_root, args.study_dir, ds, args.show_legend):
                ok += 1
        generate_legend(plots_root / "study3_legend.png")
        print(f"Done. Plotted {ok}/{len(datasets)} datasets.")
        return

    if not args.dataset:
        parser.error("--dataset is required unless using --legend-only or --all-complete")
    plot_dataset(project_root, args.study_dir, args.dataset, args.show_legend)
    generate_legend(plots_root / "study3_legend.png")


if __name__ == "__main__":
    main()
