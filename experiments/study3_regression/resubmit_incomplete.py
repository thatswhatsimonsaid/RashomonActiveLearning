"""Resubmit unfinished active-learning regression seeds under the right charge.

Out-of-memory seeds go to largemem at 60 GB. On that partition the
scheduler bills one CPU. Timed-out seeds that already labeled at least
a quarter of the pool go to medium at 30 GB, which this cluster bills
as two CPUs. Seeds that never left the first part of the pool are left
alone: another day does not make a stuck PySORTD fit return.

Run on the cluster, from the repository root, with the project Python:

    .RAL_CL/bin/python experiments/study3_regression/resubmit_incomplete.py
    .RAL_CL/bin/python experiments/study3_regression/resubmit_incomplete.py --submit
"""

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.method_complete import expected_curve_length
from src.utils.retry_plan import (
    ACCOUNT,
    LARGEMEM_MEMORY,
    MEDIUM_MEMORY,
    RETRY_TIME,
    classify_seed,
    rewrite_sbatch,
)

STUDY_DIR = "study3_regression/regression_predictor"
N_REPLICATIONS = 25
DATASETS = [
    "concrete-slump",
    "concrete-flow",
    "concrete-cs",
    "autompg",
    "no2",
    "pm10",
    "cps",
    "concrete",
]
METHODS = [f"M{index}" for index in range(1, 14)]
DATASET_ROOT = SCRIPT_DIR / "job_scripts" / "regression_predictor" / "datasets"
RETRY_ROOT = SCRIPT_DIR / "job_scripts" / "regression_predictor" / "retries"
RESULTS_ROOT = PROJECT_ROOT / "results" / STUDY_DIR
MEMORY = {"largemem": LARGEMEM_MEMORY, "medium": MEDIUM_MEMORY}


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def latest_states() -> dict[tuple[str, str, int], str]:
    """Most recent SLURM state for each dataset, method, and seed."""
    raw = _run(
        [
            "sacct",
            "-u",
            "simondn",
            "-S",
            "2026-10-13",
            "-n",
            "-X",
            "-P",
            "-o",
            "JobID,JobName,State",
        ]
    )
    states: dict[tuple[str, str, int], tuple[int, str]] = {}
    allowed = set(DATASETS)
    for line in raw.splitlines():
        job_id, name, state = line.split("|")
        if "_" not in job_id:
            continue
        parent, seed_text = job_id.rsplit("_", 1)
        if not seed_text.isdigit() or "_M" not in name:
            continue
        dataset, method = name.removesuffix("_retry").rsplit("_M", 1)
        method = f"M{method}"
        if dataset not in allowed or method not in METHODS:
            continue
        seed = int(seed_text)
        job_number = int(parent)
        previous = states.get((dataset, method, seed))
        if previous is None or job_number >= previous[0]:
            states[(dataset, method, seed)] = (job_number, state.split()[0])
    return {key: value[1] for key, value in states.items()}


def running_seeds() -> set[tuple[str, str, int]]:
    raw = _run(["squeue", "-u", "simondn", "-h", "-r", "-o", "%j|%a"])
    running: set[tuple[str, str, int]] = set()
    allowed = set(DATASETS)
    for line in raw.splitlines():
        name, array_id = line.split("|", 1)
        if not array_id.isdigit() or "_M" not in name:
            continue
        dataset, method = name.removesuffix("_retry").rsplit("_M", 1)
        method = f"M{method}"
        if dataset in allowed and method in METHODS:
            running.add((dataset, method, int(array_id)))
    return running


def curve_length(dataset: str, method: str, seed: int) -> int:
    path = RESULTS_ROOT / dataset / method / f"{method}_S{seed}.pkl"
    if not path.exists():
        return 0
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    history = getattr(result, "rmse_history", None)
    return len(history) if history is not None else 0


def build_plan() -> tuple[dict[tuple[str, str], list[int]], dict[str, int]]:
    states = latest_states()
    running = running_seeds()
    expected = {
        dataset: expected_curve_length(dataset, PROJECT_ROOT / "src" / "data")
        for dataset in DATASETS
    }
    grouped: dict[tuple[str, str], list[int]] = {}
    counts = {"largemem": 0, "medium": 0, "running": 0, "full": 0, "left": 0}
    for dataset in DATASETS:
        for method in METHODS:
            for seed in range(N_REPLICATIONS):
                key = (dataset, method, seed)
                length = curve_length(dataset, method, seed)
                if length >= expected[dataset]:
                    counts["full"] += 1
                    continue
                if key in running:
                    counts["running"] += 1
                    continue
                partition = classify_seed(
                    states.get(key, ""),
                    length,
                    expected[dataset],
                    running=False,
                )
                if partition is None:
                    counts["left"] += 1
                    continue
                grouped.setdefault((dataset, method, partition), [])
                grouped[(dataset, method, partition)].append(seed)
                counts[partition] += 1
    return grouped, counts


def write_and_maybe_submit(grouped: dict, submit: bool) -> None:
    RETRY_ROOT.mkdir(parents=True, exist_ok=True)
    for (dataset, method, partition), seeds in sorted(grouped.items()):
        source = DATASET_ROOT / dataset / f"submit_{dataset}_{method}.sbatch"
        text = source.read_text()
        rewritten = rewrite_sbatch(
            text,
            partition=partition,
            mem=MEMORY[partition],
            time_limit=RETRY_TIME,
            seeds=seeds,
            job_name=f"{dataset}_{method}_retry",
            account=ACCOUNT,
        )
        destination = RETRY_ROOT / f"submit_{dataset}_{method}_{partition}.sbatch"
        destination.write_text(rewritten)
        destination.chmod(0o755)
        print(
            f"{partition:9} {RETRY_TIME} {MEMORY[partition]:>4}  "
            f"{dataset} {method} seeds {','.join(str(seed) for seed in seeds)}"
        )
        if submit:
            submitted = _run(["sbatch", str(destination)]).strip()
            print(f"  {submitted}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="submit the sbatch files")
    args = parser.parse_args()
    grouped, counts = build_plan()
    print(
        "full {full} | still running {running} | largemem {largemem} | "
        "medium {medium} | left unfinished {left}".format(**counts)
    )
    print(
        f"account {ACCOUNT}; largemem bills 1 CPU at {LARGEMEM_MEMORY}; "
        f"medium bills 2 CPUs at {MEDIUM_MEMORY}"
    )
    write_and_maybe_submit(grouped, args.submit)
    if not args.submit:
        print("Dry run only. Pass --submit to launch these jobs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
