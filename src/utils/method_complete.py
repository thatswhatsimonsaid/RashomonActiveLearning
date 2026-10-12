"""Report whether one method already has a finished curve for every seed."""

import argparse
import pickle
import sys
from pathlib import Path

from src.utils.data_handler import load_data, split_test_pool


def expected_curve_length(dataset: str, data_dir: Path) -> int:
    """Number of RMSE points when the candidate pool is labeled to the end."""
    frame = load_data(dataset, data_dir)
    _, pool = split_test_pool(frame, test_proportion=0.2, random_state=0)
    initial = 20 if 20 < len(pool) else len(pool) // 2
    return len(pool) - initial + 1


def method_is_complete(method_dir: Path, method_name: str, n_replications: int, expected: int) -> bool:
    if not method_dir.is_dir():
        return False
    for seed in range(n_replications):
        path = method_dir / f"{method_name}_S{seed}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as handle:
            result = pickle.load(handle)
        history = getattr(result, "rmse_history", None)
        if history is None or len(history) != expected:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method_number", required=True, type=int)
    parser.add_argument("--study_dir", required=True)
    parser.add_argument("--n_replications", type=int, default=25)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    method_name = f"M{args.method_number}"
    method_dir = project_root / "results" / args.study_dir / args.dataset / method_name
    expected = expected_curve_length(args.dataset, project_root / "src" / "data")
    complete = method_is_complete(method_dir, method_name, args.n_replications, expected)
    print(
        f"{args.dataset} {method_name}: "
        f"{'complete' if complete else 'incomplete'} (expected {expected} points)"
    )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
