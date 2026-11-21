"""Write embeddings and call the unmodified coMDS R package."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


R_SCRIPT = Path(__file__).resolve().parents[1] / "r" / "fit_consensus.R"


def _write_payload(
    path: Path,
    embeddings: list[np.ndarray],
    weights: np.ndarray | None,
    weightmat: np.ndarray | None,
) -> None:
    arrays = [np.asarray(z, dtype=np.float64) for z in embeddings]
    n, d = arrays[0].shape
    m = len(arrays)
    if any(z.shape != (n, d) for z in arrays):
        raise ValueError("All embeddings must share the same shape")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(np.asarray([n, d, m, 0 if weights is None else 1], dtype=np.int32).tobytes())
        for embedding in arrays:
            handle.write(np.asfortranarray(embedding).tobytes(order="F"))
        if weights is not None:
            handle.write(np.asarray(weights, dtype=np.float64).tobytes())
        has_wm = 1 if weightmat is not None else 0
        handle.write(np.asarray([has_wm], dtype=np.int32).tobytes())
        if weightmat is not None:
            if weightmat.shape != (n, n):
                raise ValueError("weightmat must be n by n")
            # The same pair-weight matrix is applied to every view.
            block = np.asfortranarray(weightmat.astype(np.float64)).tobytes(order="F")
            for _ in range(m):
                handle.write(block)


def _read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        if key in {"niter", "nobj"}:
            meta[key] = int(float(value))
        elif key == "model":
            meta[key] = value
        else:
            meta[key] = float(value)
    return meta


def fit_consensus(
    embeddings: list[np.ndarray],
    out_dir: Path,
    kind: str = "comds",
    weights: np.ndarray | None = None,
    weightmat: np.ndarray | None = None,
    tau: float = 0.1,
    percentile: float = 0.5,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = out_dir / "payload.bin"
    _write_payload(payload, embeddings, weights, weightmat)
    command = ["Rscript", str(R_SCRIPT), str(payload), str(out_dir), kind]
    if kind == "locomds":
        command.extend([str(tau), str(percentile)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    log_path = out_dir / "rscript.log"
    log_path.write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"coMDS failed for {out_dir.name} (exit {completed.returncode}). See {log_path}"
        )
    gspace = pd.read_csv(out_dir / "gspace.csv", header=None).to_numpy(dtype=float)
    meta = _read_meta(out_dir / "meta.txt")
    meta["embedding"] = gspace
    meta["log"] = completed.stdout.strip()
    return meta
