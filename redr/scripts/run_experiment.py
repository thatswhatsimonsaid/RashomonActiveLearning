#!/usr/bin/env python3
"""Train method-specific Rashomon candidates and fit CoMDS / LoCoMDS.

Example:
  PYTHONPATH=redr python redr/scripts/run_experiment.py --datasets iris
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*pin_memory.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redr.common_knowledge import candidate_edges, edges_to_graph, mean_score_edges
from redr.comds_bridge import fit_consensus
from redr.data import load_dataset
from redr.metrics import _sample_triplets, evaluate_embedding
from redr.plotting import plot_gallery, plot_metric_bars
from redr.probe import contrastive_probe, pacmap_probe
from redr.rashomon import (
    edge_weight_matrix,
    gibbs_weights,
    mark_rashomon,
    mean_vote_entropy,
    method_balanced_weights,
    uniform_weights,
    vote_weights,
)
from redr.train import fit_cne, fit_pacmap, high_d_neighbors, pacmap_pairs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["iris"])
    parser.add_argument("--methods", nargs="+", default=["umap", "infonce", "pacmap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 5.0])
    parser.add_argument("--cne-epochs", type=int, default=100)
    parser.add_argument("--pacmap-epochs", type=int, default=450)
    parser.add_argument("--undertrained-epochs", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--k-eval", type=int, default=10)
    parser.add_argument("--loco-tau", type=float, default=0.1)
    parser.add_argument("--loco-percentile", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-common-knowledge", action="store_true")
    return parser.parse_args()


def epochs_for(method: str, args, undertrained: bool) -> int:
    if undertrained:
        return args.undertrained_epochs
    if method == "pacmap":
        return args.pacmap_epochs
    return args.cne_epochs


def train_candidate(method, data, seed, lam, n_epochs, pairs, neighbor_index):
    if method == "pacmap":
        embedding, seconds = fit_pacmap(
            data["X"], data["X_pca"], seed, lam, n_epochs, pairs
        )
        probe = pacmap_probe(
            embedding, pairs[0], pairs[1], pairs[2], data["n"]
        )
    else:
        embedding, seconds = fit_cne(
            data["X"], data["X_pca"], method, seed, lam, n_epochs
        )
        probe = contrastive_probe(embedding, neighbor_index, method)
    return embedding, seconds, probe


def matched_unrestricted(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen = []
    for _method, group in frame.groupby("method"):
        k = int(group["in_rashomon"].sum())
        if k == 0:
            continue
        take = rng.choice(group.index.to_numpy(), size=k, replace=False)
        chosen.extend(take.tolist())
    return frame.loc[chosen]


def evaluate_row(data, embedding, triplets, k):
    return evaluate_embedding(
        data["X"],
        embedding,
        data["labels"] if data["discrete_labels"] else None,
        k=k,
        triplets=triplets,
    )


def run_dataset(name: str, args) -> pd.DataFrame:
    data = load_dataset(name)
    out_dir = args.out / name
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name}: n={data['n']} p={data['p']} ===", flush=True)

    neighbor_index = high_d_neighbors(data["X"], k=15)
    pairs = pacmap_pairs(data["X"], seed=0) if "pacmap" in args.methods else None
    triplets = _sample_triplets(data["n"], 4000, np.random.default_rng(0))
    previous_seconds = {}
    previous_csv = out_dir / "metrics.csv"
    if previous_csv.exists():
        previous = pd.read_csv(previous_csv)
        for row in previous.itertuples(index=False):
            if row.kind == "candidate" and np.isfinite(row.train_seconds):
                previous_seconds[row.embedding_id] = float(row.train_seconds)

    records = []
    embeddings = {}
    specs = []
    for method in args.methods:
        for seed in args.seeds:
            for lam in args.lambdas:
                specs.append((method, seed, lam, epochs_for(method, args, False), False))
        specs.append((method, 0, 0.0, epochs_for(method, args, True), True))

    for method, seed, lam, n_epochs, undertrained in specs:
        embedding_id = f"{method}_s{seed}_l{lam:g}_e{n_epochs}"
        if undertrained:
            embedding_id += "_undertrained"
        cached = emb_dir / f"{embedding_id}.npy"
        reused = False
        if cached.exists():
            cached_embedding = np.load(cached)
            if (
                cached_embedding.shape == (data["n"], 2)
                and np.isfinite(cached_embedding).all()
            ):
                reused = True
                embedding = cached_embedding
        print(f"  {'reuse' if reused else 'train'} {embedding_id}", flush=True)
        try:
            if reused:
                seconds = previous_seconds.get(embedding_id, np.nan)
                if method == "pacmap":
                    probe = pacmap_probe(embedding, pairs[0], pairs[1], pairs[2], data["n"])
                else:
                    probe = contrastive_probe(embedding, neighbor_index, method)
            else:
                embedding, seconds, probe = train_candidate(
                    method, data, seed, lam, n_epochs, pairs, neighbor_index
                )
            error = ""
        except Exception as exc:
            traceback.print_exc()
            embedding = np.full((data["n"], 2), np.nan)
            seconds = np.nan
            probe = np.nan
            error = f"{type(exc).__name__}: {exc}"
        np.save(emb_dir / f"{embedding_id}.npy", embedding)
        embeddings[embedding_id] = embedding
        records.append(
            {
                "dataset": name,
                "embedding_id": embedding_id,
                "kind": "candidate",
                "method": method,
                "seed": seed,
                "lambda_pca": lam,
                "n_epochs": n_epochs,
                "undertrained": undertrained,
                "probe_loss": probe,
                "train_seconds": seconds,
                "error": error,
                "consensus_stress": np.nan,
                "consensus_seconds": np.nan,
                "n_views": 1,
                "source_weight": np.nan,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame = mark_rashomon(frame, args.delta)
    print(
        frame.groupby("method")["in_rashomon"].agg(["sum", "count"]).to_string(),
        flush=True,
    )

    metric_rows = []
    for row in frame.itertuples(index=False):
        if not np.isfinite(embeddings[row.embedding_id]).all():
            continue
        scores = evaluate_row(data, embeddings[row.embedding_id], triplets, args.k_eval)
        metric_rows.append({"embedding_id": row.embedding_id, **scores})
    metrics = pd.DataFrame(metric_rows)
    frame = frame.merge(metrics, on="embedding_id", how="left")

    rashomon = frame[frame["in_rashomon"] & frame["error"].eq("")].copy()
    ensembles = {
        "conventional_best": _best_per_method(frame),
        "unrestricted_all": frame[frame["error"].eq("")].copy(),
        "unrestricted_matched": matched_unrestricted(frame[frame["error"].eq("")]),
        "rashomon_uniform": rashomon,
        "rashomon_method_balanced": rashomon,
        "rashomon_gibbs": rashomon,
        "rashomon_vote": rashomon,
        "rashomon_edge_weightmat": rashomon,
    }

    consensus_rows = []
    gallery = []
    reference = data["X_pca"]
    # Show the best candidate of each method in the gallery.
    for method, group in frame[frame["error"].eq("")].groupby("method"):
        best = group.sort_values("probe_loss").iloc[0]
        gallery.append((f"{method} best", embeddings[best.embedding_id]))

    vote_entropy = np.nan
    if len(rashomon) >= 2:
        vote_entropy = mean_vote_entropy(
            [embeddings[i] for i in rashomon["embedding_id"]], k=args.k_eval
        )

    for ensemble_name, members in ensembles.items():
        members = members[members["error"].eq("")].drop_duplicates("embedding_id")
        if len(members) < 2:
            print(f"  skip {ensemble_name}: only {len(members)} view(s)", flush=True)
            continue
        views = [embeddings[i] for i in members["embedding_id"]]
        if ensemble_name == "rashomon_method_balanced":
            weights = method_balanced_weights(members)
        elif ensemble_name == "rashomon_gibbs":
            weights = gibbs_weights(members)
        elif ensemble_name == "rashomon_vote":
            weights = vote_weights(members, views, k=args.k_eval)
        else:
            weights = uniform_weights(members)
        weightmat = None
        if ensemble_name == "rashomon_edge_weightmat":
            # Floor keeps every pair in the stress while upweighting stable edges.
            weightmat = edge_weight_matrix(views, k=args.k_eval) + 1e-3
            np.fill_diagonal(weightmat, 0.0)
        for kind in ("comds", "locomds"):
            if kind == "locomds" and weightmat is not None:
                # LoCoMDS builds its own local weight matrices and does not take weightmat.
                continue
            run_name = f"{ensemble_name}_{kind}"
            print(f"  consensus {run_name} views={len(views)}", flush=True)
            run_dir = out_dir / "consensus" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "embedding_id": members["embedding_id"].to_numpy(),
                    "method": members["method"].to_numpy(),
                    "weight": weights,
                }
            ).to_csv(run_dir / "source_weights.csv", index=False)
            try:
                fit = fit_consensus(
                    views,
                    run_dir,
                    kind=kind,
                    weights=weights,
                    weightmat=weightmat if kind == "comds" else None,
                    tau=args.loco_tau,
                    percentile=args.loco_percentile,
                )
                # Uniform runs still pass explicit weights of 1 so the call path is tested.
                embedding = fit["embedding"]
                error = ""
                stress = fit["stress"]
                seconds = fit["seconds"]
            except Exception as exc:
                traceback.print_exc()
                embedding = np.full((data["n"], 2), np.nan)
                error = f"{type(exc).__name__}: {exc}"
                stress = np.nan
                seconds = np.nan
            np.save(emb_dir / f"{run_name}.npy", embedding)
            embeddings[run_name] = embedding
            scores = {}
            if np.isfinite(embedding).all():
                scores = evaluate_row(data, embedding, triplets, args.k_eval)
                gallery.append((run_name, embedding))
            consensus_rows.append(
                {
                    "dataset": name,
                    "embedding_id": run_name,
                    "kind": "consensus",
                    "method": ensemble_name,
                    "seed": np.nan,
                    "lambda_pca": np.nan,
                    "n_epochs": np.nan,
                    "undertrained": False,
                    "probe_loss": np.nan,
                    "train_seconds": np.nan,
                    "error": error,
                    "consensus_stress": stress,
                    "consensus_seconds": seconds,
                    "n_views": len(views),
                    "source_weight": np.nan,
                    "in_rashomon": np.nan,
                    "loss_reference": np.nan,
                    "loss_tolerance": np.nan,
                    "vote_entropy": vote_entropy,
                    **scores,
                }
            )

    if not args.skip_common_knowledge:
        ck_row = _common_knowledge(data, frame, embeddings, triplets, args)
        if ck_row is not None:
            consensus_rows.append(ck_row[0])
            gallery.append(("common_knowledge_umap", ck_row[1]))
            embeddings["common_knowledge_umap"] = ck_row[1]

    consensus = pd.DataFrame(consensus_rows)
    combined = pd.concat([frame, consensus], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "metrics.csv", index=False)
    id_list = list(embeddings)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        ids=np.asarray(id_list),
        embeddings=np.stack([embeddings[i] for i in id_list]),
    )

    plot_gallery(
        out_dir / "embeddings.png",
        gallery,
        data["color"],
        data["discrete_labels"],
        reference,
    )
    plot_metric_bars(
        out_dir / "consensus_metrics.png",
        combined,
        ["trustworthiness", "continuity", "lcmc", "triplet_accuracy"],
    )
    manifest = {
        "dataset": name,
        "n": data["n"],
        "p": data["p"],
        "delta": args.delta,
        "k_eval": args.k_eval,
        "cne_epochs": args.cne_epochs,
        "pacmap_epochs": args.pacmap_epochs,
        "seeds": args.seeds,
        "lambdas": args.lambdas,
        "methods": args.methods,
        "loco_tau": args.loco_tau,
        "loco_percentile": args.loco_percentile,
        "vote_entropy": vote_entropy,
        "note": (
            "Rashomon membership is a relative probe-loss threshold on the "
            "trained candidates. It is not an exhaustive Rashomon set."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {out_dir / 'metrics.csv'}", flush=True)
    return combined


def _best_per_method(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[frame["error"].eq("") & np.isfinite(frame["probe_loss"])]
    rows = []
    for _method, group in ok.groupby("method"):
        rows.append(group.sort_values("probe_loss").iloc[0])
    if not rows:
        return frame.iloc[0:0]
    return pd.DataFrame(rows)


def _common_knowledge(data, frame, embeddings, triplets, args):
    umap_set = frame[
        (frame["method"] == "umap")
        & frame["in_rashomon"]
        & frame["error"].eq("")
        & ~frame["undertrained"]
    ]
    if len(umap_set) < 2:
        print("  skip common knowledge: fewer than 2 UMAP Rashomon embeddings", flush=True)
        return None
    views = [embeddings[i] for i in umap_set["embedding_id"]]
    edges = candidate_edges(data["X"], k=30)
    refined = mean_score_edges(views, edges, keep_per_point=15)
    graph = edges_to_graph(refined, data["n"])
    print(f"  common-knowledge UMAP on {len(refined)} edges", flush=True)
    try:
        embedding, seconds = fit_cne(
            data["X"],
            data["X_pca"],
            "umap",
            seed=0,
            label_weight=0.0,
            n_epochs=args.cne_epochs,
            graph=graph,
        )
        probe = contrastive_probe(embedding, high_d_neighbors(data["X"], 15), "umap")
        error = ""
    except Exception as exc:
        traceback.print_exc()
        embedding = np.full((data["n"], 2), np.nan)
        seconds = np.nan
        probe = np.nan
        error = f"{type(exc).__name__}: {exc}"
    scores = {}
    if np.isfinite(embedding).all():
        scores = evaluate_row(data, embedding, triplets, args.k_eval)
    np.save(args.out / data["name"] / "embeddings" / "common_knowledge_umap.npy", embedding)
    row = {
        "dataset": data["name"],
        "embedding_id": "common_knowledge_umap",
        "kind": "common_knowledge",
        "method": "umap",
        "seed": 0,
        "lambda_pca": 0.0,
        "n_epochs": args.cne_epochs,
        "undertrained": False,
        "probe_loss": probe,
        "train_seconds": seconds,
        "error": error,
        "consensus_stress": np.nan,
        "consensus_seconds": np.nan,
        "n_views": len(views),
        "source_weight": np.nan,
        "in_rashomon": np.nan,
        "loss_reference": np.nan,
        "loss_tolerance": np.nan,
        **scores,
    }
    return row, embedding


def main():
    args = parse_args()
    started = time.perf_counter()
    frames = []
    for name in args.datasets:
        frames.append(run_dataset(name, args))
    args.out.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_csv(path) for path in sorted(args.out.glob("*/metrics.csv"))]
    if parts:
        pd.concat(parts, ignore_index=True, sort=False).to_csv(
            args.out / "metrics_all.csv", index=False
        )
    print(f"\nFinished in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
