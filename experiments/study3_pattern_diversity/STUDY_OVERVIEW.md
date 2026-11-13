# Study 3 — Pattern Disambiguation (Synthetic Only)

This study is **isolated** from Study 1 / Study 2.

| Item | Value |
|------|--------|
| Study id | `study3_pattern_diversity` |
| Results root | `results/study3_pattern_diversity/` |
| Job scripts | `experiments/study3_pattern_diversity/job_scripts/` |
| Datasets | Synthetic XOR only (9 datasets; no UCI / TreeFarms) |
| Predictor | PySORTD sparse optimal tree |
| Seeds | 0 … 24 (`N_REPLICATIONS = 25`) |

## Method legend

| Method | Selector model | Strategy | Role |
|--------|----------------|----------|------|
| M1 | RandomForest | Random | Passive baseline |
| M2 | RandomForest | QBC | Synthetic-diversity QBC |
| M3 | PySORTD | QBC (`use_unique_trees=False`, `β=0`) | UNREAL vote entropy |
| M4 | PySORTD | QBC (`use_unique_trees=True`, `β=0`) | Vote entropy on unique patterns |
| M5 | PySORTD | PatternDisambiguation (`count`, `β=0`) | Maximize \|distinct extended patterns\| |
| M6 | PySORTD | PatternDisambiguation (`effective`, `β=0`) | Maximize effective pattern count \(e^{H}\) |
| M7 | PySORTD | PatternDisambiguation (`count`, `β` calibrated) | Gibbs-weighted pattern disambiguation |

## Simulation grid

\[
9 \text{ datasets} \times 7 \text{ methods} \times 25 \text{ seeds} = 1{,}575 \text{ jobs}
\]

Each job is one full active-learning trajectory on one `(dataset, method, seed)`.

### Per-run protocol (same engine as Study 1)

1. Load `src/data/<Dataset>.pkl` (N=500, D=20 binary features).
2. Hold out 20% test; remaining pool starts with 20 stratified labels.
3. Calibrate depth / regularization / Rashomon multiplier (and β if requested) on the pilot labeled set.
4. Iterate until the candidate pool is empty:
   - fit predictor + selector models
   - record accuracy, F1, oracle agreement, TED, Rashomon size, committee size
   - query one point via the method’s selector
5. Write one pickle:
   `results/study3_pattern_diversity/tree_predictor/<Dataset>/M{k}/M{k}_S{seed}.pkl`

### Datasets

| Dataset | Stress |
|---------|--------|
| `Synthetic_XOR_Baseline` | Pure XOR, no noise |
| `Synthetic_XOR_Phi_{05,10,25,45}` | Label noise φ |
| `Synthetic_XOR_Alpha_{25,50,75,100}` | Tree→linear misspecification α |

## What a result pickle contains

Each `SimulationResult` stores:

- `accuracy_history`, `f1_history`, `oracle_agreement_history`, `tree_edit_distance_history`
- `rashomon_size_history`, `committee_size_history`, `epsilon_history`
- `selection_history`, `entropy_history` (per-candidate scores at each step)
- `calibrated_params`, `elapsed_time`

After aggregation (same scripts as Study 1), expect per-dataset plots under:

```text
results/study3_pattern_diversity/tree_predictor/<Dataset>/accuracy_images/
  accuracy_history.png
  f1_history.png
  oracle_agreement_history.png
  tree_edit_distance_history.png
  rashomon_size_history.png
  committee_size_history.png
  ...
```

## How to submit on the cluster

From a login node with the project env:

```bash
# 1. Ensure synthetic pickles exist
python src/utils/preprocess_data.py   # or regenerate Synthetic_* only

# 2. Generate SLURM arrays into this study's job_scripts/
cd experiments/study3_pattern_diversity
python generate_sbatch_arrays.py

# 3. Ignite only this study (writes under results/study3_pattern_diversity/)
cd job_scripts/tree_predictor
bash 0a_ignite.sh
```

Do **not** run Study 1’s ignite script for these jobs — Study 3 keeps its own `job_scripts/` and `results/` tree.
