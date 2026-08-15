# LDFGN Reproducibility Package

This directory provides the deterministic LDFGN implementation used in the
paper experiments, 24 datasets, the dataset-specific 48/72/96-point
hyperparameter grid-search code, reference best results, and scripts for quick
reproduction.

## Repository Structure

```text
LDFGN/
├── Datasets/                  # 24 experimental datasets
├── reference/grid_best.csv    # Reference best configurations and metrics
├── data.py                    # Data loading, min-max scaling, and one-hot encoding
├── ldfgn.py                   # LDFGN model and deterministic construction
├── protocol.py                # Fixed training setup and 48/72/96-point candidate rules
├── quick_validate.py          # Train the 24 reference best configurations directly
├── grid_search.py             # Complete resumable 48/72/96-point grid search
├── run_quick_validation.sh    # Quick-validation entry point
├── run_full_grid.sh           # Full-grid entry point
├── dataset_sha256.txt         # Dataset hashes
├── reference_sha256.txt       # Reference-result hash
├── requirements.txt
└── tests/test_protocol.py
```

## Experimental Protocol

### Deterministic Construction

- Attributes are partitioned into deterministic views in their original order,
  while balancing encoded widths.
- All expanded coordinates of a one-hot-encoded attribute remain in the same
  view.
- Granule centers use unscrambled Sobol sequence indices `1..K`; indices
  `1..16` are reserved for center initialization, with `K <= 16`.
- Low-normality reference points use unscrambled Sobol sequence indices `32+`.
- If an encoded reference point coincides with a granule center after
  projection, it is deterministically rejected and the next point is used.
- Categorical coordinates of reference points are always mapped to valid
  one-hot vectors.
- The trusted set is stably sorted by normality in descending order; ties are
  resolved by the original sample index.
- PyTorch deterministic algorithms are enabled, while cuDNN benchmarking and
  TF32 are disabled.
- The anomaly score is fixed as `S = 1 - Z`; label-based orientation reversal is
  not used.

### Fixed Training Configuration

| Setting | Value |
|---|---:|
| Epochs | 150 |
| Optimizer | Adam |
| Training mode | Full batch |
| Learning rate | 0.02 |
| Reference ratio `p` | 0.8 |
| Loss aggregation | Sum |
| Normality sharpness `kappa` | 3.0 |
| Variance floor `epsilon` | 1e-3 |
| Feature-weight offset `delta` | 0.01 |
| Feature regularization | 0 |
| Feature weights | ReLU-positive, unnormalized |
| Center/log-variance weight decay | 1e-4 |
| Feature/view weight decay | 0 |

### 48/72/96-Point Search Space

```text
K     = {1, 5}
q     = {0.80, 0.95, 0.975}
gamma = {0.5, 2.0, 3.0, 5.0}
```

The base number of views is computed solely from the number of original
attributes, `D_o`:

```text
b(D_o) = 1                                      if D_o < 4
b(D_o) = max(2, min(10, floor(D_o / 3)))       otherwise
```

When `D_o <= 50`, the following candidates are generated around the anchor
`b(D_o)`:

```text
{round_half_up(b/2), b, min(D_o, 2b)}
```

This usually produces three values of `V`, giving
`3 x 2 x 3 x 4 = 72` configurations per dataset.

When `D_o > 50`, the additional anchor `min(20, floor(D_o/2))` is used. The
candidates generated from both anchors are combined into a set, usually
producing `{5, 10, 20, 40}` and therefore `4 x 2 x 3 x 4 = 96`
configurations. After deduplication, the `http` dataset has only `{1, 2}` as
its view candidates, resulting in 48 configurations. Across all 24 datasets,
the search contains 1,872 configurations.

For each dataset, configuration selection is based exclusively on AUC. The
candidate grid has a fixed enumeration order: `V` in ascending order, followed
by `K`, `q`, and `gamma` in the orders shown above. A candidate replaces the
incumbent only when its AUC is strictly greater. If multiple configurations
have exactly the same AUC, the first configuration in this fixed enumeration
order is retained. MaxF1 is computed by sweeping all score thresholds under
the configuration selected exclusively by AUC; no independent,
metric-specific search is performed. AP may be retained as an auxiliary output
inside the code, but it is not a configuration-selection criterion. Because
labels from the same complete dataset are used to select configurations, this
is a labeled oracle benchmark rather than an unsupervised model-selection
protocol.

## Environment Setup

Tested environment:

```text
Python        3.12.7
NumPy         1.26.4
SciPy         1.16.1
scikit-learn  1.5.1
PyTorch       2.8.0
CUDA          12.8
cuDNN         91002
GPU           NVIDIA A100-PCIE-40GB
```

Using Conda:

```bash
cd /path/to/LDFGN
conda env create -f environment.yml
conda activate ldfgn
```

Alternatively, install the dependencies in an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

Changes to the GPU, CUDA, PyTorch, or underlying mathematical libraries may
cause small floating-point differences. The environment above is recommended
for the closest numerical reproduction of the bundled reference results.

## Integrity Checks

```bash
cd /path/to/LDFGN
sha256sum -c dataset_sha256.txt
sha256sum -c reference_sha256.txt
python -m unittest discover -s tests -v
```

The SHA-256 hash of `reference/grid_best.csv` is:

```text
532436264e2502dc894364766dbd8f11810ae420b9efd65753d51ba7836bc49f
```

## Quick Validation

Quick validation does not repeat the search over all 1,872 configurations.
Instead, it reads `reference/grid_best.csv` and directly trains the selected
configuration once for each dataset. It strictly checks:

- AUC, MaxF1, and AP with a tolerance of `5e-10`;
- that the selected parameters belong to the dataset's 48/72/96-point
  candidate grid;
- the total number of candidates;
- the deterministic view partition; and
- the number of rejected Sobol reference-point collisions.

AP is checked only as an auxiliary implementation output; it is neither a
paper-reported metric nor a configuration-selection criterion.

Run all 24 datasets:

```bash
DEVICE=cuda:0 PYTHON_BIN=python ./run_quick_validation.sh
```

Validate only selected datasets:

```bash
python quick_validate.py --device cuda:0 \
  --datasets Lymphography annthyroid
```

Outputs:

```text
results/quick_validation.csv
results/quick_validation.metadata.json
results/quick_validation.log
```

Strict reproduction succeeds only when the process exits with status code 0
and prints `ALL CHECKS PASSED`.

## Full Grid Search

The full grid search trains all 1,872 configurations. The implementation writes
each configuration to CSV immediately and supports resuming interrupted runs.

```bash
DEVICE=cuda:0 PYTHON_BIN=python ./run_full_grid.sh
```

It can also be run directly:

```bash
python grid_search.py \
  --device cuda:0 \
  --output-dir results/full_grid
```

By default, completed datasets and ROC outputs are skipped. To clear the output
and restart from the beginning, use:

```bash
python grid_search.py \
  --device cuda:0 \
  --output-dir results/full_grid \
  --overwrite
```

After the full grid finishes, the script automatically compares the generated
selected parameters, AUC, MaxF1, AP, view partitions, and Sobol reference-point
collision counts against `reference/grid_best.csv`. It exits with a nonzero
status if any value does not match.

Main outputs:

```text
results/full_grid/grid_detail.csv
results/full_grid/grid_best.csv
results/full_grid/grid_best_roc_101.csv
results/full_grid/grid_best_roc_raw.csv
results/full_grid/grid_best_roc_101_wide.csv
results/full_grid/grid_metadata.json
results/full_grid.log
```

## Quick Validation vs. Full Search

| Script | Training runs | Purpose |
|---|---:|---|
| `quick_validate.py` | 24 | Verify that the algorithm, environment, and reference selected results are reproducible |
| `grid_search.py` | 1,872 + 24 selected-configuration reruns | Regenerate the LDFGN main-grid results and ROC outputs reported in the paper |

Quick validation shows that the metrics can be reproduced from the given
selected parameters. Only the full grid search also demonstrates that the
search procedure selects the same configurations again.
