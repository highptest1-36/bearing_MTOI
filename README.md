# VTOI — reproducibility package

Code, fitted parameters and per-bearing result tables for:

> **VTOI: Controlled Attribution of a Transparent Vibration Health Index in Leakage-Aware
> Bearing Prognostics**
> Cao-Phuc Ha, Phu-Nguyen Le — under review at *IEEE Access*.

The article builds a transparent vibration-only health index (a Mahalanobis abnormality score
combined with a state-change magnitude under a convex weight), evaluates it under a leakage-free
leave-one-bearing-out protocol on PRONOSTIA and XJTU-SY, and then uses a ten-arm control battery
to separate what the index contributes from what the conditioning channel contributes.

This repository holds the four artefacts named in the article's Data Availability statement.

## Contents

| Path | What it is |
|---|---|
| `src/vtoi.py` | The index implementation, including the leakage assertion (`assert_no_leakage`, line 334) |
| `src/lobo_v2.py` | The leakage-free leave-one-bearing-out driver |
| `src/`, `scripts/`, `configs/` | Feature extraction, models, losses, samplers, baselines, analysis scripts |
| `results/tables/v2_vtoi_params/` | Per-bearing estimator records — one file per (dataset, seed, held-out bearing): the fitted weight, the four normalisation bounds, the covariance estimator used, its shrinkage coefficient and the baseline size |
| `results/tables/v2/lobo_v2_*_perfold.csv` | Fold definitions and per-fold results |
| `results/tables/v2/*.csv` | The per-bearing result tables underlying every table in the article |
| `results/figures/` | The six figures that appear in the article |

## Which file backs which table

| Article | File in `results/tables/v2/` |
|---|---|
| Table 1 — indicator quality | `hi_quality_all.csv` |
| Table 2 — fitted weights and realised range | `weights_distribution.csv`, `vtoi_range.csv` |
| Table 3 — early warning | `early_warning_v2.csv` |
| Table 4 — onset sensitivity | `onset_sensitivity.csv` |
| Table 5 — leave-one-bearing-out results | `main_results.csv` |
| Table 6 — Wilcoxon signed-rank tests | `wilcoxon.csv` |
| Table 7 — factorial design and seed variability | `factorial_2x2.csv`, `seed_variability.csv` |
| Table 8 — classical and naive baselines | `classical.csv` |
| Table 9 — conformal intervals and attribution | `conformal.csv`, `attribution.csv` |
| Table 10 — ten-arm control battery | `controls.csv` |
| Table 11 — failure cases | `failure_cases.csv` |
| Table 12 — IMS zero-shot transfer | `ims_transfer_seed42.csv` |
| Fig. 4 — weight sweep | `weight_sweep.csv` |

Per-bearing values are also in `per_bearing_all.csv`; the Coble comparison of Section V-A uses
`coble_seed_spread.csv` and `coble_random_baseline.csv`, and the hour-scale conversions of
Section V-D use `deployable_hours.csv`.

## What makes the protocol leakage-free

For each fold the component weight and the four normalisation bounds are fitted on the training
bearings alone and then applied frozen to the validation and held-out bearings. The per-record
quantities (robust centre and scale, baseline mean and covariance) are computed from the first
20 % of that same record, which needs no failure label and is therefore available for a held-out
bearing at inference.

`src/vtoi.py::assert_no_leakage` fails loudly if a held-out bearing appears among the training or
validation bearings, or if the recorded number of training bearings does not match. Records of the
same physical PRONOSTIA bearing distributed twice, as a truncated and as a complete run, are
removed before any split.

## Datasets

Not redistributed here. Obtain them from their original sources:

- **PRONOSTIA / FEMTO-ST** — IEEE PHM 2012 Prognostic Challenge
- **XJTU-SY** — Xi'an Jiaotong University and Changxing Sumyoung Technology
- **IMS** — NASA Prognostics Data Repository

Paths are set in `configs/*.yaml`.

## Requirements

Python 3.10+, with `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `pyyaml` and `matplotlib`.
The analyses that regenerate the tables run on CPU; training the networks used a single GPU.

## Reproducing the tables

```bash
python3 scripts/q1_v2_tier3.py        # indicator quality, early warning, onset, conformal
python3 scripts/q1_v2_stats.py        # Wilcoxon tests with Holm correction
python3 scripts/q1_v2_attribution.py  # gradient, perturbation and occlusion attribution
```

`scripts/verify_paper_numbers.py --check` re-reads the values quoted in the manuscript and
compares each against the corresponding cell of these CSV files.

## Licence

Code is released for academic use. Please cite the article if you use it.
