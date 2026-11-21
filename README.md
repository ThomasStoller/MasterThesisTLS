# Multi-Group BLE Cluster Fingerprinting in High-Density IoT Environments

Code, results and figures of the master thesis of Thomas Stoller at the Communication Systems
Group (CSG), Department of Informatics, University of Zurich (supervisors: Katharina O. E. Müller,
Chao Feng, Prof. Dr. Burkhard Stiller; July 2026 to January 2027).

The thesis asks whether **two co-located groups of BLE devices with identical hardware** (each a
phone, a pair of earbuds and a tracker, as one person would carry them) can be told apart from
each other and from the background, using only passively sniffed BLE advertisement packets and
without the received signal strength (RSSI) as a model input.

## Background: the two preceding theses

This work extends the pipeline of two earlier CSG master theses, which identified **one** device
group against background traffic (a binary task):

| Thesis | Repository | Contribution |
|---|---|---|
| Joel Meier (2025), *Clustered Fingerprinting and Tracking in High-Density Internet of Things (IoT) Environments* | [JoeMei0603/Master-Thesis-Joel-Meier](https://github.com/JoeMei0603/Master-Thesis-Joel-Meier) | Original pipeline: labelling, feature engineering, supervised (RF, MLP, SGD) and unsupervised (k-means, BIRCH, bisecting k-means) identification of one Samsung device group |
| Johanna Bieri (2026), *Scalable BLE Cluster Fingerprinting for High-Density IoT Environments* | [johannauzh/BLEGroupID](https://github.com/johannauzh/BLEGroupID) | Transfer of the pipeline across ecosystems (Samsung to Apple) |

This repository started as a copy of the BLEGroupID repository. The notebooks of the preceding
theses are kept unchanged in `code/archive/` for reference. Everything in `code/thesis/` was
written or extended for this thesis: three classes instead of two (background, group 1, group 2),
17 new features, 10 hour captures, the per-source split, the binary/three-class comparison, the
per-source majority vote, significance tests, feature family ablation and the error analysis.

## The six experiments

Each experiment places two groups of three devices next to a passive nRF Sniffer (Wireshark,
advertising channels 37/38/39). Training uses a long capture at a residential location (Seewis,
split into two hour blocks); evaluation uses a separate two hour capture in a university building
(different day, different location, much denser background).

| ID | Device set | Group 1 | Group 2 | Distance condition | Training | Evaluation |
|---|---|---|---|---|---|---|
| E1 | Apple | iPhone 12, AirPods Pro (1st gen), Galaxy SmartTag | identical models | equal distance | 10 h (5 x 2 h) | 2 h university |
| E2 | Apple | iPhone 12, AirPods Pro (1st gen), Galaxy SmartTag | identical models | different distance | 10 h (5 x 2 h) | 2 h university |
| E3 | Samsung | Galaxy A54, Galaxy Buds FE, AirTag | identical models | equal distance | 10 h (5 x 2 h) | 2 h university |
| E4 | Samsung | Galaxy A54, Galaxy Buds FE, AirTag | identical models | different distance | 10 h (5 x 2 h) | 2 h university |
| E5 | Mixed | Galaxy A54, Galaxy Buds FE, AirTag | iPhone 12, AirPods Pro, Galaxy SmartTag | equal distance | 6 h (3 x 2 h) | 2 h university |
| E6 | Mixed | Galaxy A54, Galaxy Buds FE, AirTag | iPhone 12, AirPods Pro, Galaxy SmartTag | different distance | 6 h (3 x 2 h) | 2 h university |

* **Equal distance:** both groups at the same nominal distance from the sniffer (the control).
  **Different distance:** one group close, one farther away (about 30 dB RSSI difference). RSSI is
  recorded but never used as a feature, so the pair tests how much separation comes from geometry
  through other features.
* **Labels:** a manual whitelist of source addresses per capture (address chains traced by time
  continuity, payload structure and documented placement). Every packet gets `group_id`
  (0 background, 1, 2), `group_label_multi` (0/1/2) and `group_label_bin` (0 background, 1 any group).
* **Models:** random forest (50 trees, balanced class weights), multi-layer perceptron and SGD,
  all with fixed hyperparameters and seed 1 for every experiment.
* **Features:** 42 features in eight families (PDU role, channel and PHY, payload shape, burst
  microstructure, joint activity, payload dynamics, timing, shape and session span). Payload
  content and RSSI are excluded.
* **Evaluation:** macro F1 (main metric, chance level 0.33 for three classes), MCC, AUC-PR and per
  class F1; packet split and split grouped by source address (no address in both training and
  test); in domain (30 % held out of the training capture) and transfer (university capture);
  per-source majority vote; confidence intervals and tests by bootstrap over source addresses.

### Current results (random forest, split grouped by source, university capture)

| Experiment | Three-class macro F1 [95 % CI] | Binary macro F1 [95 % CI] | Three-class, in domain |
|---|---|---|---|
| E1 apple same | 0.38 [0.36, 0.41] | 0.64 [0.61, 0.67] | 0.91 |
| E2 apple diff | 0.49 [0.43, 0.55] | 0.56 [0.53, 0.58] | 0.89 |
| E3 samsung same | 0.47 [0.43, 0.51] | 0.68 [0.64, 0.73] | 0.97 |
| E4 samsung diff | 0.57 [0.54, 0.60] | 0.71 [0.70, 0.73] | 1.00 |
| E5 mixed same | 0.39 [0.37, 0.42] | 0.61 [0.59, 0.64] | 0.90 |
| E6 mixed diff | 0.54 [0.49, 0.59] | 0.62 [0.60, 0.65] | 0.91 |

In domain the groups are separated almost perfectly; after the transfer to the university capture
the scores drop by 0.37 to 0.53. The different-distance experiments transfer better in all three
pairs. Full tables: [`code/thesis/analysis/results/`](code/thesis/analysis/results/), all runs:
[`code/thesis/modeling/results_summary.csv`](code/thesis/modeling/results_summary.csv).

## Repository structure

```
code/
  thesis/                      everything the thesis results are based on
    labeling/                  one labelling notebook per capture set (E1-E6, incl. university capture)
    modeling/                  one modelling notebook per experiment + results_comparison.ipynb
      results_summary.csv      all runs, parsed from the executed notebooks
      artifacts/<experiment>/  exported frames, predictions, importances, metrics (not in git)
    analysis/
      01_metrics_and_significance.ipynb      all metrics, binary vs three class, CIs, tests, block variability
      02_feature_family_ablation.ipynb       one feature family removed at a time
      03_error_analysis_by_device.ipynb      errors per device, address, manufacturer and time
      04_feature_importance_per_family.ipynb importance summed per feature family
      05_clustering_and_payload.ipynb        unsupervised clustering, unit-specific payload bytes
      analysis_utils.py                      shared helpers (metrics, bootstrap, tests, loading)
      results/                               result tables (csv) and figures (pdf)
  archive/                     earlier and superseded work, kept for reference (not used in the thesis)
    labeling/  modeling/       first experiment, apartment runs, 8 h runs, notebooks of the preceding theses
    ablation/                  early single-feature ablation study
plots/                         plots of the preceding theses
data/                          raw and labelled captures (not in git, see below)
```

| Experiment | Labelling notebook | Modelling notebook |
|---|---|---|
| E1 | `dataset_labeling_10h_IAPST_same.ipynb` | `modeling_10h2h_IP12_AAP1P_ST_SameDist.ipynb` |
| E2 | `dataset_labeling_10h_IAPST_diff.ipynb` | `modeling_10h2h_IP12_AAP1P_ST_DiffDist.ipynb` |
| E3 | `dataset_labeling_10h_SA54_SBFE_AAT_same.ipynb` | `modeling_10h2h_sa54_sbfe_aat_sameDist.ipynb` |
| E4 | `dataset_labeling_10h_SA54_SBFE_AAT_diff.ipynb` | `modeling_10h2h_sa54_sbfe_aat_diffDist.ipynb` |
| E5 | `dataset_labeling_6h_A54FEAirtags_IP12PodsST_same.ipynb` | `modeling_6h2h_A54FEAirtags_IP12PodsST_SameDist.ipynb` |
| E6 | `dataset_labeling_6h_A54FEAirtags_IP12PodsST_diff.ipynb` | `modeling_6h2h_A54FEAirtags_IP12PodsST_DiffDist.ipynb` |

## Installation

Python 3.12. In a fresh environment (conda or venv):

```bash
pip install -r requirements.txt
```

`requirements.txt` contains pandas, numpy, scikit-learn, matplotlib, scipy and pyarrow (parquet
export). To run the notebooks, also install Jupyter (`pip install notebook`) or use PyCharm.

### Data

The captures are not in the repository (10 h captures are 250 to 300 MB per two hour block, the
university captures up to 1 GB). Place them in `data/` at the repository root, one folder per
capture set, e.g.

```
data/2Groups_3Devices_10h_Seewis_SamsungA54BudsFEAirtags_SameDistances/
    2Groups_3Devices_10h_Seewis_SamsungA54BudsFEAirtags_SameDistances_Hours0To2.csv          (raw Wireshark export)
    2Groups_3Devices_10h_Seewis_SamsungA54BudsFEAirtags_SameDistances_Hours0To2_labeled.csv  (written by the labelling notebook)
    ...
    2Groups_3Devices_2h_Uni_SamsungA54BudsFEAirtags_SameDistances.csv
    2Groups_3Devices_2h_Uni_SamsungA54BudsFEAirtags_SameDistances_labeled.csv
```

All notebooks use paths relative to their own folder (`../../../data/...`), so run them with the
notebook folder as working directory (the default in Jupyter and PyCharm).

## Usage

As in the preceding theses, cells that perform an action are marked with a `CODE FIELD X` header;
all other cells only define functions or variables and must be run before them (run the notebook
top to bottom).

### 1. Labelling (`code/thesis/labeling/`)
Only needed for new captures; the labelled CSVs already exist for E1-E6.

* **Code Field 0** resets the global state and holds the whitelists (addresses and device rules)
  of the capture.
* **Code Field 1** loops over the two hour blocks and the university capture, assigns
  `labelled_device`, `sublabel_device`, `group_id` and the two derived labels, drops malformed
  packets and writes `*_labeled.csv` next to the raw file.
* The cells after Code Field 1 summarise the sources per block (packet counts, RSSI, payload
  structure) to check for missed address chains. RSSI is used here only to trace chains during
  labelling, never as a model input.

### 2. Modelling (`code/thesis/modeling/`)
Run each experiment notebook from the top **through CODE FIELD 6** (about 30 to 55 minutes per
experiment, about 7 GB RAM; run them one at a time).

| Code Field | Purpose |
|---|---|
| 1 | Load the training blocks and the university capture, source aggregates, `block` index, `EXPERIMENT_ID` |
| 2 | Packet counts in windows of 2 ms, 25 ms, 100 ms and 1 s per source |
| 2b | Joint activity, payload dynamics and timing features (500 ms bins), computed separately per capture |
| 3 | Quantile binning (15 bins per capture) and cap of 1000 packets per source |
| 3b | Export of the balanced frames to `artifacts/<EXPERIMENT_ID>/` |
| 4, 5 | Class counts and feature cardinalities (informative only) |
| 6 | RF, MLP and SGD x packet / grouped split x three-class / binary label; in domain and university evaluation; saves predictions, class scores, importances and a metrics row (macro F1, MCC, AUC-PR, majority vote per source) |
| 7, 8 | Learning curves (optional) |
| 9-13 | Unsupervised clustering and PCA plots (optional; superseded by `analysis/05`, these cells include the label columns in the clustering input) |

Afterwards run `results_comparison.ipynb` (seconds) to rebuild `results_summary.csv`.

### 3. Analysis (`code/thesis/analysis/`)
The analysis notebooks read only `modeling/artifacts/` and never redo the feature engineering.
Run them top to bottom in any order after the modelling notebooks:

| Notebook | Runtime | Output |
|---|---|---|
| 01 metrics and significance | about 10 min (incl. leave-one-block-out) | `metrics_all_runs.csv`, `h2_binary_vs_multi.csv`, `confidence_intervals.csv`, `significance_tests.csv`, `block_variability.csv` |
| 02 feature family ablation | about 5 min | `ablation_feature_families.csv`, ablation heatmaps |
| 03 error analysis by device | about 2 min | `error_by_device*.csv`, `false_positives_by_company.csv`, `error_over_time.csv` |
| 04 feature importance per family | seconds | `feature_importance_*.csv`, importance heatmaps |
| 05 clustering and payload | about 1 min | `clustering.csv`, `payload_unit_bytes.csv` |

Tables are written to `code/thesis/analysis/results/`, figures (pdf) to `results/figures/`.

## Status (November 2026)

* Done: labelling and modelling of E1-E6 (after a label correction pass), binary task, majority
  vote, all analysis notebooks.
* Open: thesis text (results, evaluation, conclusion), final presentation.
