# SHIVA

**Substate-Handling Iterative Vector Assessment**
*A pipeline for predicting amide hydrogen chemical shifts.*

---

## Installation

The easiest way to set up SHIVA and its dependencies is via Conda:

```bash
conda env create -f SHIVA_NMR.yml
```

---

## Overview

This repository provides the basic tools to generate chemical-shift estimates for the amide hydrogens of any protein ensemble. SHIVA is still a work in progress, but the core functionality is in place.

You can run SHIVA in two ways:

1. **PDB directory**: collect a series of PDB files in one folder (topologies must be identical - any non-matching topologies are thrown out).
2. **BioEmu trajectory**: generate an XTC trajectory with BioEmu (recommended, as it matches the training protocol).

> If you supply your own PDBs, please remove hydrogens beforehand. SHIVA can add them for you, and inconsistent hydrogen naming/placement is a common source of error.

---

## Dependencies

Make sure you have the following versions installed:

| Package    | Version |
| ---------- | ------- |
| Python     | 3.12.9  |
| bioemu     | 0.1.12  |
| numpy      | 2.2.5   |
| scipy      | 1.15.2  |
| pandas     | 2.2.3   |
| openmm     | 8.2     |
| MDAnalysis | 2.9.0   |
| mdtraj     | 1.10.3  |
| xgboost    | 3.0.0   |

> **Notes:**
>
> * **BioEmu** isn’t strictly required (and it isn't in the yaml), but it simplifies ensemble generation (requires CUDA ≥12). See its [GitHub](https://github.com/microsoft/bioemu).
> * **xgboost** is only needed to unpickle the trained models.

---

## Quick Start

### 1. Generate features

From the repository root:

* **With BioEmu (XTC input):**

  ```bash
  python scripts/run_pipeline_batch_individualDSSP_hbond_withSS_predictOnly.py \
    --input-dir input_xtc \
    --output-dir feats
  ```

* **With PDB directory:**

  ```bash
  python scripts/run_pipeline_batch_individualDSSP_hbond_withSS_predictOnly_pdbDir.py \
    --input-dir input_pdb \
    --output-dir feats
  ```

### 2. Process features & predict

Run the following, in order:

```bash
python scripts/add_ensemble_statistics_to_features_v3.py feats feats_SS_ensemble
python scripts/split_ss_types_v2.py feats_SS_ensemble feats_SS_ensemble_H feats_SS_ensemble_E feats_SS_ensemble_C
python scripts/xgboost_loadedModel_predOnly_C.py
python scripts/xgboost_loadedModel_predOnly_E.py
python scripts/xgboost_loadedModel_predOnly_H.py
python scripts/append_HEC_results.py
python scripts/xgboost_aggregate_and_eval.py test_predictions_all.csv
```

After completion, your final per-state predictions will be in `test_predictions_all.csv` with averaged predictions in `test_after_avg.csv`.

> **Note:** All subdirectories under `--input-dir` will be processed.

---

## Example

An example is included for BMRB entry **6395**, with predictions from both a BioEmu ensemble and an NMR ensemble. The BioEmu–based result is more accurate (though further testing is ongoing).

---

Happy predicting!
