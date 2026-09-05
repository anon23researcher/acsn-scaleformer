# Adaptive Cross-Scale Normalisation (ACSN)

This repository contains the code and experimental results for:

**"Is Normalization Necessary in Multi-Scale Exchange-Rate Forecasting? A Diagnostic Study with Adaptive Cross-Scale Normalization"**

---

## Overview

We introduce **Adaptive Cross-Scale Normalisation (ACSN)**, a learnable mechanism that allows the model to determine the strength of normalisation at each temporal scale.

The goal is **diagnostic**, not performance improvement:
we test whether multi-scale transformer models actually *use* normalisation.

---

## Key Findings

- Normalisation has **negligible impact** on forecasting performance  
- Differences across strategies are **small and inconsistent**  
- Learned parameters converge toward **near-identity** behaviour  
- Models do **not strongly rely** on explicit normalisation  

---

## Repository Structure

    .
    ├── models/            Model definitions (Autoformer-MS, Informer-MS)
    ├── exp/               Training and experiment logic
    ├── data_provider/     Data loading utilities
    ├── layers/            Transformer components
    ├── utils/             Helper functions
    ├── scripts/           Reproducibility scripts
    ├── results/           CSV outputs (main results + statistical tests)
    ├── paper_figures/     Figures used in the paper
    ├── run.py             Main experiment entry point
    ├── requirements.txt   Python dependencies
    └── README.md

---

## Reproducing Results

Run the main experiment:

    bash scripts/run_all.sh

---

## Key Outputs

Results used in the paper:

- results/paired_tests_clean_5seed.csv  
- results/paper_results_final.csv  
- results/paper_table_ready.csv  

ACSN diagnostics:

- results/alpha_trajectory.csv  
- paper_figures/alpha_trajectory.png  
- paper_figures/alpha_by_scale.png  

---

## Experimental Setup

- Dataset: Exchange-Rate (Lai et al. 2018)  
- Models: Autoformer-MS, Informer-MS  
- Forecast horizons: 96, 192  
- Seeds: 5 (matched paired design)  
- Metrics: MSE, MAE  
- Statistical tests: paired t-tests  

---

## Notes on Reproducibility

- All configurations used in the paper are specified in `run.py`  
- The provided script reproduces the main experiment setting  
- Results are averaged over 5 seeds for statistical validity  
- GPU is recommended for faster training  

---

---

## Acknowledgement

This repository builds upon publicly available implementations of transformer-based time-series forecasting architectures.

Adapted in part from:
- Scaleformer: https://github.com/BorealisAI/scaleformer

---


## License

This project is based on Scaleformer and adapted for ACSN research.
