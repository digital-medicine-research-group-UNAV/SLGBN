# Semi-Linear Gaussian Bayesian Networks (SLGBN)

This repository contains the code used for the development and experimental evaluation of the paper: **“Semi-Linear Gaussian Bayesian Networks”**

## 🧠 Model Overview

The **SLGBN model** extends Linear Gaussian Bayesian Networks by:

- Allowing **mixed node types** (linear and nonlinear)  
- Using **GAM + B-splines** for nonlinear dependencies  
- Maintaining **Gaussian assumptions**  
- Enabling **structure learning via Hill-Climbing**

---

## 📁 Repository Structure

```
├── output/            # Generated figures
├── results/           # Experimental results
├── synthetic_data/    # Scripts to generate synthetic networks
│
├── greedy_hc_slgbn.py # Hill-Climbing algorithm for SLGBN structure learning
├── experiments.py     # Runs the experiments from the paper
├── auxiliar.py        # Auxiliary functions (metrics, structure learning utilities, etc.)
├── generate_plots.py         # Generates figures from stored results
```

---

## ⚙️ Main Components

#### `greedy_hc_slgbn.py`
Implements the **Greedy Hill-Climbing (HC)** algorithm used for structure learning in SLGBNs.

#### `experiments.py`
Runs all experiments reported in the paper. It:
- Generates data
- Executes structure learning with fixed and free structure for SLGBNs, SPBNs and LGBNs.
- Stores results in the `/results` directory

#### `auxiliar.py`
Contains helper functions, including:
- Model evaluation metrics
- Structural learning utilities
- Other supporting functions

#### `figures.py`
Generates all figures included in the paper using the results stored in `/results`.

#### `synthetic_data/`
Includes scripts to generate the synthetic Bayesian networks used in the experiments.

#### `results/`
Stores outputs from experiments, including:
- Log-likelihood values
- Structural metrics

#### `output/`
Contains the generated figures.

---

## 🚀 Reproducing the Experiments

To reproduce the results from the paper:

1. Run the experiments:
```bash
python experiments.py
```

2. Generate the figures:
```bash
python figures.py
```

Results will be stored in:

- `/results` (raw outputs)
- `/output` (figures)

---

## 📄 Reference

If you use this code, please cite:

```bash
Semi-Linear Gaussian Bayesian Networks
```
