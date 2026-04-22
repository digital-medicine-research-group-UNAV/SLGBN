# Semi-Linear Gaussian Bayesian Networks (SLGBN)

This repository contains the code used for the development and experimental evaluation of the paper:

**“Semi-Linear Gaussian Bayesian Networks”**

The proposed model, referred to as **SLGBN**, extends classical Bayesian networks by allowing two types of nodes:
- **Linear Gaussian nodes**
- **Nonlinear Gaussian nodes**, estimated using **Generalized Additive Models (GAM) with B-splines**

This framework enables efficient modeling and structure learning in datasets with nonlinear Gaussian relationships.

---

## 📁 Repository Structure

```
├── output/            # Generated figures
├── results/           # Experimental results
├── synthetic_data/    # Scripts to generate synthetic networks
│
├── greedy_hc_slgbn.py # Hill-Climbing algorithm for SLGBN structure learning
├── experiments.py     # Runs the experiments from the paper
├── Auxiliar.py        # Auxiliary functions (metrics, structure learning utilities, etc.)
├── Figures.py         # Generates figures from stored results
```

---

## ⚙️ Main Components

#### `greedy_hc_slgbn.py`
Implements the **Greedy Hill-Climbing (HC)** algorithm used for structure learning in SLGBNs.

### `experiments.py`
Runs all experiments reported in the paper. It:
- Generates or loads data
- Executes structure learning
- Stores results in the `/results` directory

### `Auxiliar.py`
Contains helper functions, including:
- Model evaluation metrics
- Structural learning utilities
- Other supporting computations

### `Figures.py`
Generates all figures included in the paper using the results stored in `/results`.

### `synthetic_data/`
Includes scripts to generate the synthetic Bayesian networks used in the experiments.

### `results/`
Stores outputs from experiments, including:
- Log-likelihood values
- Structural metrics
- Intermediate results

### `output/`
Contains the figures generated from the experimental results.

---

## 🚀 Reproducing the Experiments

To reproduce the results from the paper:

1. Run the experiments:
```bash
python experiments.py
```

2. Generate the figures:
```bash
python Figures.py
```

Results will be stored in:

- `/results` (raw outputs)
- `/output` (figures)

## 🧠 Model Overview

The **SLGBN model** extends Bayesian networks by:

- Allowing **mixed node types** (linear and nonlinear)  
- Using **GAM + B-splines** for nonlinear dependencies  
- Maintaining **Gaussian assumptions**  
- Enabling efficient **structure learning via Hill-Climbing**

This makes SLGBN suitable for modeling complex dependencies while preserving interpretability.

---

## 📌 Notes

- All experiments and figures in the paper can be reproduced using this repository.  
- Synthetic datasets are generated via the scripts in `synthetic_data/`.

---

## 📄 Reference

If you use this code, please cite:

```bash
Semi-Linear Gaussian Bayesian Networks
```
