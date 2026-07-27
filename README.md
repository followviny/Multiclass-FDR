<div align="center">

# Multiclass False Discovery Rate Control

**Selective multiclass classification with statistical control of incorrect accepted predictions**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-reproducible-F37626?logo=jupyter&logoColor=white)](code/simulations_clean.ipynb)
[![Thesis](https://img.shields.io/badge/PDF-full_thesis-B31B1B?logo=adobeacrobatreader&logoColor=white)](paper/multiclass_fdr_thesis.pdf)
[![License](https://img.shields.io/badge/code_license-MIT-2EA44F)](LICENSE)

[Read the thesis](paper/multiclass_fdr_thesis.pdf) ·
[Explore the experiments](code/simulations_clean.ipynb)

</div>

## Overview

Modern classifiers should be able to abstain when they are uncertain. This project studies how to accept only reliable multiclass predictions while keeping the **False Discovery Rate (FDR)** - the expected fraction of mistakes among accepted predictions - below a user-defined level α.

The work combines theoretical analysis, controlled simulations, and experiments on CIFAR-10 neural-network logits. It compares naive empirical-null constructions with knockoff, knock-in, Mix-Max, Benjamini-Hochberg, and cascaded procedures.

![Balanced CIFAR-10 results](figures/cifar10_balanced_results.png)

## Main findings

- Naive null-distribution constructions can produce invalid p-values in the multiclass setting.
- The single-stage knockoff procedure controls FDR under a homogeneous-null assumption.
- All seven evaluated methods control FDR on CIFAR-10 when calibrated with generated null scores.
- Cascaded FDR is most useful at strict FDR levels when class proportions are heterogeneous.
- On balanced data, and at more permissive FDR levels, single-stage methods generally have higher power.

The imbalanced experiment illustrates the regime where the cascade is most useful:

![Imbalanced CIFAR-10 results](figures/cifar10_imbalanced_results.png)

## Methods

| Family | Procedures | Role |
| --- | --- | --- |
| Oracle | Ground-truth FDP | Reference for evaluating power and calibration |
| Target-decoy | Knockoff, knock-in | Estimate false discoveries through target-decoy competition |
| Distributional | Mix-Max | Estimate both null samples and misclassifications |
| Multiple testing | Benjamini-Hochberg | Control FDR using empirical per-class p-values |
| Sequential | Cascaded FDR | Process classes one at a time, prioritizing easier classes |

## Repository structure

```text
.
├── code/
│   └── simulations_clean.ipynb
├── data/
│   ├── cifar10_train_logits.npz
│   ├── cifar10_test_logits.npz
│   └── null_viki_cifar_test_logits.npy
├── figures/
│   ├── cifar10_balanced_results.png
│   └── cifar10_imbalanced_results.png
├── paper/
│   └── multiclass_fdr_thesis.pdf
├── requirements.txt
└── README.md
```

## Reproducing the experiments

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/followviny/Multiclass-FDR.git
cd Multiclass-FDR

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab code/simulations_clean.ipynb
```

The notebook locates `data/` and `figures/` automatically whether Jupyter is launched from the repository root or from `code/`.

Some simulation sections use large sample sizes and repeated parallel runs. Reduce the corresponding `N`, `M`, or repetition parameters for a quick local run.

## Data

- `cifar10_train_logits.npz`: logits and labels for 50,000 CIFAR-10 training samples.
- `cifar10_test_logits.npz`: logits and labels for 10,000 CIFAR-10 test samples.
- `null_viki_cifar_test_logits.npy`: 10,000 generated null-logit vectors used for calibration.

## Thesis

The complete theoretical development, proofs, experimental setup, and discussion are available in [the thesis PDF](paper/multiclass_fdr_thesis.pdf).

## Author

**Viktoriia Fokina**

## License

The code and notebook are released under the [MIT License](LICENSE). The thesis text, figures extracted from it, and provided data are © 2026 Viktoriia Fokina and are not covered by the MIT license.
