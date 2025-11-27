# Neural Operator Evaluations

This repository contains the thesis project **“Evaluation of Neural Operators for Parametric PDEs”**, focusing on benchmarking different neural-operator architectures on a range of PDE problems.

## Overview
Neural operators are deep learning architectures designed to learn mappings between function spaces, enabling efficient and resolution-invariant PDE solving.  
This project evaluates multiple neural-operator variants on 1D, 2D, and 3D PDE datasets, comparing their accuracy, generalization, and computational performance.

## Repository Structure
```text
data/                    # PDE datasets (inputs/outputs)
models/                  # Model definitions (FNO, low-rank, custom variants)
utilities*.py            # Data loading, preprocessing, metrics, visualization
*.ipynb                  # Experiment notebooks for each PDE and model type
model_comparison.ipynb   # Summary comparison of all models
