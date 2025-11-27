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
```

##  Features
- Fourier Neural Operator (FNO) experiments  
- Low-rank operator variants  
- 1D & 2D Burgers’ equation experiments
- 3D Navier-Stokes experiments
- Custom visualization + error analysis utilities  
- Direct comparison of models under identical conditions  

## Running Experiments
1. Install dependencies `environment.yml` (via conda).  
2. Place datasets in the `data/` directory.  
3. Open any notebook (e.g., `FNO3D.ipynb`, `Lowrank2D_Burger.ipynb`).  
4. Run training & evaluation cells.  
5. Use `model_comparison.ipynb` to compare results across models.

## Results
The project reports:
- Relative & absolute error metrics  
- Generalization across time and resolution  
- Model stability and efficiency  
- Visual comparisons (target vs prediction vs error)

## Thesis Context
This work was completed as part of a Master’s thesis at FAU Erlangen-Nürnberg, evaluating neural operators for scientific machine learning and PDE surrogate modeling.

## References
- Z. Li et al., *Fourier Neural Operator for Parametric PDEs*

