# Fourier Neural Operator for Acoustic Wavefield Simulation

This repository provides **2D and 3D** acoustic wavefield simulation codes based on Fourier Neural Operators (FNO). It includes data preparation, model architecture, training loops, inference scripts, and result comparisons.

The repo is designed for **educational purposes**. You can directly train the models and easily modify them for your own experiments.

## 2D Acoustic Wavefield Simulation using FNO

The 2D implementation is adapted from Yang et al. (2021). We evaluate the FNO using Gaussian random velocity models. The model is trained on wavefield snapshots generated from **20 random source locations** and tested on **5 unseen sources** during inference.

## 3D Acoustic Wavefield Simulation using Modified UFNO

3D wavefield modeling is significantly more computationally demanding than the 2D case. To address this, we adopt the **U-Net enhanced Fourier Neural Operator (UFNO)**, which augments the standard FNO block with a parallel U-Net branch. This improves the model's ability to capture fine-grained temporal and spatial features.

To fit the model on a single NVIDIA RTX 3090 GPU, we modified the original UFNO by reducing the number of dense connections, substantially lowering memory usage while maintaining strong performance.

### Implementation Details

Target wavefield data were generated using the **Deepwave** finite-difference solver on a $96 \times 96 \times 96$ velocity model with **10 m** grid spacing. The acoustic wave equation was solved with:
- Fourth-order spatial discretization
- 20-cell Perfectly Matched Layer (PML) boundaries
- 5 Hz Ricker wavelet source
- 1.0 s simulation time, from which **20 spatiotemporal snapshots** were uniformly sampled

The UFNO surrogate model learns to map the static velocity model and source location directly to the full time-evolving wavefield. It was trained on **30 source locations** and evaluated on **5 unseen test sources**.

As shown in the results (Figure \ref{fig:ufno-wavefields}), the data-driven surrogate accurately reproduces the spatiotemporal dynamics of the propagating wavefield. Rather than producing a smoothed kinematic approximation, the model captures high-frequency effects such as **diffraction and multipathing**. Errors remain low throughout the simulation, indicating that the surrogate preserves both phase coherence and amplitude fidelity even after the wavefront interacts with structural heterogeneities.

# Reference
We utilized the FNO-torch 1.6.0 in this package:  https://github.com/zongyi-li/fourier_neural_operator/tree/master/FNO-torch.1.6

We also made some modifications using this package: https://github.com/yanyangg/AcousticNeuralOperator

We used Devito to generate the training samples: https://github.com/devitocodes/devito


Feel free to contact me if you have any questions: yang.cui512@gmail.com
