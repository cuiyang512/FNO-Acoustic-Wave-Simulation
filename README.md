# Fourier Neural Operator for Acoustic Wavefield Simulation
<div style="text-align: right; font-family: Helvetica, Arial, sans-serif; color: #555; margin-top: 20px;">
  <strong>Authors: Yang Cui</strong><br>
  King Fahd University of Petroleum and Minerals, Saudi Arabia<br>
  Uppsala University, Uppsala, Sweden
</div> <br>

This repository provides **2D and 3D** acoustic wavefield simulation codes based on Fourier Neural Operators (FNO). It includes data preparation, model architecture, training loops, inference scripts, and result comparisons.

The repo is designed for **educational purposes**. You can directly train the models and easily modify them for your own experiments.

## 2D Acoustic Wavefield Simulation using FNO

Yang, Yan, et al. "Seismic wave propagation and inversion with neural operators." *The Seismic Record* 1.3 (2021): 126-134.<br>

This project evaluates the effectiveness of the Fourier Neural Operator (FNO; Li et al., 2020) for simulating acoustic wavefields in the time-spatial domain. We utilize a 2D velocity model extracted from the OpenFWI dataset to generate wavefields. A total of 50 wavefields, each comprising 1,000 snapshots, are split into 42 training and 8 validation datasets. The FNO model was trained for 1,000 epochs, with each epoch taking approximately 6.5 seconds on a workstation equipped with an NVIDIA RTX A4500 GPU. To facilitate ease of use, we have provided pre-trained models, allowing users to test the simulation during the course and experiment with the training code later.
![2D Results](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/blob/main/figs/2d_results_fno.png)

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

As shown in the results below, the data-driven surrogate accurately reproduces the spatiotemporal dynamics of the propagating wavefield. Rather than producing a smoothed kinematic approximation, the model captures high-frequency effects such as **diffraction and multipathing**. Errors remain low throughout the simulation, indicating that the surrogate preserves both phase coherence and amplitude fidelity even after the wavefront interacts with structural heterogeneities.
![3D Results](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/blob/main/figs/ufno_results_comparison_new.png)

# Reference
    We utilized the FNO-torch 1.6.0 in this package:  https://github.com/zongyi-li/fourier_neural_operator/tree/master/FNO-torch.1.6
    
    We also made some modifications using this package: https://github.com/yanyangg/AcousticNeuralOperator
    
    We used Devito to generate the training samples: https://github.com/devitocodes/devito


## Development
    We welcome contributions from the open-source community. To contribute, please contact the development team for guidance.

## Contact
For questions, bug reports, development ideas, or collaboration opportunities, please reach out to:
  - Yang Cui: [yang.cui512@gmail.com](mailto:yang.cui512@gmail.com)
