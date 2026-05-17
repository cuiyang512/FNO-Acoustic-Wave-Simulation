# Fourier Neural Operator for Acoustic Wavefield Simulation
In this repo, we provide both 2D and 3D acoustic wavefield simulation codes, including data preparation, network model, training loop, inference code, and the results comparison. This repo is for education purpose, where you can directly train the model and hopefully make some modification on it. 

## 2D Acoustic Wavefield Simulation using FNO 
The 2D acoustic wavefield simulation code is modified from the previous work by Yang et al., 2021. In this repo, we used the Gaussian random velocity model to test the performance of FNO, where we train the model using 20 random sources and generating the corresponding 2D wavefield snapshots. For the inference, we generate another 5 random sources. 

## 3D Acoustic Wavefield Simulation using Modified UFNO
As the 3D acoustic wavefield is harder than the 2D version, so we used the UFNO to perform the training. Compared to vanilla FNO model, UFNO integrates another U-Net branch in the FNO block to improve the feature representation capability in time domain of the model. However, when we turn to 3D, it requires large GPU memory to train it, to save the memory, we modified the UFNO by reducing the desen connection layers. By doing so, we can easily perform the training loop with a single NVIDIA RTX 3090 GPU. To extend this validation to full dynamic wavefield propagation, we implemented a 3D acoustic wavefield surrogate using a U-Net enhanced FNO (UFNO). Because the core dataset provides structural models rather than raw wavefields, target data were generated using the Deepwave finite-difference solver on a $96 \times 96 \times 96$ velocity model with 10~m grid spacing. The acoustic wave equation was solved using a fourth-order spatial discretization with 20-cell perfectly matched layer (PML) boundaries, driven by a 5~Hz Ricker wavelet source. Each simulation spanned 1.0~s, from which 20 spatiotemporal wavefield snapshots were uniformly sampled.

The UFNO surrogate was trained to map the static velocity model and source location directly onto the complete wavefield evolution, using 30 training sources and 5 unseen test sources. As illustrated in Figure \ref{fig:ufno-wavefields}, the data-driven model accurately predicts the spatiotemporal dynamics of the propagating wavefield. Rather than approximating a smoothed kinematic trend, the network reproduces the high-frequency dynamics of the acoustic wave equation, including phenomena such as diffraction and multipath. The errors remain low throughout the simulation, suggesting that the surrogate preserves both phase coherence and amplitude fidelity after the wavefront is scattered by the underlying structural heterogeneities.

# Reference
We utilized the FNO-torch 1.6.0 in this package:  https://github.com/zongyi-li/fourier_neural_operator/tree/master/FNO-torch.1.6

We also made some modifications using this package: https://github.com/yanyangg/AcousticNeuralOperator

We used Devito to generate the training samples: https://github.com/devitocodes/devito


Feel free to contact me if you have any questions: yang.cui512@gmail.com
