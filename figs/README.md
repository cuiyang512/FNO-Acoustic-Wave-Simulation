
## Data Preparation
We extracted 2D velocity maps from the 3D [OpenFWI dataset](https://openfwi-lanl.github.io/docs/data.html#vel) and randomly shuffled the indices along the first dimension. Wavefields were simulated using the acoustic wave equation with second-order spatial and temporal accuracy, implemented via the Devito Python library. An absorbing boundary condition of 50 layers was applied, with a time step (dt) of 4 ms. Wavefields with dominant frequencies of 3 Hz, 5 Hz, and 10 Hz were synthesized.

![Data Preparation Workflow](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/tree/main/figs/data_prep.png)

## Shooting Setup
As shown in the figure below, multiple receivers are positioned on the surface, while sources are randomly placed within the velocity model.

![Velocity Model Example](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/tree/main/figs/vel_model_Style_A_model58_f_5.0Hz_nbl_50_loc_190.0_20.0.png)

## Wavefield Example
The following figure displays snapshots of the wavefield with a dominant frequency of 5 Hz.

![Wavefield Example](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/tree/main/figs/wavefield_Style_A_model58_f_5.0Hz_nbl_50_loc_190_20.png)
