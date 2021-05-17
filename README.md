# RotationAnalysis
Functions for analyzing noisy rotation time series (angle vs. time) with random
changes in direction of the rotation. Employs a deep learning based method (onedimensional
UNet) for semantic segmentation of time series into (counter-)clockwise segments and random
noise. Training data are synthesized via a linear model with capabilities for incorporating
different types of noise and changes in direction.

![alt text](https://github.com/adrianfessel/RotationAnalysis/blob/main/synthetic_labels_original.png?raw=true)

![alt text](https://github.com/adrianfessel/RotationAnalysis/blob/main/synthetic_unlabeled.png?raw=true)

![alt text](https://github.com/adrianfessel/RotationAnalysis/blob/main/synthetic_labels_recovered.png?raw=true)

![alt text](https://github.com/adrianfessel/RotationAnalysis/blob/main/synthetic_fits.png?raw=true)

![alt text](https://github.com/adrianfessel/RotationAnalysis/blob/main/text.png?raw=true)