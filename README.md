# ATML_BetaVAE
Reproducibility Report for Beta-VAE
# Pose Adaptive Dual Mixup for Few-Shot Single-View 3D Reconstruction

This repository provides the source code for the Reproducibility Report for Beta-VAE.


## Self-Implementation and other's implementations

The beta-VAEs and conventional VAEsare self-implemented architectures. This includes the beta-VAEs under multiple datasets (dSprite, MNIST, FashionMNIST, 3DChairs) along with their designated dataloaders. 

DC-IGN is also self-implemented as there exists no PyTorch implementations.

InfoGAN is derived from the official implementation [here](https://github.com/Natsu6767/InfoGAN-PyTorch), and then trained by ourselves to obtain the results in the reproducibility paper.

Several Metrics are derived from the Disentanglement Metric Library [here](link) and then further changed by us to perform the evaluation.



## Datasets

Our evaluations ar performed under 4 datasets which are available below.

- dSprite: https://github.com/deepmind/dsprites-dataset
- MNIST and FashionMNIST: https://pytorch.org/vision/stable/datasets.html
- 3D Chairs: https://www.di.ens.fr/willow/research/seeing3Dchairs/
