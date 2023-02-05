This Matlab code implements the FFT-CCA algorithm described in the paper:  

**Quick start**:
- demo.m: demonstrates the application of FFT-CCA, KNOI, and GradKCCA on MNIST images.
  Use 60K/10K samples for training/validation.

**List of functions**:
- dataset
  - createMNIST.m: generating the dataset of left and right halves of MNIST images.
  - createMNIST2.m: another strategy for generating two views of the data.
- fftcca-master
  - fft_cca_rand_fixed.m: Large-Scale Canonical Correlation Analysis in Fourier Domain.
  - interfftcca.m: Invoking Functions provided for demo.
- gradKCCA-master
  - gradKCCA.m: the solution of "Large-Scale Sparse Kernel Canonical Correlation Analysis".
  - intergradKCCA.m: Invoking Functions provided for demo.
- knoi-master
  - KNOI.m: the Kernel Nonlinear Orthogonal Iterations algorithm.
  - KNOI_forward: applies KNOI projection mapping to test samples.
  - linCCA.m: the linear CCA algorithm.
  - getRandFeature: generates random Fourier features.
  - interknoi.m: Invoking Functions provided for demo.
External packages/data:
- mnist_all.mat: all MNIST images in MATLAB format can be downloaded from 
  webpage http://www.cs.nyu.edu/~roweis/data.html.
