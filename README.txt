Paper: Time and Memory Efficient Large-Scale Canonical Correlation Analysis in Fourier Domain.

Author: Xiang-Jun Shen, Jiangsu University Zhenjiang, China, xjshen@ujs.edu.cn;
  Liangjun Wang, Jiangsu University Zhenjiang, China, ljwang0819@ujs.edu.cn;
  Zhaorui Xu, Jiangsu University Zhenjiang, China, zhaorxu@stmail.ujs.edu.cn;
  Zechao Li, Nanjing University of Science& Technology Nanjing, China, zechao.li@njust.edu.cn.
  
Conference Name: MM '22: The 30th ACM International Conference on Multimedia Proceedings.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
MM ’22, October 10–14, 2022, Lisboa, Portugal
© 2022 Association for Computing Machinery. 
ACM ISBN 978-1-4503-9203-7/22/10
https://doi.org/10.1145/3503161.3547988

This Matlab code implements the FFT-CCA algorithm described in the paper:

Quick start:
- demo.m: demonstrates the application of FFT-CCA, KNOI, and GradKCCA on MNIST images.
  Use 60K/10K samples for training/validation.

List of functions:
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
