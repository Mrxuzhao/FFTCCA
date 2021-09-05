function [ ]=interknoi(Xtrain,trainLabel,Xtest,testLabel,Ytrain,Ytest,dim,XV1,XV2)
narginchk(7, 9);
if nargin < 8
    XV1=Xtest;
end
if nargin < 9
    XV2=Ytest;
end
X1=Xtrain;
X2=Ytrain;
XTe1=Xtest;
XTe2=Ytest;

[nXSmp,dXSmp]=size(X1);
[nXTSmp,~]=size(XTe1);
%% Final CCA dimension.
%if fix(nXSmp*0.1)<50
%    K=fix(nXSmp*0.1);
%else
%    K=50;
%end
K=dim;
%% Regularizations for each view.
%rcov1=1e-4; rcov2=1e-4;
% rcov1=0.01; rcov2=100;
%% Random seed for replicating experiments.
randseed=0;
%% KNOI: kernel nonlinear orthogonal iterations.
% Number of random Fourier features to use in each view.
M1=50000;  M2=50000;
% Kernel widths.
s1=5;  s2=5;
% Time constant.
rho=0;
% Size of minibatches used by the algorithm.
batchsize=500;
% l2 regularization for projection matrices.
l2penalty=0.0005;
% Learning rate, momentum, max number of epochs.
eta=0.01;  momentum=0.99;  maxepoch=5;
% Set USEGPU to zero for running on CPU.
USEGPU=0;
% Seeds for generating random Fourier features in each view.
SEED1=1;  SEED2=2;

filename=['result_MNIST_randseed=' num2str(randseed) ...
  '_M1=' num2str(M1) '_M2=' num2str(M2) '_s1=' num2str(s1) '_s2=' num2str(s2) ...
  '_rho=' num2str(rho) '_batchsize=' num2str(batchsize) ...
  '_l2penalty=' num2str(l2penalty) '_eta=' num2str(eta) ...
  '_momentum=' num2str(momentum) '_maxepoch=' num2str(maxepoch) '.mat'];
% if exist(filename,'file')
%     delete(filename);
% end

asta=tic;
[A1opt,A2opt,R1opt,R2opt,b1opt,b2opt,CORR_train,CORR_tune]=KNOI(X1,X2,XV1,XV2,...
  M1,s1,SEED1,M2,s2,SEED2,K,filename,...
  [0 0],rho,batchsize,l2penalty,eta,momentum,maxepoch,randseed,0,USEGPU);
aend=toc(asta);
X1proj=KNOI_forward(X1,M1,s1,SEED1,0,A1opt,R1opt,b1opt,USEGPU);
XTe1proj=KNOI_forward(XTe1,M1,s1,SEED1,0,A1opt,R1opt,b1opt,USEGPU);
XTe2proj=KNOI_forward(XTe2,M2,s2,SEED2,0,A2opt,R2opt,b2opt,USEGPU);
end