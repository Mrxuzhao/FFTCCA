function Y=KNOI_forward(X,M,SIGMA,SEED,KTYPE,A,R,b,USEGPU)
% Y=KNOI_forward(X,M,SIGMA,SEED,KTYPE,A,R,b,USEGPU) applies the KNOI 
%   projection mapping.
%
% Inputs
%   X: data matrices containing samples rowwise.
%   M: the lists of number of random samples for each kernel widths.
%   SIGMA: lists of Gaussian kernel widths k(x,y)=exp(-0.5*|x-y|^2/s^2).
%   SEED: lists of random seeds to generate features.
%   KTYPE: determines the type of kernel to be used
%     0: additive (default),
%     1: multiplicative,
%    -1: linear kernel.
%   A: projection matrix, (M+1)xL matrix.
%   R/b: linear transformation in R^L to satisfy the whitening constraint.
% 
% Outputs
%   Y: projection of X, containing samples rowwise. 

% Input dimensionality.
N=size(X,1);
% Projection dimensionality.
L=size(A,2);

if USEGPU
  Y=gpuArray.zeros(N,L);
else
  Y=zeros(N,L);
end

batchsize=2500;  % Adjust the batchsize to fit each minibatch in memory.
for i=1:ceil(N/batchsize)
  idx=batchsize*(i-1)+1:min(batchsize*i,N);
  Y(idx,:)=getRandFeature(X(idx,:),M,SIGMA,SEED,KTYPE,USEGPU)*A;
end
%% R and b does a rotation and shift in feature space R^L.
if ~isempty(R)
  Y=bsxfun(@minus,Y,b)*R;
end
