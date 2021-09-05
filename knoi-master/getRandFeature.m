function Y=getRandFeature(X,M,SIGMA,SEED,KTYPE,USEGPU)
% Y=getRandFeature(X,M,SIGMA,SEED,KTYPE,USEGPU) generates random Fourier
%   features.
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
% 
% Outputs
%   Y: output feature matrix containing samples rowwise. I also rescale 
%      the random Fourier features by MYSCALE and add a constant dimension 
%      (bias) of 1.
%
% The scale works well with (U,V) initialization, which are sampled from 
% Gaussian of std 0.1.
MYSCALE=10;   

if ~exist('KTYPE','var') || isempty(KTYPE)
  KTYPE=0;
end

if ~exist('USEGPU','var') || isempty(USEGPU)
  USEGPU=0;
end

if USEGPU
  X=gpuArray(X);
end

[N,D]=size(X);

%% Additive kernel, concatenate the features.
if KTYPE==0
  
  if USEGPU
    Y=gpuArray.zeros(N,sum(M)+1);
  else
    Y=zeros(N,sum(M)+1);
  end
  
  idx=0;
  for i=1:length(SIGMA)
    m=M(i);  s=SIGMA(i);  seed=SEED(i);
    % Generate random features.
    if USEGPU
      parallel.gpu.rng(seed);  W=gpuArray.randn(D,m)/s;  b=2*pi*gpuArray.rand(1,m);
    else
      rng(seed);  W=randn(D,m)/s;  b=2*pi*rand(1,m);
    end
    Y(:,idx+1:idx+m)=cos(bsxfun(@plus,X*W,b))*sqrt(2/m)*MYSCALE;
    idx=idx+m;
    clear W b;
  end
  %% Constant bias.
  Y(:,end)=1;
  return;
end

%% Multiplicative kernel, add the samples.
if KTYPE==1   
  
  M=unique(M);
  if length(M)>1
    error('Multiplicative kernel has to use same #features for all widths\n');
  end
  
  if USEGPU
    Y=gpuArray.zeros(N,M+1);  W=gpuArray.zeros(D,M);
  else
    Y=zeros(N,M+1);  W=zeros(D,M);
  end
  for i=1:length(SIGMA)
    s=SIGMA(i);  seed=SEED(i);
    % Generate random features.
    if USEGPU
      parallel.gpu.rng(seed);  W=W+gpuArray.randn(D,M)/s;  b=2*pi*gpuArray.rand(1,M);
    else
      rng(seed);  W=W+randn(D,M)/s;  b=2*pi*rand(1,M);
    end
  end
  % Cosine transform.
  Y(:,1:end-1)=cos(bsxfun(@plus,X*W,b))*sqrt(2/M)*MYSCALE;
  % Constant bias.
  Y(:,end)=1;
  return;
end

%% Linear kernel.
if KTYPE==-1
  Y=[X, ones(N,1)]; return;
end