function [A1opt,A2opt,R1opt,R2opt,b1opt,b2opt,CORR_train,CORR_tune]=...
  KNOI(X1,X2,XV1,XV2,M1,SIGMA1,SEED1,M2,SIGMA2,SEED2,L,filename,...
  rcov,rho,batchsize,l2penalty,eta,momentum,maxepoch,randseed,KTYPE,USEGPU)
% [A1opt,A2opt,R1opt,R2opt,b1opt,b2opt,CORR_train,CORR_tune]=...
%   KNOI(X1,X2,XV1,XV2,M1,SIGMA1,SEED1,M2,SIGMA2,SEED2,L,filename,...
%   rcov,rho,batchsize,l2penalty,eta,momentum,maxepoch,randseed,KTYPE,USEGPU)
% implements the Kernel Nonlinear Orthogonal Iterations algorithm.
%
% Inputs
%   X1/X2: data matrices containing training samples rowwise.
%   XV1/XV2: data matrices containing tuning samples rowwise.
%   M1/M2: the lists of number of random samples for each view and each
%     kernel widths.
%   SIGMA1/SIGMA2: lists of Gaussian kernel widths used in each view,
%     i.e., k(x,y)=exp(-0.5*|x-y|^2/s^2).
%   SEED1/SEED2: lists of random seeds to generate features.
%   L: the final feature dimensionality.
%   filename: name of temporary file for saving intermediate results.
%   rcov: 2D vector containing (small) regularization to the diagonal of
%     covariance estimate.
%   rho: the time constant, in [0,1).
%   batchsize/l2penalty/eta/momentum/maxepoch: hyperparameters for
%     NOI optimization (appropriate values are eg. 2500/1e-5/1e-2/0.99/10).
%   KTYPE: determines the type of kernel to be used
%     0: additive (default),
%     1: multiplicative,
%    -1: linear kernel.
%
% Outputs
%   A1opt/A2opt: linear mapping from nonlinear features to R^L.
%   R1opt/R2opt, b1opt/b2opt: rotation and bias of the projections (by A1/A2)
%     to satisfy the whitening constraints.
%   CORR_train/CORR_tune: total correlation achieved on train/tune set
%     during training.

if ~exist('rcov','var') || isempty(rcov)
  rcov=[0 0];
else
  if numel(rcov)==1
    rcov=[rcov rcov];
  end
end

if ~exist('rho','var') || isempty(rho)
  rho=0;
end

if ~exist('batchsize','var') || isempty(batchsize)
  batchsize=2500;
end

if ~exist('l2penalty','var') || isempty(l2penalty)
  l2penalty=1e-4;
end

if ~exist('eta','var') || isempty(eta)
  eta=1e-2;
end

if ~exist('momentum','var') || isempty(momentum)
  momentum=0.9;
end

if ~exist('maxepoch','var') || isempty(maxepoch)
  maxepoch=10;
end

if ~exist('randseed','var') || isempty(randseed)
  randseed=0;
end

if ~exist('KTYPE','var') || isempty(KTYPE)
  KTYPE=0;
end

if ~exist('USEGPU','var') || isempty(USEGPU)
  USEGPU=0;
end

N=size(X1,1);

if ~exist(filename,'file')
  
  switch KTYPE,
    case 0, % Additive kernel.
      DA1=sum(M1); DA2=sum(M2);
    otherwise,
      DA1=M1(1);   DA2=M2(1);
  end
  
  %% Initialize the weight matrices.
  if USEGPU
    parallel.gpu.rng(randseed);
    A1=gpuArray.randn(DA1+1,L)*.1;
    A2=gpuArray.randn(DA2+1,L)*.1;
  else
    rng(randseed);
    A1=randn(DA1+1,L)*.1;
    A2=randn(DA2+1,L)*.1;
  end
  
  %% Initialize the mean and covariances.
  B0=batchsize;
  fprintf('initializing ...\n');  tic;
  rp=randperm(N);  idx=rp(1:B0);
  
  % Get view 1 random feature.
  Y1batch=getRandFeature(X1(idx,:),M1,SIGMA1,SEED1,KTYPE,USEGPU);
  Z1batch=Y1batch*A1;
  m1=mean(Z1batch,1);
  Z1batch=bsxfun(@minus,Z1batch,m1);
  SXX=(Z1batch'*Z1batch)/B0;  clear Y1batch Z1batch;
  
  % Get view 2 random feature.
  Y2batch=getRandFeature(X2(idx,:),M2,SIGMA2,SEED2,KTYPE,USEGPU);
  Z2batch=Y2batch*A2;
  m2=mean(Z2batch,1);
  Z2batch=bsxfun(@minus,Z2batch,m2);
  SYY=(Z2batch'*Z2batch)/B0;  clear Y2batchZ2batch;
  toc;
  
  CORR_train=-inf;  CORR_tune=-inf;
  delta1=0;  delta2=0;
  its=0; optvalid=-inf;
else
  load(filename,'CORR_train','CORR_tune','optvalid','delta1','delta2','m1','SXX','m2','SYY','A1','A2', ...
    'A1opt','A2opt','R1opt','R2opt','b1opt','b2opt');
  its=length(CORR_tune)-1;
end

numbatches=ceil(N/batchsize);

while its<maxepoch
  
  rng('shuffle');
  rp=randperm(N);
  fprintf('Epoch %d: \n',its+1);tic;
  
  for i=1:numbatches
    
    idx1=(i-1)*batchsize+1;
    idx2=min(i*batchsize,N);
    idx=[rp(idx1:idx2),rp(1:max(0,i*batchsize-N))];
    
    %% View 1 output.
    Y1batch=getRandFeature(X1(idx,:),M1,SIGMA1,SEED1,KTYPE,USEGPU);
    Z1batch=Y1batch*A1;
    m1=rho*m1+(1-rho)*mean(Z1batch,1);
    Z1batch=bsxfun(@minus,Z1batch,m1);
    SXX=rho*SXX+(1-rho)*(Z1batch'*Z1batch)/batchsize;
    
    %% View 2 output.
    Y2batch=getRandFeature(X2(idx,:),M2,SIGMA2,SEED2,KTYPE,USEGPU);
    Z2batch=Y2batch*A2;
    m2=rho*m2+(1-rho)*mean(Z2batch,1);
    Z2batch=bsxfun(@minus,Z2batch,m2);
    SYY=rho*SYY+(1-rho)*(Z2batch'*Z2batch)/batchsize;
    
    grad1=Y1batch'*(bsxfun(@plus,Z1batch,m1)-Z2batch*(SYY+rcov(2)*eye(L))^(-1/2));
    grad1=grad1/batchsize+l2penalty*A1;
    grad2=Y2batch'*(bsxfun(@plus,Z2batch,m2)-Z1batch*(SXX+rcov(1)*eye(L))^(-1/2));
    grad2=grad2/batchsize+l2penalty*A2;
    clear Y1batch Y2batch Z1batch Z2batch
    
    delta1=momentum*delta1-eta*grad1;
    delta2=momentum*delta2-eta*grad2;
    A1=A1+delta1;  A2=A2+delta2;
    
  end
  toc;
  
  its=its+1;
  
  if (mod(its,1)==0) || (its==maxepoch)
    % Compute correlations and errors.
    FX1=KNOI_forward(X1,M1,SIGMA1,SEED1,KTYPE,A1,[],[],USEGPU);
    FX2=KNOI_forward(X2,M2,SIGMA2,SEED2,KTYPE,A2,[],[],USEGPU);
    [R1,R2,b1,b2,corr]=linCCA(FX1,FX2,L);
    clear FX1 FX2;
    CORR_train=[CORR_train, sum(corr(1:L))];
    SIGN=sign(R1(1,:)+eps);
    R1=bsxfun(@times,R1,SIGN);  R2=bsxfun(@times,R2,SIGN);
    
    FXV1=KNOI_forward(XV1,M1,SIGMA1,SEED1,KTYPE,A1,R1,b1,USEGPU);
    FXV2=KNOI_forward(XV2,M2,SIGMA2,SEED2,KTYPE,A2,R2,b2,USEGPU);
    [~,~,corr_tune]=canoncorr(gather(FXV1),gather(FXV2));
    CORR_tune=[CORR_tune, sum(corr_tune)];
    clear FXV1 FXV2;
    
    if CORR_tune(end)>optvalid
      optvalid=CORR_tune(end);
      A1opt=A1;  A2opt=A2;
      R1opt=R1;  R2opt=R2;
      b1opt=b1;  b2opt=b2;
    end
    
    save(filename,'CORR_train','CORR_tune','optvalid','delta1','delta2',...
      'm1','SXX','m2','SYY','A1','A2', ...
      'A1opt','A2opt','R1opt','R2opt','b1opt','b2opt');
  end
end
