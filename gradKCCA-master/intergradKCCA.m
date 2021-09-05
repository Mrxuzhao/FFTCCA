function [ ]=intergradKCCA(Xtrain,trainLabel,Xtest,testLabel,Ytrain,Ytest,M)
X_trlable=trainLabel;
X_telable=testLabel;
%% gradKCCA
degree = 2; % 1 - linear kernel % 2 - quadratic kernel
hyperparams.normtypeX = 1; % norm constraint on u
hyperparams.normtypeY = 1; % norm constraint on v
hyperparams.Cx = 1; % value of the norm constraint
hyperparams.Cy = 1; % value of the norm constraint
hyperparams.Rep = 15; % number of repetitions
hyperparams.eps = 1e-10; % stoppin criterion
hyperparams.degree1 = degree; % degree of the polynomial kernel
hyperparams.degree2 = degree; % degree of the polynomial kernel
astr=tic;
[u1,v1] = gradKCCA(Xtrain,Ytrain,M,hyperparams);
aend=toc(astr);
X_train=(Xtrain * u1).^degree;
X_test=(Xtest * u1).^degree;    
end