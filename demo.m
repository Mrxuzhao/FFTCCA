rootdirect='./';
addpath(genpath([rootdirect 'gradKCCA-master']));
addpath(genpath([rootdirect 'knoi-master']));
addpath(genpath([rootdirect 'fftcca-master']));
%% MnistÊý¾Ý¼¯
load MNIST.mat
X_train=X1;
X_trlable=trainLabel;
Y_train=X2;
Y_trlable=trainLabel;
X_test=XTe1;
Y_test=XTe2;
X_telable=testLabel;

interfftcca(X_train, X_trlable,X_test, X_telable,Y_train,Y_test,0.01);
interknoi(X_train, X_trlable,X_test,X_telable,Y_train,Y_test,50);
intergradKCCA(X_train, X_trlable,X_test,X_telable,Y_train,Y_test,30);