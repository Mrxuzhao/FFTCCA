function []=interfftcca(X_train, X_trlable,X_test,X_telable,Y_train,Y_test,radio)

%%
[nXSmp,dXSmp]=size(X_train);
X_train_fft= fft(X_train,[],2);
X_test_fft= fft(X_test,[],2);
%%
each=ceil(nXSmp*radio);

star=tic;
[vx,xlambda,vy,ylambda,ei] = fft_cca_rand_fixed(X_train,Y_train,dXSmp,each);%
send=toc(star);
X_train_redim=X_train_fft*vx;
X_test_redim=X_test_fft*vx;
disp('time:');
disp(send);
end