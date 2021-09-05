function [eigvectorX,Xlambda,eigvectorY,Ylambda,endi] = fft_cca_rand_fixed(X,Y,dSmp,step,eff)
%FFTcca can seek projections by choosing some discriminative Fourier bases,
%which are predefined in advance.
%   Objective Function:
%       -
%   Input:
%       -X: training data from view X
%       -Y: training data from view Y
%       -dSmp: the dimension of the basis
%       -step: the batch size
%       -eff: the control factor of threshold r
%   Output
%       -eigenvectorX: the eiengvectors of X
%       -eigenvectorY: the eigenvectors of Y
%       -Xlambda: the eiengvectors of X
%       -Ylambda: the eigenvectors of Y
%       -endi£ºthe total of inputted samples
[nSmp, dXSmp] = size(X);
[~, dYSmp] = size(Y);

% mv=mean(X); %
% X=(X-repmat(mv,nSmp,1)); %
% mv=mean(Y);
% Y=(Y-repmat(mv,nSmp,1));

d = [1:nSmp]';
endi=0;
narginchk(2, 5);
if nargin < 3 ||dXSmp>dSmp||dYSmp>dSmp
    dSmp=max(dXSmp, dYSmp);
    if dXSmp<dSmp
        X=[X,zeros(nXSmp,dSmp-dXSmp)];
    end
    if dYSmp<dSmp
        Y=[Y,zeros(nYSmp,dSmp-dYSmp)];
    end
end 
if nargin < 4
   step = 100;
end

if nargin < 5 
    eff=0.9;
end
nargoutchk(2, 6);
%% Cca
cum_old=zeros(1,dSmp);
cum_oldindex=zeros(1,1);
iternum=fix(nSmp/step);

for i=1:iternum
    rp=randperm(length(d),step);%
    rdom=d(rp);%
    Xiter=fft(X(rdom,:),[],2);
    XEiter=conj(Xiter);
    Yiter=fft(Y(rdom,:),[],2);
    YEiter=conj(Yiter);
    xxdiagsum=sum((Xiter.*XEiter));
    yydiagsum=sum((Yiter.*YEiter));
    xydiagsum=sum((XEiter.*Yiter));
    vsum=(1./xxdiagsum).*xydiagsum.*(1./yydiagsum).*conj(xydiagsum);
    cum_new=cum_old+vsum;
    [cum_sort,cum_newindex]=sort(cum_new,'descend');
    cum_newindex=cum_newindex(1:selectenergy(cum_sort,eff));
    
    if  length(setdiff(cum_newindex,cum_oldindex))<length(cum_newindex)*0.05
%         disp(i*step);
        endi=i*step;
        break;
    else
        cum_old=cum_new;
        cum_oldindex=cum_newindex;  
        d = setdiff(d, rdom);
    end
end  
 %[cum_sort,cum_newindex]=sort(cum_new,'descend');
eigvalueX=diag(cum_new);

% Eigen Eq.
% K = (X*X')+0.05*eye(size(dimMin));
XF=transpose(dftmtx(dSmp))/sqrt(dSmp);


% sort eigvalues
[eigvectorX, eigvalueX] = sort_eigvalues(XF,eigvalueX);

% Hyperparameter
Xlambda = sqrt(eigvalueX);  %
Xlambda = Xlambda/(max(Xlambda));
Ylambda=Xlambda;
eigvectorY=eigvectorX;
% Return to v
%v =  real(ifft(eigvectorX,[],1));
 
end
function [k]=selectenergy(E,eff)
% ---------------Calculated contribution rate---------------%
ratio=0;
n=length(E);
for k=1:n
    r=E(k)/sum(E);   
    ratio=ratio+r;  
    if(ratio>=eff) 
        break;
    end
end
end