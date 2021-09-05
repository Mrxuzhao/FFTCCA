%% Generating the dataset of left and right halves of MNIST images.
%% Dataset.
% Training data are named X1 and X2 for view 1/2.
% Tuning data are named XV1 and XV2 for view 1/2.
% Testing data are named XTe1 and XTe2 for view 1/2.
% All data matrices contains samples rowwise.
%
% Uncomment the following line if you want to generate the dataset.
% createMNIST;
clear
load mnist_all.mat
TRAIN={train1,train2,train3,train4,train5,train6,train7,train8,train9,train0};
TEST={test1,test2,test3,test4,test5,test6,test7,test8,test9,test0};

X1=[]; X2=[]; trainLabel=[];
XV1=[]; XV2=[]; tuneLabel=[];
XTe1=[]; XTe2=[]; testLabel=[];

NUMVALID=1000;  % 10x1000 samples for tuning/validation.
load demoseed.mat

for i=1:10
  TMP=TRAIN{i};
  rp=randperm(size(TMP,1));
  TMP=TMP(rp,:);
  for j=1:size(TMP,1)
    tmp=double(TMP(j,:))/255;
    tmp=reshape(tmp,28,28)';
    left=tmp(:,1:14);
    right=tmp(:,15:end);
    if j<=NUMVALID
      XV1=[XV1; left(:)'];
      XV2=[XV2; right(:)'];
      tuneLabel=[tuneLabel; i];
    else
      X1=[X1; left(:)'];
      X2=[X2; right(:)'];
      trainLabel=[trainLabel; i];
    end
  end
end

for i=1:10
  TMP=TEST{i};
  rp=randperm(size(TMP,1));
  TMP=TMP(rp,:);
  for j=1:size(TMP,1)
    tmp=double(TMP(j,:))/255;
    tmp=reshape(tmp,28,28)';
    left=tmp(:,1:14);
    right=tmp(:,15:end);
    XTe1=[XTe1; left(:)'];
    XTe2=[XTe2; right(:)'];
    testLabel=[testLabel; i];
  end
end

save MNIST.mat X1 X2 trainLabel XV1 XV2 tuneLabel XTe1 XTe2 testLabel randseed
