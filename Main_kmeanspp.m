clc
clear all
close all
addpath(genpath('utils'))
addpath(genpath('dataset'))
%% 读入数据
% load MnistData_10_uni
dataset_n=20; %18 19 20 21 30 32
datasetslist = dir('dataset/*.mat');
%datasetslist = ["Cora_ML_uni", "WebKB_cornell_uni", "WebKB_texas_uni", "WebKB_washington_uni", "WebKB_wisconsin_uni", "MnistData_10_uni.mat","cars_uni", "crx_uni", "lenses_uni"];
%load(datasetslist(dataset_n), '-mat');
datasetslist(dataset_n).name
load(datasetslist(dataset_n).name);
format short
% load TDT2_10_uni.mat
%% 数据归一化
X =mapstd(X);
[a,b]=size(X);
% noisy = 0.2*random('Poisson',1,a,b);
% X = double(X)+noisy;
% X=imnoise(X,'speckle',0.2);
%X =mapstd(X_person2');
%Y = Y_person2;
ACC=[];NMI=[];Purity=[];T=[];JJ=[];
AR =[];RI=[];MI=[];HI=[];F_1=[];P=[];R=[];
RESULT=[];
%% 初始化及定义变量
for i=1:10
    S = length(unique(Y));       % the number of classes
[N,~] = size(X);             % the number of samples (N) and dimension (D)
tic
%% First Step: C(n*k) obtained by k-means
%[~, C] = litekmeans(X, S);   % C is the initial indicator matrix
% 
% [pc,score,latent,tsquare]=pca(X);
% index=cumsum(latent)./sum(latent);
% rdim=find(index>0.80,1)
% pcaX=score(:,1:rdim);
[pcaX,~] = fastPCA(X, 604);

% [label, center, bCon, sumD, Distance] = litekmeans(pcaX, S, 'MaxIter', iteration);   % C is the initial indicator matrix
% % from Distance calculate C
% [maxv,ind]=min(Distance,[],2);
% ind = ind';
% C = zeros(N, S);
% for i = 1 : N
%     C(i, ind(i)) = 1;
% end
[Lab,~]=kmeansplusplus(pcaX',S);
% [Lab,~] = litekmeans(pcaX, S, 'MaxIter', iteration);
C=sparse(1:N,Lab,1,N,S,N);
[maxv1,ind1]=max(C,[],2);
Result= ClusteringMeasure(Y, ind1);
RESULT=[RESULT;Result];
t=toc;
[Result t]
% dlmwrite('TDT2_20.txt',[lambda,Result,t],'-append','delimiter','\t','newline','pc');
RESULT=[RESULT;Result];T=[T,t];
end

record=[mean(RESULT(:,1)),std(RESULT(:,1));
 mean(RESULT(:,2)),std(RESULT(:,2));
 mean(RESULT(:,3)),std(RESULT(:,3));
 mean(RESULT(:,4)),std(RESULT(:,4));
 mean(RESULT(:,5)),std(RESULT(:,5));
 mean(RESULT(:,6)),std(RESULT(:,6));
 mean(RESULT(:,7)),std(RESULT(:,7));
 mean(T),std(T)];
record = record';