clc
clear all
close all
addpath(genpath('utils'))
addpath(genpath('dataset'))
%% 读入数据
dataset_n=1; 
datasetslist = dir('dataset/*.mat');
datasetslist(dataset_n).name
load(datasetslist(dataset_n).name);
% format short
%% 数据归一化
X =mapstd(X);
[a,b]=size(X);
ACC=[];NMI=[];Purity=[];T=[];JJ=[];
AR =[];RI=[];MI=[];HI=[];F_1=[];P=[];R=[];
RESULT=[];
%% 初始化及定义变量
for i=1:5
S = length(unique(Y));       % the number of classes
[N,~] = size(X);             % the number of samples (N) and dimension (D)
M =S+1;                 % the number of anchors                        
delta = 20;                  % the bandwidth of Gaussian kernel function
K = 5;                     % the number of K neighbours of KNN graph
p=5;
iteration = 200;
lambda_1 = 1e-2;
para_record = [M,K,p,iteration,lambda_1];
tic
%% First Step: C(n*k) obtained by k-means
[pcaX,~] = fastPCA(X, 300);

[Lab,~]=kmeansplusplus(pcaX',S);

C=sparse(1:N,Lab,1,N,S,N);
size(C, 2);
%% Second Step: Clustering refinement over the initial results
[label2,~]=kmeansplusplus(pcaX',M);
E = sparse(1:N,label2,1,N,M,N);  % transform label into indicator matrix
U = full((E*spdiags(1./sum(E,1)',0,M,M))'*X);    % compute center of each cluster
[ Z ] = GetZ(X,U,delta,K);   % anchor graph
D = diag(sum(Z,1));          % degree matrix
Z1 = Z*D^0.5;                % weighted matrix W = Z1*Z1'
[P,V,Q] = svd(Z1);           % svd decompose for Z1
Vp = Z1*Q(1:p,:)'*(V(1:p,1:p)^-1);         % compute Vp

%% compute Hp
% Initialising parameters
para.iter =5;
para.lambda = lambda_1;
para.mu = 10^(-3);
para.A1 = rand(N, S);
para.A2 = rand(p, S);

W_ = Z1*Z1';
D_ = diag(sum(W_, 2));
L_ = D_ - W_;
L_ = D_^(-1/2)*L_*D_^(-1/2);

para.L = L_;
para.c = 300;
para.Vp = Vp;
step_results = zeros(para.iter);
[Hp, ~] = RLGCHClustering(C,step_results, para);
%% compute new Y
new_Y = Vp*Hp;
% t=toc;
%% evaluate
[maxv1,ind1]=max(C,[],2);
%[acc1,nmi1,purity1 ] = ClusteringMeasure(Y, ind1);
%% perpare data
[maxv2,ind2]=max(new_Y,[],2);
%%

Result = ClusteringMeasure(Y, ind2);
t=toc;
[Result t]
% dlmwrite('TDT2_20.txt',[lambda,Result,t],'-append','delimiter','\t','newline','pc');
RESULT=[RESULT;Result];T=[T,t];
end

