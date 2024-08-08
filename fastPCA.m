function [pcaA V] = fastPCA( A, k )
% Fast PCA
% Input ：A --- Sample matrix , One sample for each behavior 
% k --- Dimension reduction to k dimension 
% Output ：pcaA --- After the dimension reduction k A matrix composed of eigenvectors of dimensional samples , One sample per line , Number of columns k Is the dimension of sample feature after dimension reduction 
% V --- Principal component vector 
[r c] = size(A);
% Sample mean 
meanVec = mean(A);
% Calculate transposition of covariance matrix covMatT
Z = (A-repmat(meanVec, r, 1));
covMatT = Z * Z';
% Calculation covMatT Before k Eigenvalues and eigenvectors 
[V D] = eigs(covMatT, k);
% Get the covariance matrix (covMatT)' The eigenvector of 
V = Z' * V;
% The eigenvectors are normalized to unit eigenvectors 
for i=1:k
V(:,i)=V(:,i)/norm(V(:,i));
end
% linear transformation （ Projection ） Dimension reduction to k dimension 
pcaA = Z * V;
% Save the transformation matrix V And transform the origin meanVec