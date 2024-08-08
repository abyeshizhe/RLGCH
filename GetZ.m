function [Z ] = GetZ(V,U,sigma,s)
%得到B矩阵
[m,~]=size(U);
D = pdist2(V,U);
D1=exp(-D/(2*sigma*sigma));
K=sort(D1,2);
M=repmat(K(:,s),1,size(D1,2)) ;  % 按行复制最小值和原矩阵同维数
D1(D1>M)=1e-10 ;  
Zsum=sum(K(:,1:s),2);
Zsum1=repmat(Zsum,1,m);
Z=D1./Zsum1;





  


