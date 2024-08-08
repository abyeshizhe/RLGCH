function [Z ] = GetZ(V,U,sigma,s)
%�õ�B����
[m,~]=size(U);
D = pdist2(V,U);
D1=exp(-D/(2*sigma*sigma));
K=sort(D1,2);
M=repmat(K(:,s),1,size(D1,2)) ;  % ���и�����Сֵ��ԭ����ͬά��
D1(D1>M)=1e-10 ;  
Zsum=sum(K(:,1:s),2);
Zsum1=repmat(Zsum,1,m);
Z=D1./Zsum1;





  


