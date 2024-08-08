function [Hp, step_results] = RLGCHClustering(C,step_results, para)

%  input  Data  cell nview*1      n*dim  
%  input  Lapmatrix cell nview*1  n*n
iter = para.iter;
A1 = para.A1;
A2 = para.A2;
lambda = para.lambda; 
L = para.L;  
mu = para.mu;
Vp = para.Vp;
% initialize L
% L = GlobalLapmatrix(Lapmatrix,beta,R);

% initialize Hp
% [idx_U, A, evs] = CAN(C, c);
% for i=1:length(idx_U)
%     Hp(i,idx_U(i))=1;
% end
% Hp=Hp+0.2; 

Hp = zeros(size(Vp, 2), size(C, 2));
   
pho = 1.5;

function res = normF2(A)
    B = A.^2;
    B = B(:);
    res = sum(B);
    % res = norm(A, 'fro')^2;
end
function cost = calc_f(X, M)
   % X = sym(X);
   % M = sym(M);
    cost = normF2(C- Vp*X) + lambda * trace(X'*M*X);
end 
for i=1:1:iter
    
    % update E
    temp_e = C - (Vp*Hp)+1/mu*A1;
    [E]=L21_solver(temp_e,1/mu);
    E = real(E);
    
    % updata Z
    Z = max(Hp + A2/mu - lambda/mu * Vp'*L*Vp*Hp, 0);
    
    % updata Hp
    Hp = max(((C - E + A1/mu)'*Vp)' + A2/mu + Z - (lambda/mu)*(Vp'*L*Vp)*Z, 0);
    
    % update A1, A2
    A1 = A1 + mu*(C - Vp * Hp - E);
    A2 = A2 + mu*(Z - Hp);
    
    % update mu
    mu = min(pho*mu,10^8);
   % if isnan(calc_f(Hp, Vp'*L*Vp))%||isinf(calc_f(Hp, Vp'*L*Vp))
    %    step_results(i) = step_results(i-1);
    %else
    step_results(i) = calc_f(Hp, Vp'*L*Vp);
end
end
