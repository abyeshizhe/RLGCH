function Hpi = fista(Vp, Ci, Xinit, opts)
    %% init
    opts = initOpts(opts);
    lambda = opts.lambda;

    if numel(lambda) > 1 && size(lambda, 2)  == 1
        lambda = repmat(opts.lambda, 1, size(Ci, 2));
    end
    if numel(Xinit) == 0
        Xinit = zeros(size(Vp,2), size(Ci,2));
    end
    
    %% norm1
    function res = norm1(X)
	    res = full(sum(abs(X(:))));
    end

    %% norm2
    function res = norm2(X)
	    res = sum(sqrt(sum(X.^2, 2)));
    end
    %% normF2
    function res = normF2(A)
        B = A.^2;
        B = B(:);
        res = sum(B);
	% res = norm(A, 'fro')^2;
    end 

    %% cost f
    function cost = calc_f(X)
        cost = 1/2 *normF2(Ci - Vp*X);
    end 
    %% cost function 
    function cost = calc_F(X)
        if numel(lambda) == 1 % scalar 
            cost = calc_f(X) + lambda*norm1(X);
        elseif numel(lambda) == numel(X)
            cost = calc_f(X) + norm1(lambda.*X);
        end
    end

    %% grad
    Vp_tVp = Vp'*Vp;
    Vp_tCi = Vp'*Ci;
    function res = grad(X) 
        res = Vp_tVp*X - Vp_tCi;
    end 

    %% fista_general
    L = max(eig(Vp_tVp));
    [Hpi, ~, ~] = fista_general(@grad, @proj_l1, Xinit, L, opts, @calc_F);
end