using LinearAlgebra
using DynamicPolynomials
#using TypedPolynomials

"""
    coeff_matrices(A::AbstractMatrix)

Return a vector of `(m × n)` numerical matrices containing the constant
coefficients and the coefficients of every variable that actually occurs
in `A`.

"""
function coeff_matrices(A::AbstractMatrix, vars)

    m,n = size(A)
    nv = length(vars)
    T = promote_type(map(coefficient_type, A)...)
    Ms = [zeros(T, m, n) for _ in 0:nv]         # one matrix per term

    for i in 1:m, j in 1:n
        p = A[i, j]
        # constant coefficient (monomial 1)
        Ms[1][i, j] =  p(vars => zeros(T,size(vars,1)))

        # linear terms (degree 1 in each variable)
        for (k, v) in pairs(vars)                      # k = 1:nv
            Ms[k + 1][i, j] = coefficient(p, v)        # monomial `v`
        end
    end

    return Ms
end

"""
    companion1(G::AbstractMatrix, vars::Vector{Monomial})

Computes the first companion linearization matrix from the manuscript. `G` is a numerical matrix corresponding to one evaluation of G(λ). `vars` is a vector containing the monomials `x1`,`x2`,...,`xm`. Returns a symbolic matrix `A` to be processed in `coeff_matrices`-function.
"""
function companion1(G,vars)

    s=size(vars,1)+1;
    p=sum(0.1.*vars);
    A=zeros(typeof(p),s,s);
    for j=2:s
        A[j,1]=vars[j-1]
        A[j,j]=-1;
    end

    # construct first row
    A[1,2:s] = G*vars
    A[1,1]=-1;

    return A;
end

"""
    companion2(Hrow::Vector, vars::Vector{Monomial}, xk::Monomial)

Computes the k:th non-normalization companion linearization matrix from the manuscript. `Hrow` is a numerical vector corresponding to the k:th row of one evaluation of H(λ). `vars` is a vector containing the monomials `x1`,`x2`,...,`xm`. `xk` is a monomial variable corresponding to the equation number (cf. manuscript). Returns a symbolic matrix `A` to be processed in `coeff_matrices`-function.
"""
function companion2(Hrow,vars,xk)

    p=sum(0.1.*vars); # for type-casting
    A=zeros(typeof(p),3,3); # these matrices are always 3×3

    alpha = Hrow'*vars

    A[diagind(A,0)[2:end]] .= -1
    A[diagind(A,-1)] .= alpha
    
    A[1,1]=-xk;
    A[1,end]=alpha

    return A;
end

"""
    compute_mep_matrices(G,H)

Compute the MEP matrices associated with one evaluation of G(λ), H(λ), represented by the numerical matrices `G` and `H`.
Returns a m×(m+1) matrix of matrices where one row corresponds to the matrices for one of the MEP equations. 
"""
function compute_mep_matrices(G,H)

    m = size(G,1)

    # vars = [x1,x2,...,xm]
    @polyvar(vars[1:m])

    # matrix of matrices
    W_full = Matrix{Matrix{Float64}}(undef,m,m+1)

    # first row is normalization
    A1=companion1(G,vars);
    M1 = coeff_matrices(A1,vars)
    M1[1]=-M1[1];
    M1=reshape(M1,1,m+1);
    W_full[1,:] = M1

    # fill out rest of eqns
    for rowno in 1:m-1
        Ak=companion2(H[rowno,:],vars,vars[rowno])
        Mk = coeff_matrices(Ak, vars)
        Mk[1]=-Mk[1]; # for compat with mep_solve
        Mk=reshape(Mk,1,m+1);
        W_full[rowno+1,:] = Mk
    end

    return W_full

end
