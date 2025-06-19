using Combinatorics # permutations iterator (Pkg.add("Combinatorics") first)
using Random, LinearAlgebra
"""
    sgn_perm(perm)

Return `+1` for an even permutation and `-1` for an odd one.
"""
function sgn_perm(perm) 
    invcount = 0
    for i = 1:length(perm)-1
        for j = i+1:length(perm)
            invcount += perm[i] > perm[j] ? 1 : 0   # count inversions
        end
    end
    return isodd(invcount) ? -1 : 1
end


"""
    opdet(W, i)

    Delta0=opdet(W_full,0);
    Delta1=opdet(W_full,1);
    Delta2=opdet(W_full,2);
    Delta3=opdet(W_full,3);
"""
function opdet(W, i)
    
    # Build WW = W with column i replaced (if i > 0)
    WW = copy(W[:, 2:end])              
    if i > 0
        WW[:, i] .= W[:, 1]             # replaced row
    end

    p = size(WW, 1)                    # number of equations

    nn=prod(maximum.(size.(WW)[:,1]))  # Product of all the sizes is the final size.
    
    Deltai = zeros(eltype(WW[1,1]), nn, nn)

    # Permutations version of the operator determinant 
    # Iterate over all permutations of 1:p without materialising the whole matrix
    for sigma in permutations(1:p)      # lazy iterator from Combinatorics
        tsgn = sgn_perm(sigma)

        tempdelt = tsgn * WW[1, sigma[1]]
        for j = 2:p
            tempdelt = kron(tempdelt, WW[j, sigma[j]])
        end
        Deltai .+= tempdelt     # keep result sparse
    end

    return Deltai
end

"""
    extract_eigvecs(A,ll)

Compute approximate eigvecs of MEP given an eigenvalue N-tuple `ll`=(λ1,...,λN). Returns a tuple of eigenvectors.
"""
function extract_eigvecs(A,ll)
    p=size(A,1);

    # Compute the corresponding eigenvectors via the smallest singular
    # value of the matrix.
    vv=Vector{Vector{typeof(ll[1])}}(undef,p)
    for j=1:p
        # MMj = -A[j,1]+ll[1]*A[j,2]+...ll[p]A[j,p+1]
        MMj=-A[j,1]+mapreduce(i-> ll[i]*A[j,i+1],+,  1:p) 
        FS=svd(MMj)
        vv[j]=FS.V[:,end]; 
    end
    vv=Tuple(vv);
    
    return vv
    
end

"""
    mep_solve(A;select_real_only=true)

Solve the MEP defined by the matrices in `A`. `A` is a p×(p+1) array where every element is a square matrix. The k:th row corresponds to the k:th equation in the MEP:

(-A(1,1)+s(1)*A(1,2)+...s(p)*A(1,p+1))*x1,
...
(-A(p,1)+s(1)*A(p,2)+...s(p)*A(p,p+1))*xp.

Returns a 2-tuple of vectors; (eigvals, eigvecs).

TODO: rename p to m for consistency.
"""
function mep_solve(A;select_real_only=true, discard_tol=1e10)

    p=size(A,1);

    # Create the Delta matrices
    Delta=map(i-> opdet(A,i), 0:p)
    ii=argmin(cond.(Delta[2:end]))+1

    # solve GEP corresponding to best choice of Delta-mats
    F=eigen(Delta[ii],Delta[1]);
    
    # Find wanted solutions 
    if (select_real_only)
        solnumber=findall((imag.(F.values) .== 0) .&& abs.(F.values) .< discard_tol)
    else
        solnumber=findall(abs.(F.values) .< Inf)
    end
    
    lambda_ii=real.(F.values[solnumber])
    z=real.(F.vectors[:,solnumber]);

    # Compute eigvals via RQ
    lambda1_to_p=Vector{typeof(lambda_ii)}(undef,p)
    for j=1:p
        lambda1_to_p[j]= [((z[:,k]'*Delta[1+j]*z[:,k])/(z[:,k]'*Delta[1]*z[:,k])) for k in 1:size(z,2)]
    end

    # Use values from eigen-call
    lambda1_to_p[ii-1] =lambda_ii
    
    # storage for sols
    lambdas = []

    # format eigvals correctly
    for k in 1:size(z,2)
        # Collect the eigenvalue N-Tuple 
        # ll=[lambda1[k]; lambda2[k]; [lambda3_to_p[j][k] for j=1:p-2]]
        ll=[lambda1_to_p[j][k] for j=1:p]
        ll=Tuple(ll);
        push!(lambdas, ll);
    end

    # Fetch all the eigvecs
    vectors=map(ll->extract_eigvecs(A,ll),lambdas)
    
    return (lambdas,vectors)
end

"""
    compute_mep_residual(A, lambdas, vectors)
 
Compute the residual-vector corresponding to the approximate eigen-tuples contained in `lambdas` and `vectors`. `lambdas` is a m-tuple containing the eigs, and `vectors` is a m-tuple containing the correspodning eigvecs. Returns an array containing the residuals of the m equations in the MEP.
"""
function compute_mep_residual(A, lambdas, vectors)

    m = size(A,1)
    res_vec = zeros(m)
    
    for j in 1:m
        MMj=-A[j,1]+mapreduce(i-> lambdas[i]*A[j,i+1],+, 1:m) 
        res_vec[j] = norm(MMj*vectors[j])
    end

    return res_vec

end

function single_refine_mep_sol(A,lambda,vectors)

    
end

function refine_mep_sol(A,lambdas,vectors)

end


#Random.seed!(-1);
#p=4
#A=Matrix{Any}(undef,p,p+1)
#for j=1:p
#    A[j,:]= [randn(4,4) for i=1:p+1]
#end
#A[1,:] = [randn(5,5) for i=1:p+1] 
#
#(lambdas,V)=mep_solve(A)
#
#res = compute_mep_residual(A,lambdas[2], V[2])
#
#eigval_number=4;
#for j=1:p
#    ll=lambdas[eigval_number];
#    # MMj = -A[j,1]+ll[1]*A[j,2]+...ll[p]A[j,p+1]
#    MMj=-A[j,1]+mapreduce(i-> ll[i]*A[j,i+1],+,  1:p) 
#    @show norm(MMj*V[eigval_number][j])
#end

