using LinearAlgebra
using SparseArrays
using NonlinearEigenproblems
using LowRankApprox

include("nepv_core.jl");
include("solve_polysys_mep.jl")

# Workaround for https://github.com/nep-pack/NonlinearEigenproblems.jl/issues/269
import NonlinearEigenproblems.LowRankFactorizedNEP
function NonlinearEigenproblems.LowRankFactorizedNEP(L::AbstractVector{S}, U::AbstractVector{S},f) where {S<:AbstractMatrix}
    A=map(i-> LowRankMatrix(L[i],U[i]),1:size(L,1))
    rnk = mapreduce(u -> size(u, 2), +, U)
    spmf=SPMF_NEP(A, f, align_sparsity_patterns=false, check_consistency=false);
    return LowRankFactorizedNEP(spmf,rnk,L,U)
end

"""
    musqr_num(nepv,λ,target,newton_iters)

Compute μ^2 given `λ` and `target` by solving the associated MEP from the manuscript.
Selects the branch closest to `target`.
`newton_iters` determines the number of refinement steps in the MEP-solver.
"""
# global variable for logging number of evals of g,h
num_GH_evals = 0
function musqr_num(nepv,λ,target,newton_iters)

    A0=nepv.A0;
    Av=nepv.Av;
    n=size(A0,1);
    B=nepv.B;
    E=nepv.E

    # compute transfer functions
    interm = (λ*E-A0)\Av
    G = interm'*(B*interm)
    H = (Av'*interm)

    # log 
    global num_GH_evals += 1

    musqr = solve_polysys_mep(G,H)
    #num_iters = 2
    newtonrefine_mep_sols!(G,H,musqr,newton_iters)
    
    # Pick the solution closest to the target
    distvec=map(mu-> norm([mu...]-target), musqr)
    musqr = musqr[argmin(distvec)];

    return [musqr...]

end

"""
    mu_j_sqr(j,nepv_approx,λ,target,newton_iters)

Adds functionality for pushing to a global dictionary for avoiding multiple computations of same quantity.
Assumes there is a global variable `musqr_dict` within scope.
`j` refers to the `j`th nonlinear function. `newton_iters` determines the number of refinement steps in the MEP-solver.
"""
function mu_j_sqr(j,nepv_approx,λ,target,newton_iters)
    # musqr_dict is global 
    if !haskey(musqr_dict, λ)
        musqr_dict[λ] = musqr_num(nepv_approx,λ,target,newton_iters)
    end
    return musqr_dict[λ][j]
end

"""
    m_terms(nepv,target;nepv_approx=nepv,spmf=false,newton_iters=2)

Set-up the NEP `nep` corresponding to `nepv`.
`target` determines how to choose among NEPs in the cases where the functions become multi-valued.
Selects function-values closest to `target`.
Use `nepv_approx` if mu_1,...,mu_m should be computed on a lower resolution grid.
If `spmf` is set to `true`, the NEP returned is of SPMF-type, and low-rank factorized.
`newton_iters` determines the number of Newton-polishing steps that should be taken in the MEP-solver.
"""
function m_terms(nepv::NEPv,target;nepv_approx=nepv,spmf=false,newton_iters=2)

    A0=nepv.A0;
    Av=nepv.Av
    n=size(A0,1);
    m=size(Av,2);
    B=nepv.B;
    E=nepv.E;

    if spmf
        # full-rank part
        nep1=SPMF_NEP([A0,sparse(E,n,n)], [s->one(s) , s-> -s])
        
        # nonlinear functions
        fv = [(s -> mu_j_sqr(j,nepv_approx,s,target,newton_iters)) for j in 1:m]
        
        # low-rank factors
        L = [reshape(nepv.Av[:,k],n,1) for k in 1:m]
        U = deepcopy(L)
        
        # low-rank part
        nep2= LowRankFactorizedNEP(L,U,fv)
        
        # full nep
        nep=SumNEP(nep1,nep2)

        return nep

    else
        # TODO
        error("Not implemented for m>2, use spmf instead")
    end

end


## example usage
#include("nep_helpers.jl") # for the NEP-functionality
#
## Setup nepv-problem
#A0=[
#    3.0 1 0 0 0 0;
#    1 2 0.5 0 0 0;
#    0 0.5 4 0.4 0 0;
#    0 0 0.4 7 0.5 0;
#    0 0 0 0.5 1 2;
#    0 0 0 0 2 0.4;
#]
#A0 = sparse(A0)
#n = size(A0,1)
#
#a1=[1.0;0.0;0.0;0.0;0.0;0.0];
#a2=[0.0;1.0;0.0;0.0;0.0;0.0];
#a3=[0.0;0.0;1.0;0.0;0.0;0.0];
#a4=[0.0;0.0;0.0;1.0;0.0;0.0];
#a5=[0.0;0.0;0.0;0.0;1.0;0.0];
#Av=[a1 a2 a3 a4 a5];
#nepv=NEPv(A0,Av,I,I,size(A0,1))
#
## target for poly solver
#target = [0.5,0.5,0.5,0.5,0.5]
#
## setup the nep
#tmp_nep=m_terms(nepv,target,spmf=true)
#
## we want our custom NEP type
#nep = LR_SPMF_NEP(tmp_nep)
#
## use smw...
#linsolvercreator=SMWLinSolverCreator()
#
## ...and residual errmeasure
#errmeasure=ResidualErrmeasure(nep)
#
## solve NEP
#λ0 = 8.0
#x0=ones(n)
#tol = 1e-14
#linsolver = create_linsolver(linsolvercreator,nep,λ0)
#x0=normalize!(lin_solve(linsolver,x0,tol=tol))
#x0=normalize!(lin_solve(linsolver,x0,tol=tol))
#λ,v = try
#    augnewton(Float64,nep,λ=λ0,v=x0, logger=1, maxit=30, tol=tol, armijo_factor=0.5, linsolvercreator=linsolvercreator, errmeasure=errmeasure);
#catch noconv
#    noconv.λ, noconv.v;
#end
#
#normalize!(v)
#
## should be zero
#@show norm(nepv(v,v)-λ*v)

