# create new NEP type specifically for our purpose
struct LR_SPMF_NEP <: NEP
    sum_nep::SPMFSumNEP
end

# global dict
musqr_dict = Dict() # for collecting musqr function values, avoids double comp

# need to implement size
import Base.size
function size(lr_spmf_nep::LR_SPMF_NEP)
    return size(lr_spmf_nep.sum_nep)
end
function size(lr_spmf_nep::LR_SPMF_NEP,dim)
    return size(lr_spmf_nep.sum_nep)[1]
end

# now overload compute_Mlincomb and compute_Mder
import NonlinearEigenproblems.NEPCore.compute_Mlincomb
function compute_Mlincomb(lr_spmf_nep::LR_SPMF_NEP,λ::Number,V::AbstractVecOrMat, a::Vector=ones(size(V,2)), startder::Int64=0)

    nep = lr_spmf_nep.sum_nep
    nep1=nep.nep1
    E = get_Av(nep1)[2]
    nep2=nep.nep2
    fv2=get_fv(nep2)

    m = size(nep2.L,1)

    # compute the derivatives using FDs 
    #ee=sqrt(eps());
    ee=1e-10
    musqrder = [λ -> (fv2[k](λ+ee)-fv2[k](λ-ee))/(2ee) for k in 1:m]

    if startder==0
        # we only have one nonzero coeff
        # nep2.L, nep2.U are vectors, use mapreduce
        if size(a,1) == 1

            full_rank_term = compute_Mlincomb(nep1,λ,V[:,1])
            low_rank_term = mapreduce((ff,ll,uu) -> ff(λ)*ll*(uu'*V[:,1]), +, fv2, nep2.L, nep2.U)

            return  a[1]*(full_rank_term+low_rank_term)
            
        # two nonzero coeffs
        elseif size(a,1) == 2

            # 0th derivative
            full_rank_term1 = compute_Mlincomb(nep1,λ,V[:,1])
            low_rank_term1 = mapreduce((ff,ll,uu) -> ff(λ)*ll*(uu'*V[:,1]), +, fv2, nep2.L, nep2.U)
 
            # 1st derivative
            full_rank_term2 = -E*V[:,2]
            low_rank_term2 = mapreduce((ff,ll,uu) -> ff(λ)*ll*(uu'*V[:,2]), +, musqrder, nep2.L, nep2.U)
            
            return a[1]*(full_rank_term1 + low_rank_term1) + a[2]*(full_rank_term2 + low_rank_term2)
        end
    # almost same as above, only 1st derivative this time
    elseif startder==1 

        full_rank_term = -E*V[:,1]
        low_rank_term = mapreduce((ff,ll,uu) -> ff(λ)*ll*(uu'*V[:,1]), +, musqrder, nep2.L, nep2.U)

        return a[1]*(full_rank_term + low_rank_term)
    end
end

import NonlinearEigenproblems.NEPCore.compute_Mder
# this should be used with care, creates full matrix (I think)
function compute_Mder(lr_spmf_nep::LR_SPMF_NEP,λ::Number,i::Int64=0)
    return compute_Mder(lr_spmf_nep.sum_nep, λ,i) # delegate
end


# add functionality for SMW-solve when we have SPMF-formated NEP
# global variable for num smw linsolves
num_smw_linsolves = 0
function smw_solve(A0,L,U,b)
    global num_smw_linsolves += 1
    z=A0\b
    A0bL=A0\L
    return z - A0bL*((I+U*A0bL)\(U*z))
end

struct SMWLinSolverCreator <: LinSolverCreator;
end

struct SMWLinSolver <: LinSolver;
  nep
  λ
end

import NonlinearEigenproblems.create_linsolver
import NonlinearEigenproblems.lin_solve
function create_linsolver(::SMWLinSolverCreator,nep,λ)
   return SMWLinSolver(nep,λ);
end

function lin_solve(solver::SMWLinSolver,b::Vector;tol=eps())

    nep1=solver.nep.sum_nep.nep1 # since we have a custom nep-type
    nep2=solver.nep.sum_nep.nep2
    λ=solver.λ;
    A0=compute_Mder(nep1,λ);
    fv2=get_fv(nep2)

    # mult each element by corr function eval
    Ltmp = [nep2.L[k]*fv2[k](λ) for k in 1:size(nep2.L,1)]

    # reshape to correct format
    Ltmp = mapreduce(x->x, hcat, Ltmp)
    Utmp = mapreduce(x->x, hcat, nep2.U)'

    return smw_solve(A0,Ltmp,Utmp,b)
end

# custom logger types to collect both NEP and NEPv iteration errors
struct DoubleErrorLogger <: Logger
    nep_errs::Matrix
    ll_approx::Matrix
    vv_approx::Matrix
    printlogger::Logger
end

# we only solve for one eig at a time
function DoubleErrorLogger(n,nof_eigvals=1,nof_iters=100,displaylevel=1)
    nep_s=Matrix{Float64}(undef,nof_iters,nof_eigvals)
    ll_s=Matrix{Float64}(undef,nof_iters,nof_eigvals)
    vv_s=Matrix{Float64}(undef,n,nof_iters) # eigvec approx as columns of matrix
    nep_s[:] .= NaN;
    ll_s[:] .= NaN;
    vv_s[:] .= NaN;
    printlogger=PrintLogger(displaylevel)
    return DoubleErrorLogger(nep_s,ll_s,vv_s,printlogger);
end

# import since we want to overload
import NonlinearEigenproblems.push_iteration_info!
import NonlinearEigenproblems.push_info!
function push_iteration_info!(logger::DoubleErrorLogger,level,iter;
                              err=Inf,λ=NaN,v=NaN,
                              continues::Bool=false)
    if (iter<=size(logger.nep_errs,1))
        if (size(err,1)<=size(logger.nep_errs,2))
            nep_err_vec=err;
            ll_err_vec=λ;

            if (nep_err_vec isa Number)
                nep_err_vec=[nep_err_vec]
            end
            if (ll_err_vec isa Number)
                ll_err_vec=[ll_err_vec]
            end
            logger.nep_errs[iter,1:size(err,1)]=nep_err_vec;
            logger.ll_approx[iter,1:size(err,1)]=ll_err_vec;
            logger.vv_approx[:,iter]=v;
        else
            if (logger.printlogger.displaylevel>1)
                println("Warning: Insufficient space in logger matrix");
            end
        end

    end

    push_iteration_info!(logger.printlogger,level,iter;err=err,λ=λ,v=v,continues=continues)
end

function push_info!(logger::DoubleErrorLogger,level::Int,
                    v::String;continues::Bool=false)
    push_info!(logger.printlogger,level,v,continues=continues)
end
