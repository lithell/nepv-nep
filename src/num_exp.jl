using LinearAlgebra
using NonlinearEigenproblems
using Random
using SparseArrays
using Plots
using DelimitedFiles
using Printf

include("numexp_nepv_setup.jl")
include("nep_helpers.jl")
include("m_terms.jl")
include("connect_curves.jl")

# Helper function for padding data with zeros (see end of file)
function pad_matrix(mat)
    n,m = size(mat)
    padded = [zeros(1,m+2); zeros(n,1) mat zeros(n,1); zeros(1,m+2)]
    return padded
end

# Helper function for computing NEPv error history
function compute_nepv_err_hist(nepv, vv_hist, ll_hist; dnep=Nothing, deflated=false) 

    # for later
    λ = ll_hist[end]

    # number of approximations
    ll_hist = ll_hist[ll_hist .> -Inf] # everything else is NaN
    num_approx = size(ll_hist,1)
    vv_hist = vv_hist[:,1:num_approx]
    
    # normalize
    for k = 1:num_approx
        vv_hist[:,k] = vv_hist[:,k]/sqrt(vv_hist[:,k]'*nepv.B*vv_hist[:,k])
    end

    # nep is not deflated yet, compute in simple manner
    if !deflated

        err_hist = [norm(nepv(vv_hist[:,k],vv_hist[:,k])-ll_hist[k]*nepv.E*vv_hist[:,k])/norm(vv_hist[:,k]) for k in 1:num_approx]

        return err_hist

    # nep has been deflated, need som extra steps
    else 

        err_hist = zeros(num_approx)

        for k in 1:num_approx

            dnep_tmp=deflate_eigpair(dnep,ll_hist[k],vv_hist[:,k]);

            (λv,X)=get_deflated_eigpairs(dnep_tmp)

            J=argmin(abs.(ll_hist[k] .-λv))
            ll=λv[J]; vv=X[:,J];
            vv=vv/sqrt(vv'*nepv.B*vv);

            err_hist[k] = norm(nepv(vv,vv)-ll*nepv.E*vv)/norm(vv)

        end

        return err_hist

    end

end

#=
This program tries to use Effenberger deflation to solve the 2D problem from the manuscript. 
=#

Random.seed!(0)

# setup nepv problem
n=256
h = 2/(n+1); # (-1,1)×(-1,1)
xvec = (-1+h):h:(1-h);

potential = :harmonic_lattice
c_vec = [45, 45, 45, 45, 45] # size of these determines m
sigma_vec = [6, 6, 6, 6, 6]
center_vec =[
    [0.4,-0.6],
    [0.6,0.3],
    [0.1,0.6],
    [-0.5,0.4],
    [-0.4,-0.4]
    ]

nepv = discretize_mterm_nepv(n, c_vec, sigma_vec, center_vec, potential)


# uncomment this if you want to solve the NEP, otherwise just visualization of curves

# save convergence history and timings
nep_conv_hist = Float64[]
nepv_conv_hist = Float64[]
timings = Float64[]

# setup nep
# target for poly solver
target = [0.5,0.5,0.5,0.5,0.5]

# setup the nep
tmp_nep=m_terms(nepv,target,spmf=true,newton_iters=3)

# we want our custom NEP type
nep = LR_SPMF_NEP(tmp_nep)

# initial guesses for eigvals
λ_init_vec = [92, 95, 105, 109, 120, 120, 125, 125, 135]

# storage for computed solutions
eigvecs = zeros(size(nep,1), size(λ_init_vec,1))
eigvals = zeros(size(λ_init_vec))

tol = 5e-12

for (prob_no,λ0) in enumerate(λ_init_vec)

    # first time we solve it
    if prob_no==1

        nn = size(nep,1)

        # use smw...
        global linsolvercreator=SMWLinSolverCreator()
        # ...and custom logger...
        global logger=DoubleErrorLogger(nn)
        # ...and residual errmeasure
        global errmeasure=ResidualErrmeasure(nep)
        linsolver = create_linsolver(linsolvercreator,nep,λ0)

        # init eigvec
        x0=randn(n^2);
        x0=normalize!(lin_solve(linsolver,x0,tol=tol))
        x0=normalize!(lin_solve(linsolver,x0,tol=tol))
        x0=x0/sqrt(x0'*nepv.B*x0);

        global num_smw_linsolves=0
        global num_GH_evals=0

        # solve NEP
        tic = time_ns()
        λ1,v = try
            augnewton(Float64,nep,λ=λ0,v=x0, logger=logger, maxit=15, tol=tol, armijo_factor=0.5, linsolvercreator=linsolvercreator, errmeasure=errmeasure);
        catch noconv
            noconv.λ, noconv.v;
        end
        toc = time_ns()
        time = Float64(toc-tic)*1e-9

        # log timings and convhist
        push!(timings, time)
        nep_err_hist1 = logger.nep_errs[logger.nep_errs .> -Inf]
        nepv_err_hist1 = compute_nepv_err_hist(nepv, logger.vv_approx, logger.ll_approx) 
        push!(nep_conv_hist, nep_err_hist1...)
        push!(nepv_conv_hist, nepv_err_hist1...)

        # normalize result
        v1=v/sqrt(v'*nepv.B*v);

        @printf "\nNumber of SMW linsolves λ1: %i \n" num_smw_linsolves 
        @printf "\nNumber of G,H evals λ1: %i \n" num_GH_evals
        @printf "\nλ1 NEPv relative residual: %.15e \n\n" norm(nepv(v1,v1) - λ1*nepv.E*v1)/norm(v1)

        # store sols
        eigvals[prob_no] = λ1
        eigvecs[:,prob_no] = v1

        # deflate NEP to compute next eig
        global dnep=deflate_eigpair(nep,λ1,v);

    # deflation for more eigvals
    else

        nn=size(dnep,1)
        
        # create deflated linsolver
        orglinsolver = create_linsolver(linsolvercreator,nep,λ0)
        linsolver = DeflatedNEPLinSolver(dnep, λ0, orglinsolver)
        dnep_linsolvercreator=DeflatedNEPLinSolverCreator(linsolvercreator)

        # same logger and errmeasure as before
        logger=DoubleErrorLogger(nn)
        errmeasure=ResidualErrmeasure(dnep)

        # init eigvec 
        x0=randn(nn);
        x0=normalize!(lin_solve(linsolver,x0,tol=tol))
        x0=normalize!(lin_solve(linsolver,x0,tol=tol))
        x0=x0/sqrt(x0'*nepv.B*x0)

        # reset counters
        global num_smw_linsolves = 0
        global num_GH_evals=0

        # solve deflated nep
        tic = time_ns()
        λ,v = try
            augnewton(Float64,dnep,λ=λ0,v=x0, logger=logger, maxit=15, tol=tol, armijo_factor=0.5, linsolvercreator=dnep_linsolvercreator,errmeasure=errmeasure);
        catch noconv
            noconv.λ, noconv.v;
        end
        toc = time_ns()
        time = Float64(toc-tic)*1e-9

        # log timings and convhist
        push!(timings, time)
        nep_err_hist2 = logger.nep_errs[logger.nep_errs .> -Inf]
        nepv_err_hist2 = compute_nepv_err_hist(nepv, logger.vv_approx, logger.ll_approx, dnep=dnep, deflated=true) 
        push!(nep_conv_hist, nep_err_hist2...)
        push!(nepv_conv_hist, nepv_err_hist2...)

        # deflate again for next eig
        global dnep=deflate_eigpair(dnep,λ,v);

        (λv,X)=get_deflated_eigpairs(dnep)

        J=argmin(abs.(λ .-λv))
        λk=λv[J]; v=X[:,J];
        vk=v/sqrt(v'*nepv.B*v);

        @printf "\nNumber of SMW linsolves λ%i: %i \n" prob_no num_smw_linsolves 
        @printf "\nNumber of G,H evals λ%i: %i \n" prob_no num_GH_evals
        @printf "\nλ%i NEPv relative residual: %.15e \n\n" prob_no norm(nepv(vk,vk) - λk*nepv.E*vk)/norm(vk)

        # store sols
        eigvals[prob_no] = λk
        eigvecs[:,prob_no] = vk

    end
end

# plot and save data to csv-file
num_exp_conv_hist = hcat(1:size(nep_conv_hist,1), nep_conv_hist, nepv_conv_hist)

p1 = plot(num_exp_conv_hist[:,1], num_exp_conv_hist[:,2:3], yscale=:log10, label=["NEP" "NEPv"])
ylabel!("Relative residual norm")
xlabel!("Total number fo iterations")

# save sonvergence history to file
writedlm("data/num_exp_new_conv_hist.csv", num_exp_conv_hist, ',')

# save timings to file
writedlm("data/timings_new.csv", timings, ',')

# pad modes with zeros
padded_vecs = zeros((n+2)^2,size(λ_init_vec,1))
for k in 1:size(λ_init_vec,1)
    padded_vecs[:,k] = pad_matrix(reshape(eigvecs[:,k], n, n))[:]
end

# basic plots of modes
cent = mapreduce(x-> reshape(x, 1,2), vcat, center_vec)
modep = [(contour([-1;xvec;1],[-1;xvec;1],reshape(padded_vecs[:,k],n+2,n+2), xlims=(-1,1), ylims=(-1,1), aspect_ratio=:equal, colorbar=:none); scatter!(cent[:,1], cent[:,2], markercolor=:black, label=:none, title="λ$(k)")) for k in 1:size(λ_init_vec,1)]
p2 = plot(modep...)

# pgfplots need one point per row
xvp = [-1; xvec; 1]
coords = [repeat(xvp, outer=n+2) repeat(xvp, inner=n+2)]
mode_data = hcat(coords, padded_vecs)

# save eigenmode data to csv file
writedlm("data/num_exp_new_mode.csv", mode_data, ',')

# save eigenvalue data to csv file
writedlm("data/num_exp_new_eigs.csv", eigvals, ',')

@show p1
@show p2

#=
# visualize the curves
samples = 90:0.037:138
data=sample_curves(nepv, samples);

data = mapreduce(x-> reshape(x, 1,6), vcat, data)

# since we dont do a selection, data contains many "almost equal" elements (rounding errors), remove these
data = unique(round.(tmp, digits=10), dims=1)

# this assumes single-valued data
p3 = plot(data[:,1], data[:,2:end], markerstrokewidth=0, markersize=1, label=["μ1" "μ2" "μ3" "μ4" "μ5"])
@show p3

# save curve data to csv file
writedlm("data/num_exp_new_curves.csv", data, ',')
=#
