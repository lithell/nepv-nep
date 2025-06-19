using NonlinearEigenproblems
using Plots
using DelimitedFiles
using HomotopyContinuation
using Random
using LowRankApprox

# Workaround for https://github.com/nep-pack/NonlinearEigenproblems.jl/issues/269
import NonlinearEigenproblems.LowRankFactorizedNEP
function NonlinearEigenproblems.LowRankFactorizedNEP(L::AbstractVector{S}, U::AbstractVector{S},
                              f) where {S<:AbstractMatrix}
    A=map(i-> LowRankMatrix(L[i],U[i]),1:size(L,1))
    rnk = mapreduce(u -> size(u, 2), +, U)
    spmf=SPMF_NEP(A, f, align_sparsity_patterns=false, check_consistency=false);
    return LowRankFactorizedNEP(spmf,rnk,L,U)
end

Random.seed!(0)

include("nepv_core.jl");
include("connect_curves.jl")

function polynep(nepv)

    a1=nepv.Av[:,1];
    a2=nepv.Av[:,2];
    A0=nepv.A0;
    n=size(A0,1);
    B=nepv.B;

    # poly approx
    λ_s=4.1949
    p = λ -> ((λ-λ_s)^2*(-260.9402 + 131.8256*λ - 16.1873*λ^2))^(1/3)
    #r = λ -> 420.5445 - 302.0953*λ + 72.1717*λ^2 - 5.7331*λ^3
    #r = λ -> 420.5445087380223 - 302.0952860720034*λ + 72.17170519085354*λ^2 - 5.733052772569442*λ^3
    #r = λ -> (λ-4.026628162196927)*(18.217320559692872 - 8.562075980937557*λ + 1.0*λ^2)
    r = λ -> -5.7331*(λ - 4.0266)*(18.2173 - 8.5621*λ + 1.0*λ^2)

    M = s-> begin
        return A0-s*I+p(s)*a1*a1'+r(s)*a2*a2';
    end

    nep_fun=λ -> Float64.(M(λ))
    #nep_fun=λ -> (M(λ))

    # Compute the derivative using complex step approximation
    ee=sqrt(eps());
    nep_dfun=λ -> (M(λ+ee)-M(λ-ee))/(2ee);

    # Create the nep
    polynep=Mder_NEP(size(A0,1), (s,der)->  der==0 ? nep_fun(s) : nep_dfun(s));

end


function polysys_solver_hc(G,H,guess) # HomotopyContinuation
    
    if size(G,1)==2
        @var mu1,mu2
        μ=[mu1;mu2];
    elseif size(G,1)==3
        @var mu1,mu2,mu3
        μ=[mu1;mu2;mu3];
    else 
        error("not implemented")
    end
    f1=(μ.^3)'*(G*(μ.^3)).-1
    f2=(H*(μ.^3) .- μ)[1:end-1,:]
    F=System(vec(vcat(f1,f2)));
    # Obtain all solutions to the system
    result = solve(F)
    # Fetch all real-valued solutions
    all_mu=real_solutions(result);
    # Pick the solution closest to the guess
    distvec=map(mu-> norm(mu.-guess), all_mu)
    mu = all_mu[argmin(distvec)];
    return mu;
end


# compute mu_sqr numerically, given λ and target 

# global variable for logging number of evals of g,h
num_gh_evals = 0
function musqr_num(nepv, λ, guess)

    A0=nepv.A0;
    Av=nepv.Av;
    n=size(A0,1);
    B=nepv.B;
    E=nepv.E

    @show λ
    # precompute transfer functions
    interm = (λ*E-A0)\Av # vilhelm was here
    G = interm'*(B*interm)
    H = (Av'*interm)

    # log 
    global num_GH_evals += 1

    # solve
    mu = polysys_solver_hc(G,H,guess) 

    musqr = (mu).^2

end


function musqr(nepv,λ,target)

    a1=nepv.Av[:,1];
    a2=nepv.Av[:,2];
    A0=nepv.A0;
    n=size(A0,1);
    B=nepv.B;
    E=nepv.E;

    # Create h and g functions
    # vilhelm was here, fixed to comply with E
    h11 = λ->a1'*((λ*E-A0)\a1);
    h12 = λ->a1'*((λ*E-A0)\a2);
    g11 = λ->a1'*((λ*E-A0)\(B*((λ*E-A0)\a1)));
    g12 = λ->a1'*((λ*E-A0)\(B*((λ*E-A0)\a2)));
    g22 = λ->a2'*((λ*E-A0)\(B*((λ*E-A0)\a2)));

    # Compute coeffs in third order poly
    pcoeff0= -h12(λ)^2;

    pcoeff1=g22(λ)

    pcoeff2=2*h12(λ)*g12(λ) - 2*g22(λ)*h11(λ);

    pcoeff3=h12(λ)^2*g11(λ) - 2*h12(λ)*g12(λ)*h11(λ) +g22(λ)*h11(λ)^2;


    # solve for gamma1
    p=Polynomial([pcoeff0;pcoeff1;pcoeff2;pcoeff3]);
    mu1sqr=roots(p) # gamma1 = mu1^2


    # Compute mu1 in order to compute mu2sqr
    mu1=sqrt.(mu1sqr .+0im);
    mu1=[mu1;-mu1]; # Include also negative solutions


    # Compute mu2
    mu2=((mu1.-h11(λ)*mu1.^3)/h12(λ)  .+ 0im).^(1/3)
    root_of_unity=exp(1im*2*pi/3);
    # Include all non-principal cube roots
    mu2=[mu2;mu2*root_of_unity;mu2*(root_of_unity^2)];
    mu1=[mu1;mu1;mu1]; # Corresponding to the same mu1

    # Recompute mu1sqr and mu2sqr
    mu2sqr=mu2.^2;
    mu1sqr=mu1.^2;

    # double check:
    res1=g11(λ)*mu1.^6+2*g12(λ)*mu1.^3 .* mu2.^3 +g22(λ)*mu2.^6 .- 1
    #res2=mu1 -h11(λ)*mu1.^3-h12(λ)*mu2sqr.^(3/2);
    res2=mu1 -h11(λ)*mu1.^3-h12(λ)*mu2.^3; # vilhelm was here

    # Purge all non-relevant solutions
    J=isreal.(mu1sqr) .& isreal.(mu2sqr);
    TOL=1e-8;
    J=(abs.(imag.(mu1sqr)).<TOL) .& (abs.(imag.(mu2sqr)).<TOL)
    J=J .& (real.(mu1sqr) .> TOL) .& (real.(mu2sqr) .> TOL)
    J=J .& (abs.(res1) .< TOL) .& (abs.(res2) .< TOL)

    musqr_mat=[mu1sqr mu2sqr]
    musqr_mat=musqr_mat[findall(J),:]

    if (size(musqr_mat,1)<1)

        @show λ
        #@show mu1sqr
        #@show mu2sqr
        #@show abs.(res1)
        #@show abs.(res2)
        @show pcoeff0, pcoeff1, pcoeff2, pcoeff3
        error("No musqr found");
    end


    # Prioritize target
    J=sortperm(abs.(musqr_mat[:,1] .- target))


    return musqr_mat[J,:]

end

# add functionality for pushing to musqr dictionary
function mu_j_sqr(j,nepv_approx,s,target)

    # musqr_dict is global 
    if !haskey(musqr_dict, s)
        musqr_dict[s] = musqr_num(nepv_approx,s,target)
    end

    return musqr_dict[s][j]

end

"""
    two_terms(nepv, target; nepv_approx=nepv, spmf=false)

Set-up the NEP `nep` corresponding to `nepv`. `target` determines how to choose among NEPs in the cases where the functions become multi-valued. Selects function-values closest to `target`.  Use `nepv_approx` if mu1, mu2 should be computed on a lower resolution grid. If `spmf` is set to `true`, the NEP returned is of SPMF-type, and low-rank factorized.
"""
function two_terms(nepv::NEPv,target;nepv_approx=nepv,spmf=false)

    a1=nepv.Av[:,1];
    a2=nepv.Av[:,2];
    A0=nepv.A0;
    n=size(A0,1);
    B=nepv.B;
    E=nepv.E;

    if spmf

        # full-rank part
        nep1=SPMF_NEP([A0,sparse(E,n,n)], [s->one(s) , s-> -s])

        mu1sqr = s -> mu_j_sqr(1,nepv_approx,s,target)
        mu2sqr = s -> mu_j_sqr(2,nepv_approx,s,target)
        fv = [mu1sqr; mu2sqr]
        a1=reshape(a1,n,1); # nep-pack expects matrices
        a2=reshape(a2,n,1);

        L = [a1, a2]
        U = deepcopy(L)

        # low-rank part
        nep2= LowRankFactorizedNEP(L,U,fv)

        # full nep
        nep=SumNEP(nep1,nep2)

        return nep

    else
        # First create the NEP (for scalar input)
        M = s-> begin
            
            musqr_temp=musqr_num(nepv_approx,s,target); # this seems to work better 

            return A0-s*E+musqr_temp[1]*a1*a1'+musqr_temp[2]*a2*a2'; # vilhelm was here
        end

        nep_fun=λ -> Float64.(M(λ))

        # Compute the derivative using FDs
        ee=sqrt(eps());
        nep_dfun=λ -> (M(λ+ee)-M(λ-ee))/(2ee);

        # Create the nep
        nep=Mder_NEP(size(A0,1), (s,der)->  der==0 ? nep_fun(s) : nep_dfun(s));

        return nep
    end

end


## Example use:
## Setup nepv-problem
#A0=[3.0 1 0 ;1 2 0.5; 0 0.5 4]
#A0 = sparse(A0)
#a1=[1;0.0;0.0];
#a2=[0;1.0;0.0];
#Av=[a1 a2];
#nepv=NEPv(A0,Av,I,I,size(A0,1))
#
#
## Compute the NEP
#nep=two_terms(nepv,0)
#
## To find different eigvals
## use different starting values
#λ0=4.3
##λ0=1
#
#M0=compute_Mder(nep,λ0);
#
#v0=(M0\(M0\ones(3))); normalize!(v0)
#
#λ,v = try
#    augnewton(Float64,nep,λ=λ0,logger=1,
#                maxit=50,v=v0,tol=1e-14,armijo_factor=0.5)
#catch noconv
#    noconv.λ, noconv.v
#end
#
#normalize!(v)
#
#@show (A0-λ*I+(v'*a1)^2*a1*a1'+(v'*a2)^2*a2*a2')*v
#@show norm((A0-λ*I+(v'*a1)^2*a1*a1'+(v'*a2)^2*a2*a2')*v)
#
#
## visualize the functions with the new code
#s1 = 0:0.011:3.5
#s2 = 3.501:0.0011:4.3
#s3 = 4.31:0.011:5
#samples = vcat(s1,s2,s3)
#
#samples = data=sample_curves(nepv, samples);
#splitted_data=split_data(data)
#
## Remove splitted intervals with only one data point
#deleteat!(splitted_data,findall(map(c->size(c,1),splitted_data) .== 1))
#
## Turn splitted data into curves
#all_curves=[];
#for thisdata in splitted_data
#    curves=connect_curves(thisdata);
#    for c in curves
#        push!(all_curves,c);
#        #@show size(all_curves)
#    end
#end
#
## Semi-manual fix, when we identify that
##fix_mixup!(all_curves,3.8,4.2) # not supported...
#
## Connect the curves further by gluing them together (in a greedy way)
#greedy_glue!(all_curves)
#
## Only plot one out of the two curves.
##p = visualize([all_curves[1]])
#p = visualize(all_curves)
#
## data matrix
#curve_data = mapreduce(permutedims, vcat, all_curves[1])
#curve_data[:,2:end] = curve_data[:,2:end].^2
#p2 = plot(curve_data[:,1], curve_data[:,2:end], xticks=0:5)
#
## save to csv
##writedlm("data/two_curves_data.csv", curve_data, ',')
















