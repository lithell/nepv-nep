include("mep_solve.jl")
include("companion_linearize.jl")

# solve polysys for each lambda
function solve_polysys_mep(G, H)

    # compute companion matrices
    W_full = compute_mep_matrices(G,H)

    # the eigs represent x=mu.^3
    eigs, vectors = mep_solve(W_full; select_real_only=true)

    # this is wrt ''transformed'' system
    resfunc(mu) = norm([(mu'*G*mu-1),  ((H*mu).^3-mu)[1:end-1,:]])

    # filter out values that are not good sols to polysys
    # this is just for discarding very bad sols
    inds = findall(x -> resfunc([x...])<1e-1, eigs)

    if isempty(inds)
        error("Could not find any good sols to polysys")
    end

    eigs = eigs[inds]
    vectors = vectors[inds]

    # this is not needed, we still make final selection later
    # eigs contains duplicates up to sign, remove for now, considered in polishing
    # im not so sure about this, should probably do this more thoughtfully
    #eigs = eigs[1:Int(end/2)] # is this true in general? Think so...

    # compute mu^2 
    eigs = map(x -> x.^2, eigs)
    eigs = map(x -> x.^(1/3), eigs)

    return eigs

end

# refines the solutions for one given lambda (wrt the original polysys)
function newtonrefine_mep_sols!(G,H,all_mu,num_iters)

    m = size(G,1)

    # polysys
    F(s) = reshape([
        ((s.^3)'*G*(s.^3) - 1);
        (H*s.^3 - s)[1:end-1, :]
        ],m)

    # Jacobian (from manuscript)
    J(s) = [
        6*(s.^3)'*G*diagm(s.^2);
        (3*H*diagm(s.^2) - I)[1:end-1,:]
        ]

    # refine each solution
    for (idx, mu) in enumerate(all_mu)

        # splat to a vector
        mu_tmp = sqrt.([mu...])

        # we need to consider all sign combinations
        mu_tmp3=copy(mu_tmp);

        # iterate over bitstring representing all combinations of signs
        for jj=0:(2^m-1)

            mu_tmp2=copy(mu_tmp)
            bs = bitstring(jj)
            for k=1:m
                # check what values should have negative sign
                s=bs[end-k+1]=='1' ? -1 : 1
                mu_tmp2[k]=mu_tmp2[k]*s;
            end

            # choose the solution with smallest residual
            if (norm(F(mu_tmp2))<norm(F(mu_tmp3)))
                mu_tmp3=copy(mu_tmp2);
            end
        end
        
        # chosen solution
        mu_tmp=mu_tmp3;
        
        # newton polishing
        for k in 1:num_iters
            mu_tmp = mu_tmp - J(mu_tmp)\F(mu_tmp)
        end

        # square for final result
        mu_tmp = mu_tmp.^2
        
        # save refined
        all_mu[idx] = Tuple(mu_tmp)

    end

end
