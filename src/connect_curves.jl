include("nepv_core.jl")

# Return one vector of points. Every points (lambda,mu1,...mup)
function sample_curves(nepv,λ_range)
    a1=nepv.Av[:,1];
    a2=nepv.Av[:,2];
    A0=nepv.A0;
    Av=nepv.Av;
    E=nepv.E;
    n=size(A0,1);
    B=nepv.B;
    n = size(A0,1)
    m = size(Av,2)

    # init plot
    #p = scatter()

    # polysys vars
    #@var mu1,mu2,mu3
    #μ=[mu1;mu2;mu3]; # this is harcoded for 3 terms

    @show m
    #@var μ[1:m]

    data=Vector();
    # solve for each λ in range
    for λ in λ_range

        @show λ

        # precomp
        interm = (λ*E-A0)\Av
        G = interm'*(B*interm)
        H = (Av'*interm)

        all_mu_sqr = try
            solve_polysys_mep(G,H)
        catch
            continue
        end

        num_iters = 3
        newtonrefine_mep_sols!(G,H,all_mu_sqr,num_iters)

        for i in 1:size(all_mu_sqr,1)
           push!(data,vcat(λ,[all_mu_sqr[i]...]));
        end

    end

    #for λ in λ_range

    #    @show λ

    #    # precomp
    #    interm = (λ*E-A0)\Av
    #    G = interm'*(B*interm)
    #    H = (Av'*interm)

    #    # sys
    #    f1=(μ.^3)'*(G*(μ.^3)).-1
    #    f2=(H*(μ.^3) .- μ)[1:end-1,:]
    #    F=System(vec(vcat(f1,f2)));

    #    # Obtain all solutions to the system
    #    result = solve(F)

    #    all_mu=real_solutions(result);

    #    for (i,muv) in enumerate(all_mu)
    #       push!(data,vcat(λ,muv));

    #    end

    #end
    return data;
end

# Splits the data (vector of points) into a vector of vector of points
# where every split has the same number of value per lambda values.
function split_data(data)
    full_λv=map(v->v[1], data);
    count_vec=map(λ-> count(full_λv .== λ), full_λv);

    intervals=[];

    c=count_vec[1];
    current_interval=[];
    for (i,λ) = enumerate(full_λv)
        if (count_vec[i] != c)
            push!(intervals,current_interval);
            current_interval=[];
            c=count_vec[i];
        end
        push!(current_interval,data[i]);
    end
    push!(intervals,current_interval);

end

function glue_curve!(curves; cut_off=0.1, λ_extrapolation=true)
    if (! all(map(c->size(c,1),all_curves) .> 1)) # typo
    #if (! all(map(c->size(c,1),curves) .> 1))
        λ_extrapolation = false; # Single entry curves
    end


    if (λ_extrapolation)
        # correct lambda by extrapolation
        #start_points=map(c->vcat(2*c[1][1]-c[2][1], c[1][2:end]),curves)
        start_points=map(c->2*c[1]-c[2],curves)

        # correct lambda by extrapolation
        #end_points=map(c->vcat(2*c[end][1]-c[end-1][1], c[end][2:end]),curves)
        end_points=map(c->2*c[end]-c[end-1],curves)

    else
        start_points=map(c->c[1], curves)
        end_points=map(c->c[end], curves)
    end

    # Compute distances between end-points and start points
    dist_mat=zeros(size(start_points,1),size(end_points,1));

    for (i,pi) = enumerate(start_points)
        for (j,pj) = enumerate(end_points)
            dist_mat[i,j]=norm(pi-pj);
        end
    end

    # Don't connect with itself
    dist_mat[diagind(dist_mat)] .= Inf;

    # Find smallest distance
    z=argmin(dist_mat); i=z[1]; j=z[2];

    if (dist_mat[i,j] > cut_off)
        return curves;
    end

    # start_point in curve i matches
    # end_point in curve j

    # remove the two curves and add the glued curve

    curvei=curves[i];
    curvej=curves[j];
    new_curve=vcat(curvej,curvei);

    deleteat!(curves,    sort([i;j]))
    push!(curves,new_curve);

    return curves;
end

function connect_curves(data)

    full_λv=map(v->v[1], data);
    λv=unique(sort(full_λv));


    startdata=data[findall(full_λv .== λv[1])];
    curves=Vector();

    for (j,d) in enumerate(startdata)
        push!(curves,[d]);
    end

    prev_points=startdata;
    for (i,λ) in enumerate(λv)
        if (i==1)
            continue;
        end

        thisdatav=data[findall(full_λv .== λ)];

#        @show size(curves)
#        @show curves
#        @show i
        if (i>2)
            curve_endpoints=deepcopy(map(c-> (2*c[end]-c[end-1]),curves));
        else
            curve_endpoints=deepcopy(map(c-> c[end],curves));
        end

        for (j,thisdata) in enumerate(thisdatav)
            distvec=map(c -> norm(c[2:end] - thisdata[2:end]) ,curve_endpoints)
            t=argmin(distvec);
            # Mark as taken
            curve_endpoints[t]=Inf*ones(size(curve_endpoints[t]))
            push!(curves[t],thisdata);
        end


        prev_points=thisdatav;
    end
    return curves

end

# reverse_ind can be an Inf, :random, or :none
function greedy_glue!(curves, reverse_ind=:random)

    # First try to glue with the given order until no further improvements
    s=size(curves,1);
    while true
        glue_curve!(curves);
        if (size(curves,1) == s)
            break
        end
        s=size(curves,1)
    end

    # Then try to reverse some set of points "randomly" and connect again
    if (size(curves,1) > 2  && reverse_ind != :none)
        s=size(curves,1);
        c=0;
        while true
            i=reverse_ind
            if (i==:random)
                i=mod(rand(Int),size(curves,1))+1;
            end
            curves[i] = reverse(curves[i])
            glue_curve!(curves);
            if (size(curves,1) == s  && (c>10))
                break
            end
            c=c+1;
            s=size(curves,1)
        end

    end


end

# Fixes a mixup between curves in the λ interval [λ1,λ2]
# based on considering the second derivative
function fix_mixup!(curves,λ1,λ2)

    cv=zeros(size(curves,1))
    for (i,curve) in enumerate(curves)
        λv=map(s->s[1],curve)
        cv[i]=count((λv .> λ1) .&& (λv .< λ2));
    end
    curve_index=findall(cv .> 0);

    if (size(curve_index,1)==0)
        return curves;
    end

    if (size(curve_index,1)>2)
        error("Not supported");
    end
    i=curve_index[1]; j=curve_index[2];
    curve1=curves[i];
    curve2=curves[j];
    # Largest derivative


    λv1=map(s->s[1],curve1)
    λv2=map(s->s[1],curve2)
    # Second derivative (or zero outside of the domain of interest)
    second_der=diff(diff(map(s->s[2],curve1))) .*
        ((λv1[1:end-2] .> λ1) .&& (λv1[1:end-2] .< λ2))


    # Index of largest second derivative
    t=argmax(abs.(second_der))+1;

    # Switch them
    curves[i]=vcat(curve1[1:t],curve2[(t+1):end])
    curves[j]=vcat(curve2[1:t],curve1[(t+1):end])

    return curves;
end


# Plots the set of curves in curves.
# Curves is a Vector where every element is Vector of points (lambda,mu1,..mum)
function visualize(curves; sq=false)

    p=plot(;legend=false);

    for (i,curve) in enumerate(curves);
        λv=map(c->c[1], curve);

        for j=2:size(curve[1],1)
            muv=map(c->c[j], curve)
            if sq
                p = plot!(λv,muv.^2);
            else
                p = plot!(λv,muv);
            end
        end
    end


    return p
end

# quick helper function for creating sample vectors
function refine_samples(lb,ub,interest_ps,init_res,fine_res)

    init_vec = Vector(lb:init_res:ub)

    num_extra_points = floor(init_res/fine_res)

    for point in interest_ps

        extra_points = range(point-num_extra_points*fine_res, point+num_extra_points*fine_res, Int(2*num_extra_points))

        push!(init_vec, Vector(extra_points)...)
        sort!(init_vec)

    end

    refined = init_vec

    return refined


end




# Example use
#data=sample_curves(nepv, 0:0.01:9);
# data=sample_curves(nepv,4:0.001:4.5)
# data=sample_curves(nepv,6:0.01:8)
#data=sample_curves(nepv,6.8:0.005:7.6)
#data=sample_curves(nepv,3:0.01:10)
#data=sample_curves(nepv,3.5:0.001:4.5)


#splitted_data=split_data(data)
#
## Remove splitted intervals with only one data point
#deleteat!(splitted_data,findall(map(c->size(c,1),splitted_data) .== 1))
#
#
## Turn splitted data into curves
#all_curves=[];
#for thisdata in splitted_data
#    curves=connect_curves(thisdata);
#    for c in curves
#        push!(all_curves,c);
#    end
#end
#
## Semi-manual fix, when we identify that
#fix_mixup!(all_curves,7,7.5)
#
## Connect the curves further by gluing them together (in a greedy way)
#greedy_glue!(all_curves)
#
#
## Only plot one out of the two curves.
#visualize([all_curves[2]])
