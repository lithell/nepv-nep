include("nepv_core.jl")

"""
    gen_disc_mat(n; kwargs...)

Generate FD matrix 'A', vector of x-values 'xv', and disc. step 'h' for the 1D/2D GPE-type problem in the manuscript.
"""
function gen_disc_mat(n;potential=:square, dim=1)

    if dim==1
        xv = range(-1,1,length=n+2); xv=xv[2:end-1];

        h = xv[2]-xv[1];

        A = -spdiagm(-1 => ones(n-1), 1 => ones(n-1), 0 =>-2*ones(n));

        if potential==:square
            d = 10*(xv.^2)
        elseif potential==:quantum_well
            d= -(abs.(xv) .< 0.5)*100;
        end


        A = A./h^2 + spdiagm(d);

    elseif dim==2

        # same disc in x and y dir
        xv = range(-1,1,length=n+2); xv=xv[2:end-1];
        h = xv[2]-xv[1];

        d = zeros(n,n)

        # potential vector
        if potential==:quantum_well
            p1(x,y) = (abs(x)<0.5 || abs(y)<0.5) ? 10 : 0
            d = -p1.(xv,xv)*p1.(xv,xv)'
        elseif potential==:harmonic_lattice
            #p2(x,y) = 1/2*(x^2 + 4*y^2) + 8^2*(sin(6*pi*x)^2 + sin(4*pi*y)^2)
            p2(x,y) = 16*(x^2 + 4*y^2) + 64*(sin(4*pi*x)^2 + sin(4*pi*y)^2) # new values
            d = [p2(xx,yy) for xx in xv, yy in xv]' # notice trnsp!
        else
            @error "Not implemented"
        end

        d = reshape(d, n^2)

        d = spdiagm(d)

        # disc Laplacian
        D2 = 1/h^2*spdiagm(-1 => ones(n-1), 1 => ones(n-1), 0 =>-2*ones(n));
        spI = spdiagm(0 => ones(n))

        A = kron(D2,spI) + kron(spI, D2)

        # final A
        A = -A + d

    end

    return A, xv, h
end

"""
    discretize_NEPv(n; kwargs...)

Discretize the 1D, two term, GPE-type problem in the manuscript, returning a NEPv-object 'nepv'.
"""
function discretize_NEPv(n; c1=15, c2=17,
                         sigma1=30, sigma2=30,
                         x1=-0.1,
                         x2=0.2)

    A, xv, h = gen_disc_mat(n, potential=:quantum_well);

    psi1 =c1*exp.(-sigma1*(xv .- x1).^2);
    psi2 =c2*exp.(-sigma2*(xv .- x2).^2);
    a1 = h*psi1
    a2 = h*psi2

    Av = [a1 a2]

    B = h*I;

    E = h*I;

    nepv = NEPv(A*h, Av, B, E)

    return nepv
end

"""
    discretize_2D_NEPv(n; kwargs...)

Discretize the 2D, two term, GPE-type problem in the manuscript, returning a NEPv-object 'nepv'.
"""
function discretize_2D_NEPv(n; c1=15, c2=17,
                         sigma1=30, sigma2=30,
                         x1=-0.1,
                         y1=-0.1,
                         x2=0.2, 
                         y2=0.2, 
                         potential=:quantum_well
                         )

    A, xv, h = gen_disc_mat(n, potential=potential, dim=2);

    # same disc in x and y dir
    psi1(x,y) =c1*exp(-sigma1*((x - x1)^2 + (y - y1)^2));
    psi2(x,y) =c2*exp(-sigma2*((x - x2)^2 + (y - y2)^2));

    psi1v = [psi1(x,y) for x in xv, y in xv]' # notice trnsp!
    psi2v = [psi2(x,y) for x in xv, y in xv]'

    psi1v = reshape(psi1v, n^2, 1) 
    psi2v = reshape(psi2v, n^2, 1) 

    a1 = h^2*psi1v
    a2 = h^2*psi2v

    # trucate and sparse format a1,a2
    a1 = (a1 .> 1e-16).*a1; 
    a2 = (a2 .> 1e-16).*a2;

    Av = [a1 a2]

    B = h^2*I;

    E = h^2*I;

    nepv = NEPv(A*h^2, Av, B, E)

    return nepv
end

function discretize_mterm_nepv(n, c_vec, sigma_vec, center_vec, potential=:quantum_well)

    # num nonlin terms
    m = size(c_vec, 1)

    # FD disc mat
    A0, xv, h = gen_disc_mat(n, potential=potential, dim=2);

    # k:th functional
    psik(x,y,k) = c_vec[k]*exp(-sigma_vec[k]*((x - center_vec[k][1])^2 + (y - center_vec[k][2])^2));

    # compute the Av matrix 
    Av = mapreduce(k -> reshape([psik(x,y,k) for x in xv, y in xv]', n^2, 1), hcat, 1:m)

    # set small elems to zero
    Av = (Av .> 1e-16).*Av; 

    # normalization 
    B = h^2*I
    E = h^2*I

    nepv = NEPv(A0*h^2, Av, B, E)

    return nepv
end
