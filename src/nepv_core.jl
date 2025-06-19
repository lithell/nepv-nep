using LinearAlgebra
using Polynomials

# Fundamental NEPv object
struct NEPv

    A0 # system matrix
    Av # matrix of vectors defining nonlinear terms
    B # normalization matrix
    E # RHS-matrix
    n::Int # size

end

# Evaluate A(v) (not very fast, forms lr terms explicitly)
function (nepv::NEPv)(v)
    num_nonlins = size(nepv.Av, 2)
    lr_sum = zeros(eltype(nepv.A0),size(nepv.A0))
    for k in 1:num_nonlins
        lr_sum += (nepv.Av[:,k]'*v)^2 * nepv.Av[:,k]*nepv.Av[:,k]'
    end
    eval = nepv.A0 + lr_sum
    return eval
end

# Evaluate A(v)*w
function (nepv::NEPv)(v,w)
    num_nonlins = size(nepv.Av, 2)
    lr_sum = zeros(eltype(nepv.A0),size(nepv.A0,1))
    for k in 1:num_nonlins
        lr_sum += (nepv.Av[:,k]'*v)^2 * nepv.Av[:,k]*(nepv.Av[:,k]'*w);
    end
    eval = nepv.A0*w + lr_sum
    return eval
end

# Constructors
function NEPv(A0,Av,B,E)
    return NEPv(A0,Av,B,E,size(A0,1))
end

function NEPv(A0,Av,B)
    return NEPv(A0,Av,B,I,size(A0,1))
end

function NEPv(A0,Av)
    return NEPv(A0,Av,I,I,size(A0,1))
end

struct HomogenizedNEPv
    nepv::NEPv
end

# For evaluating A(v/norm_B(v)) in the low-rank-NEPv
function (hnepv::HomogenizedNEPv)(v)

    org_nepv=hnepv.nepv;
    num_nonlins = size(org_nepv.Av, 2)

    lr_sum = zeros(size(org_nepv.A0))
    for k in 1:num_nonlins
        lr_sum += (org_nepv.Av[:,k]'*v)^2/(v'*org_nepv.B*v)*org_nepv.Av[:,k]*org_nepv.Av[:,k]'
    end

    eval = org_nepv.A0 + lr_sum

    E=hnepv.nepv.E;
    return E\eval
end

"""
    eval_lr_Jinv_action(nepv, v, σ, b)

Compute the action on 'b' of the inverse of the shifted Jacobian of 'nepv'.
I.e. compute (J(v) - σI)⁻¹b, by using the generalised SMW formula.
Here J(v) is the Jacobian of A(v)v of the low-rank NEPv defined by:

    A(v) = A + (a₁^Tv)^2⋅a₁a₁^T + ⋯ + (aₘ^Tv)^2⋅aₘaₘ^T.
"""
function eval_lr_Jinv_action(hnepv::HomogenizedNEPv, v, σ::Number, b; solve_method=:SMW)
# solve_method :full
# solve_method= :SMW
# solve_method= :SMW_mod

    org_nepv=hnepv.nepv;

    n = size(org_nepv.Av,1)
    m = size(org_nepv.Av,2)

    B = org_nepv.B;

    U = org_nepv.Av

    VT = similar(U');

    # Jacobian of (a_j'*v)^3/(v'*B*v) w.r.t v:
    Av = org_nepv.Av; # For convenience
    A0 = org_nepv.A0;

    if (solve_method == :SMW_mod)

        vTBv=v'*B*v;
        vTB=v'*B
        z=zeros(eltype(VT),size(VT,1));
        y=zeros(eltype(VT),size(VT,1));
        for j=1:size(VT,1)
            a=((Av[:,j]'*v)/(vTBv))^2 * 3 *vTBv;
            y[j]=a;
            z[j]=-((Av[:,j]'*v)/(vTBv))^2 * 2*(Av[:,j]'*v);
        end

        UVT1=U*Diagonal(y)*U'
        ueff=U*z;

        Aeff=A0-σ*I + UVT1;

        # solve with SMW: act = (Aeff +  ueff*vTB)\b
        F=factorize(Aeff);
        x1=F\b;
        x2=F\ueff;

        act = x1 - (1+vTB*x2)\(vTB*x1)*  x2

        return act
    end


    vTBv=v'*B*v;
    for j=1:size(VT,1)
        VT[j,:]=(((Av[:,j]'*v)/(vTBv))^2)*(3*vTBv*Av[:,j]'-2*(Av[:,j]'*v)*(v'*B));
    end

    if solve_method==:SMW

        AA = (A0-σ*I)\b
        BB = (A0-σ*I)\U

        act = AA-BB*((I+VT*BB)\(VT*AA))

    elseif (solve_method == :full)

        act = (A0-σ*I + U*VT)\b
    else
        error("Incorrect solver $solve_method");
    end

    return act

end

