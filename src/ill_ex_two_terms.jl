include("two_terms.jl")
using Random

Random.seed!(205)

# Setup nepv-problem
A0=[
    6 5 4;
    5 16 23;
    4 23 20
    ]


alpha = 2
beta = 2
a1=alpha*[1;0;0]; 
a2=beta*[0;1;0];

Av=[a1 a2];
nepv=NEPv(A0,Av,I,I,size(A0,1))

# Compute the NEP
target = 0.5
nep=two_terms(nepv,target)

# To find different eigvals use different starting values
#λ0 = 0 #λ=-1.3447192878973713
#λ0 = 10 #λ=19.01651658506354
#λ0 = 30 #λ=46.433654584941976

init_ll_vec = [0, 10, 30]
eig_branch_data = zeros(3,3)

for (idx,λ0) in enumerate(init_ll_vec)

    @show idx

    global M0=compute_Mder(nep,λ0);
    global v0=(M0\(M0\ones(3))); normalize!(v0)
    global λ,v = try
        augnewton(Float64,nep,λ=λ0,logger=1,
                    maxit=20,v=v0,tol=1e-14,armijo_factor=0.5)
    catch noconv
        noconv.λ, noconv.v
    end

    normalize!(v)

    @show norm(nepv(v,v)-λ*v)
    @show v
    eig_branch_data[idx, :] = [λ; musqr_num(nepv, λ, target)]
end

#writedlm("data/eig_branch_data.csv", eig_branch_data , ',')

asd

# deflate NEP to compute next eig
#dnep=deflate_eigpair(nep,λ,v);
#
#nn=size(dnep,1)
#λ0=27
#M0=compute_Mder(dnep,λ0);
#v0=(M0\(M0\ones(nn))); normalize!(v0)
#
#λ,v = try
#    augnewton(Float64,dnep,λ=λ0,v=v0, logger=1, maxit=80, tol=1e-14, armijo_factor=0.5);
#catch noconv
#    noconv.λ, noconv.v;
#end
#
## deflate again for next eig
#dnep=deflate_eigpair(dnep,λ,v);
#
#(λv,X)=get_deflated_eigpairs(dnep)
#
#J=argmin(abs.(λ .-λv))
#λ=λv[J]; v=X[:,J];
#normalize!(v)
#
#@show norm(nepv(v,v)-λ*v)
#
#nn=size(dnep,1)
#
#λ0=-16.54
#M0=compute_Mder(dnep,λ0);
#v0=(M0\(M0\ones(nn))); normalize!(v0)
#
#λ,v = try
#    augnewton(Float64,dnep,λ=λ0,v=v0, logger=1, maxit=50, tol=1e-12, armijo_factor=0.1);
#catch noconv
#    noconv.λ, noconv.v;
#end
#normalize!(v)
#
#dnep=deflate_eigpair(dnep,λ,v);
#
#(λv,X)=get_deflated_eigpairs(dnep)
#
#J=argmin(abs.(λ .- λv))
#λ=λv[J]; v=X[:,J];
#normalize!(v)
#
#@show norm(nepv(v,v)-λ*v)

# visualize the functions 
# sample interesting regions more
#s1 = -20:0.031:-5.5
#s2 = -5.501:0.003:-4.5
#s3 = -4.55:0.031:18.5
#s4 = 18.501:0.003:19.5
#s5 = 19.55:0.031:41.5
#s6 = 41.501:0.003:42.5
#s7 = 42.5:0.031:60
#samples=vcat(s1,s2,s3,s4,s5,s6,s7)
samples = -20:0.053:60


data=sample_curves(nepv, samples);
splitted_data=split_data(data)

# Remove splitted intervals with only one data point
deleteat!(splitted_data,findall(map(c->size(c,1),splitted_data) .== 1))

# Turn splitted data into curves
all_curves=[];
for thisdata in splitted_data
    curves=connect_curves(thisdata);
    for c in curves
        push!(all_curves,c);
    end
end

# Semi-manual fix, when we identify that
#fix_mixup!(all_curves,3.8,4.2) # not supported...

# Connect the curves further by gluing them together (in a greedy way)
greedy_glue!(all_curves)

# Only plot one curves.
p = visualize(all_curves, sq=true)

asd

# data matrix
curve_data = mapreduce(permutedims, vcat, all_curves[3])
curve_data[:,2:end] = curve_data[:,2:end].^2
curve_data = reverse(curve_data, dims=1)
p2 = plot(curve_data[:,1], curve_data[:,2:end], xticks=-20:5:60)

# save to csv
writedlm("data/two_curves_data_new.csv", curve_data, ',')
