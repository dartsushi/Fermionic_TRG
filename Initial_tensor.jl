using TensorKit
using LinearAlgebra
using Base.Iterators
include("impurity_pCTMRG.jl")
V = Vect[FermionParity](0 => 1, 1 => 1)



function delta_func(i, j)
    return Int(i == j)
end

function Levin_TRG_1step(T::AbstractTensorMap,χ::Int64)
    S1,S3 = decomposeT(T,χ,(1,2),(3,4))
    S2,S4 = decomposeT(T,χ,(2,3),(4,1))
    Tnew = ncon([S1,S2,S3,S4],[[1,4,-2],[3,1,-3],[-4,2,3],[-1,4,2]])
    #Tnew = permute(Tnew,(3,4),(1,2))
    #trace_norm = norm(@plansor Tnew[1 2; 1 3] *edge_parity_two[3; 2])
    trace_norm = norm(@tensor Tnew[1 2; 1 2])
    return Tnew/trace_norm, trace_norm
end

function Levin_TRG(A_initial::AbstractTensorMap,χ::Int64,step::Int64;)
    initial_norm = norm(@tensor A_initial[1 2; 1 2])
    A = A_initial/initial_norm
    time_list = []
    norms = [initial_norm]
    for i=1:step
        @info i
        start = time()
        A, norm = Levin_TRG_1step(A,χ)
        push!(time_list,time()-start)
        push!(norms, norm)
        #cft_xn(A)
    end
    return A,time_list, norms
end

function P_tensor()
    P_tens = zeros(ComplexF64, 2,2,2,2,2,2,2,2)
    for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
        P_tens[pi1+1, pj1+1, pi2+1, pj2+1, i1+1, j1+1, i2+1, j2+1] = 
            i1*(j1+j2+pi1+pi2) + i2*(j2+ pi1 + pi2) + pj1*(pi1+pi2) + pj2*pi2 + pi1+pi2
    end
    return P_tens
end
p_tensor = P_tensor()

A_tensor = zeros(ComplexF64,2,2,2,2)
A_tensor[2,2,1,1] = 1 + 1im
A_tensor[1,1,2,2] = -1 - 1im
A_tensor[2,1,1,2] = 1 - 1im
A_tensor[2,1,2,1] = 2
A_tensor[1,2,1,2] = -2im
A_tensor[1,2,2,1] = 1 - 1im

A_bar_tensor = zeros(ComplexF64, 2,2,2,2)
A_bar_tensor[2,2,1,1] = -1 + 1im
A_bar_tensor[1,1,2,2] = 1 - 1im
A_bar_tensor[2,1,1,2] = -1 - 1im
A_bar_tensor[2,1,2,1] = -2
A_bar_tensor[1,2,1,2] = -2im
A_bar_tensor[1,2,2,1] = -1 - 1im

function get_init_Tensor(mu, m, g)
    
    T = zeros(ComplexF64, 2,2,2,2,2,2,2,2)
    
    V = Vect[FermionParity](0 => 1, 1 => 1)
    for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
        p = p_tensor[pi1+1, pj1+1, pi2+1, pj2+1, i1+1, j1+1, i2+1, j2+1]
        T[pi1+1, pj1+1, pi2+1, pj2+1, i1+1, j1+1, i2+1, j2+1] = 
            ((-1)^p)*exp(0.5*mu*(i2 - j2 + pi2 - pj2))*((1/sqrt(2))^(i1 + i2 + j1 + j2 + pi1 + pi2 + pj1 + pj2))*
            (((m + 2)^2 + 2*g^2)*delta_func(i1 + i2 + pj1 + pj2, 0)*delta_func(j1 + j2 + pi1 + pi2, 0) -
            (m + 2)*delta_func(i1 + i2 + pj1 + pj2, 1)*delta_func(j1 + j2 + pi1 + pi2, 1) - 
            ((-1)^(i1 + i2 + j2 + pi1))*(1im^(i2 + j2 + pi2 + pj2))*(m + 2)*delta_func(i1 + i2 + pj1 + pj2, 1)*delta_func(j1 + j2 + pi1 + pi2, 1) - 
            A_bar_tensor[i1+1,i2+1,pj1+1,pj2+1]*A_tensor[j1+1,j2+1,pi1+1,pi2+1])
            #(delta_func(i1 + i2 + pj1 + pj2, 1)*delta_func(pi1 + pi2 + j1 + j2, 1) - m*delta_func(i1 + i2 + pj1 + pj2, 0)*delta_func(pi1 + pi2 + j1+ j2, 0))*
            #(sqrt(t)^(i1 +i2 +j1 +j2 +pi1 + pi2+ pj1 +pj2))*((-1)^(j1*(i2 + pj1 + pj2) + j2*(pj1 + pj2) + pi1*pj2))
    end
    return TensorMap(T, V⊗V⊗V⊗V←V⊗V⊗V⊗V)
end
T_initial = get_init_Tensor(0, 0, 0)
χ = 24
steps = 20
V_fuse = fuse(V, V)
U = isometry(V_fuse,V⊗V)
Udg = adjoint(U)

@tensor T_fused[-1 -2; -3 -4] := T_initial[1 2 3 4;5 6 7 8] *U[-1;1 2]*U[-2;3 4]*Udg[5 6;-3]*Udg[7 8;-4]
T_final, time_list, norms = Levin_TRG(T_fused, χ,steps)
sum = 0
for (index, value) in enumerate(norms)
    global sum+= log(value) * 2^float(1-index)
end
println(sum)

println(norms)