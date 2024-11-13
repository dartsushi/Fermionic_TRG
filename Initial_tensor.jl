using TensorKit
using LinearAlgebra
using Base.Iterators

V = Vect[FermionParity](0 => 1, 1 => 1)

T = zeros(ComplexF64, 2,2,2,2,2,2,2,2)

t = 1
m = 0
function delta_func(i, j)
    return ComplexF64(i == j)
end

for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
    global T[pi1+1, pj1+1, pi2+1, pj2+1, i1+1, j1+1, i2+1, j2+1] = 
        (delta_func(i1 + i2 + pj1 + pj2, 1)*delta_func(pi1 + pi2 + j1 + j2, 1) - m*delta_func(i1 + i2 + pj1 + pj2, 0)*delta_func(pi1 + pi2 + j1+ j2, 0))*
        (sqrt(t)^(i1 +i2 +j1 +j2 +pi1 + pi2+ pj1 +pj2))*((-1)^(j1*(i2 + pj1 + pj2) + j2*(pj1 + pj2) + pi1*pj2))
end

T_initial = TensorMap(T, V⊗V⊗V⊗V←V⊗V⊗V⊗V)
