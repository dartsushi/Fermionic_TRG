using TensorOperations
using TensorKit


# some required operators for Quantum TFIM 
X = Tensor(ComplexF64[0 1; 1 0], ComplexSpace(2)' ⊗ ComplexSpace(2));
Y = Tensor(ComplexF64[0 -im; im 0], ComplexSpace(2)' ⊗ ComplexSpace(2));
Z = Tensor(ComplexF64[1 0; 0 -1], ComplexSpace(2)' ⊗ ComplexSpace(2));
Id = Tensor(ComplexF64[1 0; 0 1], ComplexSpace(2)' ⊗ ComplexSpace(2));
@tensor ZZ[-1 -2 -3 -4] := Z[-1 -2] * Z[-3 -4];

function load_tensor(λ::Float64, χ::Int64)
    
    filename = "peps/peps_ising_lambda_" * string(λ) * "_D_" * string(χ) * ".txt"
    raw_data = parse.(ComplexF64, readlines(filename))
    
    # convert list data into Tensor(left, up, right, down, physical) = (3,3,3,3,2)
    tensored_data = Tensor(raw_data, ComplexSpace(χ) ⊗ ComplexSpace(χ) ⊗ ComplexSpace(χ)' ⊗ ComplexSpace(χ)' ⊗ ComplexSpace(2))
    
    return tensored_data
end


function make_2d_tensor(A::TensorMap, insert_z = false)
    
    if insert_z
        # insert Z gate for magnetization impurity
        @tensor E[-1 -2 -3 -4 -5 -6 -7 -8] := A[-1 -3 -5 -7 1] * conj(A[-2 -4 -6 -8 2]) * Z[1 2]
    else
        @tensor E[-1 -2 -3 -4 -5 -6 -7 -8] := A[-1 -3 -5 -7 1] * conj(A[-2 -4 -6 -8 2]) * Id[1 2]
    end
    
    #combine two legs into one leg
    sep_space = space(A,1) ⊗ space(A,1)'

    comb_space = fuse(sep_space);
    F = isomorphism(comb_space ⊗ comb_space ⊗ comb_space' ⊗ comb_space', codomain(E));
    E_new = F * E;
    
    return permute(E_new, (1,2), (3,4))
end


function get_two_site_impurity(A::TensorMap, O::TensorMap, H::TensorMap)

    # prepare single site operator
    @tensor O_I[-1 -2 -3 -4] := O[-1 -2] * Id[-3 -4]
    @tensor I_O[-1 -2 -3 -4] := Id[-1 -2] * O[-3 -4]
    
    O_total = 0.5 * (O_I + I_O)
    H_total = O_total + H
    
    @tensor W_total[-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12] := A[-1 -5 -7 1 2] * A[-3 1 -9 -11 3] * conj(A[-2 -6 -8 4 5]) * conj(A[-4 4 -10 -12 6]) * H_total[2 5 3 6]
    
    sep_space = space(A,1) ⊗ space(A,1)';
    comb_space = fuse(sep_space);
    
    # place (1,2) legs in co-domain and fuse  
    W_total = permute(W_total, (1,2), (3,4,5,6,7,8,9,10,11,12));
    F_1 = isomorphism(comb_space, codomain(W_total));
    W_total = F_1 * W_total;

    # place (3,4) legs in co-domain and fuse  
    W_total = permute(W_total, (2,3), (4,5,6,7,8,9,10,11,1));
    F_1 = isomorphism(comb_space, codomain(W_total));
    W_total = F_1 * W_total;
    
    # place (5,6) legs in co-domain and fuse  
    W_total = permute(W_total, (2,3), (4,5,6,7,8,9,10,1));
    F_1 = isomorphism(comb_space, codomain(W_total));
    W_total = F_1 * W_total;
    
    # place (7,8) legs in co-domain and fuse  
    W_total = permute(W_total, (2,3), (4,5,6,7,8,9,1));
    F_1 = isomorphism(comb_space', codomain(W_total));
    W_total = F_1 * W_total;
    
    # place (9,10) legs in co-domain and fuse  
    W_total = permute(W_total, (2,3), (4,5,6,7,8,1));
    F_1 = isomorphism(comb_space', codomain(W_total));
    W_total = F_1 * W_total;

    # place (11,12) legs in co-domain and fuse  
    W_total = permute(W_total, (2,3), (4,5,6,7,1));
    F_1 = isomorphism(comb_space', codomain(W_total));
    W_total = F_1 * W_total;

    W_total = permute(W_total, (2,3,4),(5,6,1));
end

# insure rotational symmetry + vert/horiz flips for local peps
function make_symmetric(A::TensorMap)
    
    A_c4_sym = 1/4 * (A + _make_it_fit(permute(A,(2,3,4,1,5)), A) +
        _make_it_fit(permute(A,(3,4,1,2,5)), A) +
        _make_it_fit(permute(A,(4,1,2,3,5)), A))  
    
    @tensor A_herm_depth[-1 -2 -3 -4 -5] := conj(A_c4_sym[-1 -4 -3 -2 -5])
    
    A_sym = 1/2 * (A_c4_sym + _make_it_fit(A_herm_depth, A_c4_sym))
    
    return A_sym
end