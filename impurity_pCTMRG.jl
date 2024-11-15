using Logging, TensorKit, MPSKit, TensorOperations, Test, ProgressMeter,MPSKitModels
include("some_functions.jl")

using ProgressBars
using Zygote
using OptimKit


function _make_it_fit(
    y::AbstractTensorMap{S,N₁,N₂}, x::AbstractTensorMap{S,N₁,N₂}
) where {S<:IndexSpace,N₁,N₂}
    for i in 1:(N₁ + N₂)
        if space(x, i) ≠ space(y, i)
            f = unitary(space(x, i) ← space(y, i))
            y = permute(
                ncon([f, y], [[-i, 1], [-(1:(i - 1))..., 1, -((i + 1):(N₁ + N₂))...]]),
                (Tuple(1:N₁), Tuple((N₁ + 1):(N₁ + N₂))),
            )
        end
    end
    return y
end

############### Initial tensor ##################

Ising_Tc = 2.0/log(1.0+sqrt(2))
Potts_Tc = 1.0/log(1.0+sqrt(3))

function initialize_Ising(β::Float64)
    V = ℂ^2
    A_Ising = TensorMap(zeros,V⊗V←V⊗V)
    
    s(ind) = 2*ind-3
    ss(i,j) = s(i)*s(j)
    for i=1:2
        for j=1:2
            for k=1:2
                for l=1:2
                    E = -(ss(i,j)+ss(j,k)+ss(k,l)+ss(l,i))
                    A_Ising[i,j,k,l] = exp(-β*E)
                end
            end
        end
    end
    return A_Ising
end

function initialize_Ising_sym(β)
    V = Vect[Z2Irrep](0=>1,1=>1)
    Ising = zeros(2,2,2,2)
    c = cosh(β)
    s = sinh(β)
    for i=1:2
        for j=1:2
            for k=1:2
                for l=1:2
                    if (i+j+k+l)==4
                        Ising[i,j,k,l]=2*c*c
                    elseif (i+j+k+l)==6
                        Ising[i,j,k,l]=2*c*s
                    elseif (i+j+k+l)==8
                        Ising[i,j,k,l]=2*s*s
                    end
                end
            end
        end
    end
    return TensorMap(Ising,V⊗V←V⊗V)
end

function initialize_Potts(q::Int64, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros,V⊗V←V⊗V)
    
    for i=1:q
        for j=1:q
            for k=1:q
                for l=1:q
                    E = -(Int(i==j)+Int(j==k)+Int(k==l)+Int(l==i))
                    A_potts[i,j,k,l] = exp(-β*E)
                end
            end
        end
    end
    return A_potts
end

function initialize_Ising_mag(β::Float64,impurity=false,h=1e-12)
    V = ℂ^2
    A_Ising = TensorMap(zeros,V⊗V←V⊗V)
    
    s(ind) = 2*ind-3
    ss(i,j) = s(i)*s(j)
    for i=1:2
        for j=1:2
            for k=1:2
                for l=1:2
                    sum_s = (s(i)+s(j)+s(k)+s(l))/4
                    E = -(ss(i,j)+ss(j,k)+ss(k,l)+ss(l,i))
                    #small magnetic field
                    E -= sum_s*h
                    A_Ising[i,j,k,l] = exp(-β*E)
                    if impurity
                        A_Ising[i,j,k,l] = sum_s*exp(-β*E)
                    end
                end
            end
        end
    end
    return A_Ising
end

############### Levin TRG ##################

function get_diag(mat::AbstractTensorMap)
    len = Int(sqrt(length(mat.data)))
    return [mat[i,i] for i=1:len]
end

function decomposeT(T::AbstractTensorMap,χ::Int64,N₁::Tuple,N₂::Tuple)
    U, S, V, _ϵ = tsvd(T,N₁,N₂;trunc=truncdim(χ));
    sqS = sqrt(S)
    U = U*sqS
    V = sqS*V;
    return U,V
end


function Trace_Tensor(T::AbstractTensorMap)
    return @tensor T[1 2; 1 2]
    # return only(ncon([T],[[1,2,1,2]]).data)
end




function contract_4tensors(A1::AbstractTensorMap,A2::AbstractTensorMap,A3::AbstractTensorMap,A4::AbstractTensorMap)
    # return only(ncon([A1,A2,A3,A4],[[2,7,1,5],[1,8,2,6],[4,5,3,7],[3,6,4,8]]).data)
    return @tensor A1[2 7; 1 5] * A2[1 8; 2 6] * A3[4 5; 3 7] * A4[3 6; 4 8]
end

# function Levin_TRG_1step(T::AbstractTensorMap,χ::Int64)
#     S1,S3 = decomposeT(T,χ,(1,2),(3,4))
#     S2,S4 = decomposeT(T,χ,(2,3),(4,1))
#     Tnew = ncon([S1,S2,S3,S4],[[1,4,-1],[2,1,-2],[-3,3,2],[-4,4,3]])
#     Tnew = permute(Tnew,(3,4),(1,2))
#     trace_norm = Trace_Tensor(Tnew)
#     return Tnew/trace_norm, trace_norm
# end

function Levin_TRG_1step(T::AbstractTensorMap,χ::Int64)
    S1,S3 = decomposeT(T,χ,(1,2),(3,4))
    S2,S4 = decomposeT(T,χ,(2,3),(4,1))
    Tnew = ncon([S1,S2,S3,S4],[[1,4,-1],[3,1,-2],[-3,2,3],[-4,4,2]])
    Tnew = permute(Tnew,(3,4),(1,2))
    trace_norm = norm(Trace_Tensor(Tnew))
    return Tnew/trace_norm, trace_norm
end



function Levin_TRG(A_initial::AbstractTensorMap,χ::Int64,step::Int64;)
    A = A_initial/norm(Trace_Tensor(A_initial))
    time_list = []
    norms = [norm(Trace_Tensor(A_initial))]
    for i=1:step
        @info i
        start = time()
        A, norm = Levin_TRG_1step(A,χ)
        push!(time_list,time()-start)
        push!(norms, norm)
    end
    return A,time_list, norms
end

function cft_xn(A::AbstractTensorMap;return_sigma=false, klein=false, Ising=true)
    row = ncon([A,A],[[1,-1,2,-3],[2,-2,1,-4]]);
    val,vec = eigh(row,(1,2),(3,4);)
    spec = (real(get_diag(val))[end:-1:1])[1:8]
    @info "x1: $(log(spec[1]/spec[2])/pi)"
    @info "x2: $(log(spec[1]/spec[3])/pi)\n"
    if klein
        sp = space(vec)
        new_shape = TensorMap(zeros,sp[1]⊗dual(sp[2])←dual(sp[3]))
        vec_new = _make_it_fit(vec,new_shape)
        @tensor klein[-1] := vec_new[1 1;-1]
        data = klein.data[end:-1:1]
        @info data[1:6].^2
        if Ising
            #Ising(exact value)
            @info [sqrt(2)/2+1,0,1-sqrt(2)/2,0]
        else
            #3-State Potts
            @info [sqrt(3+6/sqrt(5)),0,0,sqrt(3-6/sqrt(5))]
        end
        return data[1:6].^2
    end
    if return_sigma
        return (log.(spec[1]./spec[2:end])./pi)
    end
end

############### pCTMRG ##################

function truncate_U(U::AbstractTensorMap,χ::Int64)
    t = spacetype(U)
    B = similar(U,codomain(U),t(χ))
    B[][:,:,1:χ] = U[][:,:,1:χ]
    return B
end

function find_Uh(C::AbstractTensorMap,ev::AbstractTensorMap,χ::Int64)
    @tensor cont[-1 -2;-3　-4] := ev[3 -1;1 4]*conj(ev[3 -3;2 4])*C[1 -2;6 5]*conj(C[2 -4;6 5])
    U,_,_,_ = tsvd(cont,(1,2),(3,4);trunc=truncdim(χ))
    return U
end
function find_Uv(C::AbstractTensorMap,eh::AbstractTensorMap,χ::Int64)
    @tensor cont[-1 -2;-3 -4] := eh[-1 1;5 6]*conj(eh[-3 2;5 6])*C[-2 3;4 1]*conj(C[-4 3;4 2])
    U,_,_,_ = tsvd(cont,(1,2),(3,4);trunc=truncdim(χ))
    return U
end

# function find_Uh(C::AbstractTensorMap,ev::AbstractTensorMap,χ::Int64,trunc_below=1e-14)
#     @tensor cont[-1 -2;-3　-4] := ev[1 -1;5 2]*conj(ev[1 -3;6 2])*C[5 -2;3 4]*conj(C[6 -4;3 4])
#     U,S,V,_ϵ = tsvd(cont,(1,2),(3,4);trunc=truncbelow(trunc_below))
#     if dim(domain(U)) > χ
#         return truncate_U(U,χ)
#     else
#         return U
#     end
# end
# function find_Uv(C::AbstractTensorMap,eh::AbstractTensorMap,χ::Int64,trunc_below=1e-14)
#     @tensor cont[-1 -2;-3 -4] := eh[-1 5;2 1]*conj(eh[-3 6;2 1])*C[-2 3;4 5]*conj(C[-4 3;4 6])
#     U,S,V,_ϵ = tsvd(cont,(1,2),(3,4);trunc=truncbelow(trunc_below))
#     if dim(domain(U)) > χ
#         return truncate_U(U,χ)
#     else
#         return U
#     end
# end



function find_U(C::AbstractTensorMap,e::AbstractTensorMap,χ::Int64,direction::String)
    if direction=="h"
        return find_Uh(C,e,χ)
    elseif direction=="v"
        return find_Uv(C,e,χ)
    else
        @error "The direction should be \"h\" or \"v\""
    end
end

struct pCTMRG
    A::AbstractTensorMap
    C::AbstractTensorMap
    ev::AbstractTensorMap
    eh::AbstractTensorMap
    W::AbstractTensorMap
end
# with imurity tensor W
pCTMRG(x,y) = pCTMRG(x,x,x,x,y)
# without imurity tensor W
pCTMRG(x) = pCTMRG(x,x,x,x,x)


# function copy_pCTMRG(self::pCTMRG)
#     new = pCTMRG(self.A,self.W)
#     new.C = copy(self.C)
#     new.ev = copy(self.ev)
#     new.eh = copy(self.eh)
#     new.W = copy(self.W)
#     return new
# end

function contract_x(self::pCTMRG,χ::Int64,impurity=true)
    U = find_U(self.C,self.ev,χ,"h");
    @tensor Cnew[-1 -2;-3 -4] := self.ev[-1 3;2 5]*self.C[2 1;-3 4]*U[5 4;-4]*conj(U[3 1;-2])
    @tensor ehnew[-1 -2;-3 -4] := self.eh[2 1; -3 3]*self.A[-1 4;2 5]*U[5 3;-4]*conj(U[4 1;-2])
    normC = Trace_Tensor(Cnew)
    normeh = Trace_Tensor(ehnew)

    if impurity
        if length(self.W.codom.spaces) == 3
            @error "W have 6 legs. Contract_y first to bundle the two legs of W"
        else
            @tensor Wnew[-1 -2;-3 -4] := self.ev[-1 3;2 5]*self.W[2 1;-3 4]*U[5 4;-4]*conj(U[3 1;-2])
        end
    end
    if impurity
        return pCTMRG(self.A,Cnew/normC,self.ev,ehnew/normeh,Wnew/normC)
    else
        return pCTMRG(self.A,Cnew/normC,self.ev,ehnew/normeh,self.A)
    end
end


function contract_y(self::pCTMRG,χ::Int64,impurity=true)
    U = find_U(self.C,self.eh,χ,"v")
    @tensor Cnew[-1 -2;-3 -4] := self.eh[3 2;5 -4]*self.C[1 -2;4 2]*U[5 4;-3]*conj(U[3 1;-1])
    @tensor evnew[-1 -2;-3 -4] := self.ev[3 -2;1 2]*self.A[5 2;4 -4]*U[4 1;-3]*conj(U[5 3 -1])
    normC = Trace_Tensor(Cnew)
    normev = Trace_Tensor(evnew)

    if impurity
        if length(self.W.codom.spaces) == 3
            @tensor Wnew[-1 -2;-3 -4] := self.W[1 2 -2;3 4 -4]*U[4 3;-3]*conj(U[2 1;-1])
        else
            @tensor Wnew[-1 -2;-3 -4] := self.eh[3 2;5 -4]*self.W[1 -2;4 2]*U[5 4;-3]*conj(U[3 1;-1])
        end
        normW = Trace_Tensor(Wnew)
    end
    if impurity
        return pCTMRG(self.A,Cnew/normC,evnew/normev,self.eh,Wnew/normC)
    else
        return pCTMRG(self.A,Cnew/normC,evnew/normev,self.eh,self.A)
    end
end

function expand(self::pCTMRG,χ::Int64,step::Int64;impurity=true,verbosity=false)
    if step < 0
        @error "step should be non-negative integer"
    end
    penv = self
    for i=1:step
        if verbosity
            # @info "step $i / $step"
            println("...IN PROGRESS STEP i = ", i)
        end
        penv = contract_y(penv,χ,impurity)
        penv = contract_x(penv,χ,impurity)
    end
    if verbosity
        @info "pCTMRG Finished!" 
    end
    return penv;
end


# function show_shapes(self::pCTMRG)
#     @info "A size  : (Lx,Ly) = $(self.sizes[1,:]) \tshape: $(space(self.A))"
#     @info "C size  : (Lx,Ly) = $(self.sizes[2,:]) \tshape: $(space(self.C))"
#     @info "ev size : (Lx,Ly) = $(self.sizes[3,:]) \tshape: $(space(self.ev))"
#     @info "eh size : (Lx,Ly) = $(self.sizes[4,:]) \tshape: $(space(self.eh))"
#     @info "W size  : (Lx,Ly) = $(self.sizes[5,:]) \tshape: $(space(self.W))"
# end


################## pCTMRG ######################

function fg(f, A)
    f_out, g = Zygote.withgradient(f, A)
    
    return f_out, make_symmetric(g[1])
end 

my_inner(x, y1, y2) = real(dot(y1, y2));
my_retract(x, d, α) = (make_symmetric(x + α * d), make_symmetric(d));


function expectation_value_W(self::pCTMRG,unit_cell=1)
    if unit_cell==1
        return Trace_Tensor(self.W)
    elseif unit_cell==2
        return contract_4tensors(self.W,self.C,self.C,self.C)/contract_4tensors(self.C,self.C,self.C,self.C)
    else
        @error "The argument unit_cell should be one or two" 
    end
end

function environments_twosites(self::pCTMRG,χ::Int64)
    tmp = contract_x(self,χ,false)
    @tensor E[-1 -2 -3;-4 -5 -6] := tmp.C[1 5;2 3]*tmp.eh[-4 3;-1 4]*tmp.eh[-5 4;-2 5]*self.ev[2 -6;1 -3]
    return E;
end

function cal_env(A_peps::AbstractTensorMap,χ::Int64,L::Int64;verbosity=false)
    A = make_2d_tensor(A_peps,false);
    A_pCTMRG = pCTMRG(A)
    @debug "Computing the environment tensor..."
    if L < 3
        @error "L should be larger than 2"
    else
        A_pCTMRG = expand(A_pCTMRG,χ,L-3;impurity=false,verbosity=verbosity)
    end
    env = environments_twosites(A_pCTMRG,χ)
    @debug "Finished computing the environment tensor!"
    return env;
end

function cost_local_update(A_peps::AbstractTensorMap,L::Int64,χ::Int64,opp_one_site::AbstractTensorMap,opp_twosite::AbstractTensorMap;verbosity=false)
    A = make_2d_tensor(A_peps,false);
    env = cal_env(A_peps,χ,L;verbosity=verbosity)
    W = get_two_site_impurity(A_peps,opp_one_site,opp_twosite);
    @tensor AA[-1 -2 -3;-4 -5 -6] := A[-1 -3;-4 1] * A[-2 1;-5 -6];
    @tensor W_ex = env[1,2,3,4,5,6]*W[1,2,3,4,5,6]
    @tensor AA_ex = env[1,2,3,4,5,6]*AA[1,2,3,4,5,6]
    return real(W_ex/AA_ex);
end

function optimize_local_LBFGS(A_peps::AbstractTensorMap,λ,χ::Int64,L::Int64;gradtol=1e-4,verbosity=2,maxiter=80)
    O_one,O_two = (-λ) * X,(-1)*ZZ
    Ã = copy(A_peps)
    # define the function
    opt_fun(A) = cost_local_update(A,L,χ,O_one,O_two);
    opt_fg(A) = fg(opt_fun, A);
    Zygote.refresh()
    A_res, opt_fun_res, g_res, numfg, history = optimize(opt_fg,Ã,LBFGS(10;verbosity=verbosity,gradtol=gradtol,maxiter=maxiter);inner=my_inner,retract=my_retract);
    A_peps = A_res
    return A_peps,history
end


function optimize_local(A_peps::AbstractTensorMap,λ,χ::Int64,L::Int64;gradtol=1e-4,verbosity=2,maxiter=80)
    O_one,O_two = (-λ) * X,(-1)*ZZ
    Ã = copy(A_peps)
    # define the function
    opt_fun(A) = cost_local_update(A,L,χ,O_one,O_two);
    opt_fg(A) = fg(opt_fun, A);
    Zygote.refresh()
    A_res, opt_fun_res, g_res, numfg, history = optimize(opt_fg,Ã,ConjugateGradient(;verbosity=verbosity,gradtol=gradtol,maxiter=maxiter);inner=my_inner,retract=my_retract);
    A_peps = A_res
    return A_peps,history
end

function optimize_local_XY(A_peps::AbstractTensorMap,χ::Int64,L::Int64;gradtol=1e-4,verbosity=2,maxiter=80)
    O_one,O_two = (-0.0) * X,-(0.25)*(XX+YY)
    Ã = copy(A_peps)
    # define the function
    function opt_fun(A)
        B  = _make_it_fit(permute(A,(2,3,4,1,5),()), A)
        return 1/2*(cost_local_update(A,L,χ,O_one,O_two)+cost_local_update(B,L,χ,O_one,O_two))
    end
    opt_fg(A) = fg(opt_fun, A);
    Zygote.refresh()
    A_res, opt_fun_res, g_res, numfg, history = optimize(opt_fg,Ã,ConjugateGradient(;verbosity=verbosity,gradtol=gradtol,maxiter=maxiter);inner=my_inner,retract=my_retract);
    A_peps = A_res
    return A_peps,history
end



function optimize_local_Heisenberg(A_peps::AbstractTensorMap,χ::Int64,L::Int64;gradtol=1e-4,verbosity=2,maxiter=80)
    O_one,O_two = (-0.0) * X,-(0.25)*(XX-YY+ZZ)
    Ã = copy(A_peps)
    # define the function
    opt_fun(A) = cost_local_update(A,L,χ,O_one,O_two);
    opt_fg(A) = fg(opt_fun, A);
    Zygote.refresh()
    A_res, opt_fun_res, g_res, numfg, history = optimize(opt_fg,Ã,ConjugateGradient(;verbosity=verbosity,gradtol=gradtol,maxiter=maxiter);inner=my_inner,retract=my_retract);
    A_peps = A_res
    return A_peps,history
end

function expand_d(A_peps::AbstractTensorMap,d::Int64)
    V_ini = space(A_peps)[1]
    iso_in = isometry(ℂ^d←V_ini)
    iso_out = isometry((ℂ^d)'←V_ini')
    iso_phy = isometry((ℂ^2)'←(ℂ^2)')
    @tensor A_new[-1 -2 -3 -4 -5] := A_peps[1 2 3 4 5]*iso_in[-1 1]*iso_in[-2 2]*iso_out[-3 3]*iso_out[-4 4]*iso_phy[5 -5]
    return A_new
end

function pair_PBC(L)
    N = L^2
    data = zeros(Int,(2*N,2))
    for i=1:N
        if i%L==0
            data[i,:] = [i,i-L+1]
        else
            data[i,:] = [i,i+1]
        end
        data[N+i,:] = [i,(i+L-1)%N+1]
    end
    return data
end

# function ED_TF(λ,L)
#     N = L^2
#     lat = FiniteChain(N)
#     Otwo = -1*ZZ
#     Oone = -λ*X
#     pair = pair_PBC(L)

#     H_TF = @mpoham sum(1:2*N) do i
#         return Otwo{lat[pair[i,1]],lat[pair[i,2]]}
#         end+sum(1:N) do i
#             return Oone{lat[i]}
#         end;
#     result = exact_diagonalization(H_TF)
#     return result[1][1]/N
# end
    