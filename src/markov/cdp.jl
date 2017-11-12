#=
Tools for representing and solving dynamic programs with continuous states.

=#
#=
using BasisMatrices
import Optim


#= Types and contructors =#

struct Interp{N,TM<:AbstractMatrix,TL<:Factorization}
    basis::Basis{N}
    S::Array{Float64,N}
    Scoord::NTuple{N,Vector{Float64}}
    length::Int
    size::NTuple{N,Int}
    lb::NTuple{N,Float64}
    ub::NTuple{N,Float64}
    Phi::TM
    Phi_lu::TL
end

function Interp(basis::Basis)
    S, Scoord = nodes(basis)
    grid_length = length(basis)
    grid_size = size(basis)
    grid_lb, grid_ub = min(basis), max(basis)
    Phi = BasisMatrix(basis, Expanded(), S).vals[1]
    Phi_lu = lufact(Phi)
    interp = Interp(basis, S, Scoord, grid_length, grid_size, grid_lb, grid_ub,
                    Phi, Phi_lu)
end


struct ContinuousDP{N,K}
    f::Function
    g::Function
    discount::Float64
    shocks::Array{Float64,K}
    weights::Vector{Float64}
    x_lb::Function
    x_ub::Function
    interp::Interp{N}
end

function ContinuousDP(f::Function, g::Function, discount::Float64,
                      shocks::Array{Float64}, weights::Vector{Float64},
                      x_lb::Function, x_ub::Function,
                      basis::Basis)
    interp = Interp(basis)
    cdp = ContinuousDP(f, g, discount, shocks, weights, x_lb, x_ub, interp)
    return cdp
end


#= Methods =#

function _s_wise_max(cdp::ContinuousDP, s, C)
    function objective(x)
        out = 0.
        t = Base.tail(indices(cdp.shocks))
        for (i, w) in enumerate(cdp.weights)
            e = cdp.shocks[(i, t...)...]
            out += w * funeval(C, cdp.interp.basis, cdp.g(s, x, e))
        end
        out *= cdp.discount
        out += cdp.f(s, x)
        out *= -1
        return out
    end
    res = Optim.optimize(objective, cdp.x_lb(s), cdp.x_ub(s))
    v = -res.minimum::Float64
    x = res.minimizer::Float64
    return v, x
end

function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(indices(ss))
    for i in 1:n
        Tv[i], _ = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return Tv
end

function s_wise_max!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                     C::Vector{Float64}, Tv::Vector{Float64},
                     X::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(indices(ss))
    for i in 1:n
        Tv[i], X[i] = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return Tv, X
end

function s_wise_max(cdp::ContinuousDP, ss::AbstractArray{Float64},
                    C::Vector{Float64})
    n = size(ss, 1)
    Tv, X = Array{Float64}(n), Array{Float64}(n)
    s_wise_max!(cdp, ss, C, Tv, X)
end


function bellman_operator!(cdp::ContinuousDP, C::Vector{Float64},
                           Tv::Vector{Float64})
    Tv = s_wise_max!(cdp, cdp.interp.S, C, Tv)
    A_ldiv_B!(C, cdp.interp.Phi_lu, Tv)
    return C
end


function compute_greedy!(cdp::ContinuousDP, ss::AbstractArray{Float64},
                         C::Vector{Float64}, X::Vector{Float64})
    n = size(ss, 1)
    t = Base.tail(indices(ss))
    for i in 1:n
        _, X[i] = _s_wise_max(cdp, ss[(i, t...)...], C)
    end
    return X
end

compute_greedy!(cdp::ContinuousDP, C::Vector{Float64}, X::Vector{Float64}) =
    compute_greedy!(cdp, cdp.interp.S, C, X)


function evaluate_policy!(cdp::ContinuousDP{N}, X::Vector{Float64},
                          C::Vector{Float64}) where N
    n = size(cdp.interp.S, 1)
    ts = Base.tail(indices(cdp.interp.S))
    te = Base.tail(indices(cdp.shocks))
    A = Array{Float64}(n, n)
    A[:] = cdp.interp.Phi
    for i in 1:n
        s = cdp.interp.S[(i, ts...)...]
        for (j, w) in enumerate(cdp.weights)
            e = cdp.shocks[(j, te...)...]
            s_next = cdp.g(s, X[i], e)
            A[i, :] -= ckron(
                [vec(evalbase(cdp.interp.basis.params[k], s_next[k]))
                 for k in N:-1:1]...
            ) * cdp.discount * w
        end
    end
    A_lu = lufact(A)
    for i in 1:n
        s = cdp.interp.S[(i, ts...)...]
        C[i] = cdp.f(s, X[i])
    end
    A_ldiv_B!(A_lu, C)
    return C
end
=#
