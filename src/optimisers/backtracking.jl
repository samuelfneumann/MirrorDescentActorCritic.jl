# TODO: I need to think better about how to combined the interfaces of "regular" optimisers
# and line searches
mutable struct _Descent{F} <: Optimisers.AbstractRule
  eta::F
end

Optimisers.init(o::_Descent, x::AbstractArray) = nothing

function Optimisers.apply!(o::_Descent, state, x, dx)
  η = convert(float(eltype(x)), o.eta)

  return state, Optimisers.@lazy dx * η  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

struct BackTracking{F} <: Optimisers.AbstractRule
    eta::F
    ρ::F
    max_steps::Int
    reset_every::Int
end

function BackTracking(eta, ρ, max_steps, reset_every, R)
    eta, ρ = promote(eta, ρ)
    BackTracking{typeof(eta)}(eta, ρ, max_steps, reset_every)
end

function Optimisers.setup(o::BackTracking, ps)
    return BackTrackingState(o, o.eta, ps)
end

struct BackTrackingState{F,S}
    _rule::BackTracking
    curr_update::Int
    _opt::_Descent{F}
    _opt_st::S
end

function BackTrackingState(
    r::BackTracking, curr_update, opt::_Descent{F}, opt_st::S,
) where {F,S}
    BackTrackingState{F,S}(r, curr_update, opt, opt_st)
end

function BackTrackingState(o, eta::F, ps) where {F}
    opt = _Descent(eta)
    opt_st = Optimisers.setup(opt, ps)
    BackTrackingState(o, 1, opt, opt_st)
end

function replace(s::BackTrackingState{F,S}, opt_eta, opt_st::S) where {F,S}
    s._opt.eta = opt_eta
    return BackTrackingState(s._rule, s.curr_update + 1, s._opt, opt_st)
end

function Optimisers.update(o::BackTrackingState, ps, st, grad, f, init...)
    f0, _ = f(init..., ps, st)
    new_opt_st, new_ps = Optimisers.update(o._opt_st, ps, grad)
    f1, _ = f(init..., new_ps, st)

    step = 1
    while f1 > f0 && step <= o._rule.max_steps
        step += 1

        o._opt.eta *= o._rule.ρ

        new_opt_st, new_ps = Optimisers.update(o._opt_st, ps, grad)
        f1, _ = f(init..., new_ps, st)
    end

    opt_eta = if mod(o.curr_update, o._rule.reset_every) == 0
        o._rule.eta
    else
        o._opt.eta
    end

    return replace(o, opt_eta, new_opt_st), new_ps
end

