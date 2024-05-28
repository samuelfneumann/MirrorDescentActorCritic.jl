struct REINFORCE <: AbstractPolicyGradientStyleUpdate end

function setup(
    up::REINFORCE,
    π::AbstractParameterisedPolicy,
    π_f,    # policy model
    π_θ,    # policy model parameters
    π_st,   # policy model state
    optim::Optimisers.AbstractRule,
    ::AbstractRNG,
)
    return UpdateState(up, optim, (π_optim = Optimisers.setup(optim, π_θ),))
end

function update(
    st::UpdateState{REINFORCE},
    π::AbstractParameterisedPolicy,
    π_f,    # actor policy model
    π_θ,    # actor policy model parameters
    π_st,   # actor policy model state
    s_t::AbstractArray, # Must be >= 2D
    a_t::AbstractArray,
    A_t::AbstractVector,
    γ_t::AbstractVector,
)
    up = st._update

    ∇π_θ = gradient(π_θ) do θ
        lnπ, π_st, = logprob(π, π_f, θ, π_st, s_t, a_t)
        -mean(lnπ .* A_t)
    end

    π_optim_state = st._state.π_optim
    π_optim_state, π_θ = Optimisers.update(π_optim_state, π_θ, only(∇π_θ))

    return UpdateState(
        st._update,
        st._optim,
        (π_optim = π_optim_state,),
    ), π_θ, π_st
end
