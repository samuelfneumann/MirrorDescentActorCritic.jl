"""
    DualAveragingMPI <: AbstractActorUpdate

DualAveragingMPI implementes the Dual Averaging Modified Policy Iteration algorithm from
[1]. Equivalently, this struct also implements the Mirror Descent Modified Policy Iteration
algorithm from [1].

# References

[1] Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, Rémi Munos, Matthieu
Geist. Leverage the Average: an Analysis of KL Regularization in RL. NeurIPS, 2020.
"""
struct DualAveragingMPI <: AbstractActorUpdate
    _τ::Float32
    _λ::Float32

    # Trick to ensure the policy is always slightly stochastic. This may be needed to
    # improve numerical stability when calculating entropy in the policy performance
    # gradient and when using soft action values, which also use entropy. If the policy
    # becomes deterministic, then the entropy will be -Inf.
    _ensure_stochastic::Bool
    _offset::Float32
end

function DualAveragingMPI(τ, λ; ensure_stochastic=true, offset=1f-7)
    return DualAveragingMPI(τ, λ, ensure_stochastic, offset)
end

function setup(
    up::DualAveragingMPI,
    π,
    π_f::Tabular,   # policy model
    π_θ,            # policy model parameters
    π_st,           # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,  # q function model
    qf_θ,           # q function model parameters
    qf_st,          # q function model state
    ::Nothing,
    ::AbstractRNG,
)::UpdateState{DualAveragingMPI}
    assert_uniform(π, π_θ)
    return UpdateState(
        up,
        nothing,
        NamedTuple(),
    )
end

function update(
    st::UpdateState{DualAveragingMPI},
    π::AbstractParameterisedPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    state::Int,
)
    up = st._update

    new_θ, π_st, qf_st = _update(
        up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state,
    )

    return UpdateState(st._update, nothing, NamedTuple()), new_θ, π_st, qf_st
end

# Updates using multiple states in a batch using a Dyna-style update
function update(
    st::UpdateState{DualAveragingMPI},
    π::AbstractParameterisedPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states::Vector{Int},
)
    up = st._update

    for state in states
        π_θ, π_st, qf_st = _update(
            up, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, state,
        )
    end

    return UpdateState(st._update, nothing, NamedTuple()), π_θ, π_st, qf_st
end

# Updates using multiple states in a batch using a Dyna-style update
function update(
    st::UpdateState{DualAveragingMPI},
    π::AbstractParameterisedPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    states::Matrix{Int},
)
    @assert size(states, 1) == 1
    return update(st, π, π_f, π_θ, π_st, qf, qf_f, qf_θ, qf_st, states[1, :])
end

function _update(
    up::DualAveragingMPI,
    π::SoftmaxPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    state::Int,
)
    # This function uses equations (3) - (6) in [1]
    update = function(μ_θ, qf_θ)
        logits = μ_θ
        q = qf_θ[:, state]
        μ_logits = μ_θ[:, state]
        logits[:, state] .= (up._λ .* μ_logits .+ q) ./ (up._λ + up._τ)

        infs = isinf.(logits[:, state])
        correct_infs = any(infs)
        if correct_infs
            logit_max = log(prevfloat(typemax(typeof(logits[1])))) - 1
            logit_min = nextfloat(typemin(typeof(logits[1]))) + 1
            logits[(!).(infs), state] .= logit_min
            logits[infs, state] .= logit_max
        end

        return _ensure_stochastic!(up, π, π_f, logits, state; atol=up._offset)
    end
    θ_tp1 = treemap(update, π_θ, qf_θ)

    return θ_tp1, π_st, qf_st
end

# This simplex implementation is more numerically stable than the softmax implementation
function _update(
    up::DualAveragingMPI,
    π::SimplexPolicy,
    π_f::Tabular,       # policy model
    π_θ,                # policy model parameters
    π_st,               # policy model state
    qf::DiscreteQ,
    qf_f::Tabular,      # q function model
    qf_θ,               # q function model parameters
    qf_st,              # q function model state
    state::Int,
)
    # This function uses equations (3) - (6) in [1]
    update = function(μ_θ, qf_θ)
        θ = μ_θ
        q = qf_θ[:, state]
        μ = μ_θ[:, state]
        μ_logits = log.(μ)
        logits = (up._λ .* μ_logits .+ q) ./ (up._λ + up._τ)
        probs = softmax(logits)

        infs = isinf.(probs)
        correct_infs = any(infs)
        if correct_infs
            # Project to a simplex vertex
            probs .= zero(probs[1])
            probs[infs] .= (one(probs[1]) / sum(infs))
        end

        θ[:, state] .= probs
        return _ensure_stochastic!(up, π, π_f, θ, state; atol=up._offset)
    end
    θ_tp1 = treemap(update, π_θ, qf_θ)

    return θ_tp1, π_st, qf_st
end
