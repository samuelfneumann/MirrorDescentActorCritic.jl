"""
    struct ValueAgent{P,QM,QS,ER,RNG} <: AbstractAgent

A `ValueAgent` is a value-based agent which uses an action-value function to induce a
policy.
"""
mutable struct ValueAgent{P,QM,QMΘ,QMS,QUS,ER,RNG} <: AbstractAgent where {
    P<:AbstractValuePolicy,
    QM,                     # Value function model
    QMΘ,                    # Value function model parameters
    QMS,                    # Value function model state
    QS<:UpdateState,        # Value function update state
    ER<:AbstractReplay,
    RNG<:AbstractRNG
}
    _policy::P

    _q̂::DiscreteQ
    _q̂_model::QM
    _q̂_θ::QMΘ
    _q̂_st::QMS
    _q̂_update_st::QUS

    _q̂_target_θ::QMΘ
    _q̂_target_st::QMS
    _q̂_target_refresh_steps::Int
    _q̂_polyak_avg::Float32
    _use_target_nets::Bool

    _buffer::ER

    _batch_size::Int

    _current_step::Int
    _steps_before_learning::Int

    _rng::RNG

    _is_training::Bool

    function ValueAgent(
        seed::Integer,
        env::AbstractEnvironment,
        policy::P,
        q̂::DiscreteQ,
        q̂_model::QM,
        q̂_optim,
        q̂_update,
        q̂_target_refresh_steps,
        q̂_polyak_avg,
        buffer::ER;
        batch_size,
        steps_before_learning,
    ) where {P,QM,ER}
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        RNG = typeof(rng)

        # Initialize q-function model
        q̂_θ, q̂_st = Lux.setup(Lux.replicate(rng), q̂_model)

        # Initialize q-function update
        q̂_update_st = setup(q̂_update, policy, q̂, q̂_model, q̂_θ, q̂_st, q̂_optim, seed)

        # Ensure parameters for target nets are valid
        @assert q̂_target_refresh_steps > 0
        @assert 0 < q̂_polyak_avg <= 1

        # Check if target nets should be used
        use_target_nets = q̂_target_refresh_steps != 1 || q̂_polyak_avg != 1f0
        q̂_target_θ, q̂_target_st = if use_target_nets
            deepcopy(q̂_θ), deepcopy(q̂_st)
        else
            q̂_target_θ, q̂_target_st
        end

        QMS = typeof(q̂_st)
        QMΘ = typeof(q̂_θ)
        QUS = typeof(q̂_update_st)

        agent = new{P,QM,QMΘ,QMS,QUS,ER,RNG}(
            policy,
            q̂,
            q̂_model,
            q̂_θ,
            q̂_st,
            q̂_update_st,
            q̂_target_θ,
            q̂_target_st,
            q̂_target_refresh_steps,
            q̂_polyak_avg,
            use_target_nets,
            buffer,
            batch_size,
            0,
            steps_before_learning,
            rng,
            true,
        )

        # Set the agent to training mode
        train!(agent)
        return agent
    end
end

function ValueAgent(
    seed::Integer,
    env::AbstractEnvironment,
    policy::P,
    q̂::DiscreteQ,
    q̂_model::QM,
    q̂_optim,
    q̂_update,
    buffer::ER;
    batch_size,
    steps_before_learning,
) where {P,QM,ER}
    return ValueAgent(seed, env, policy, q̂, q̂_model, q̂_optim, q̂_update, 1, 1f0, buffer,
        batch_size, steps_before_learning)
end


function train!(agent::ValueAgent)::Nothing
    agent._is_training = true
    agent._q̂_st = train(agent._q̂_st)
    return nothing
end

function eval!(agent::ValueAgent)::Nothing
    agent._is_training = false
    agent._q̂_st = eval(agent._q̂_st)
    return nothing
end

function get_qs(ag::ValueAgent, s_t; from_target)
    return if !from_target
        qs, ag._q̂_st = predict(ag._q̂, ag._q̂_model, ag._q̂_θ, ag._q̂_st, s_t)
    else
        qs, ag._q̂_target_st = predict(
            ag._q̂, ag._q̂_model, ag._q̂_target_θ, ag._q̂_target_st, s_t,
        )
    end
end

function select_action(agent::ValueAgent, s_t)
    qs, agent._q̂_st = get_qs(agent, s_t; from_target=false)
    if !agent._is_training
        return mode(agent._policy, qs)
    else
        return sample(agent._policy, agent._rng, qs)
    end
end

function start!(agent::ValueAgent, s_0)::Nothing
    if !agent._is_training
        @warn "calling start! on an agent in evaluation mode is a no-op, returning..."
        return nothing
    end

    return nothing
end

function step!(ag::ValueAgent, s_t, a_t, r_tp1, s_tp1, γ_tp1)::Nothing
    if !ag._is_training
        @warn "calling step! on an ag in evaluation mode is a no-op, returning..."
        return
    end
    ag._current_step += 1

    # Add transition to the replay buffer
    push!(ag._buffer, s_t, a_t, [r_tp1], s_tp1, [γ_tp1])

    if ag._current_step < ag._steps_before_learning
        # Only update once we have taken a sufficient number of steps in the replay buffer
        return
    end

    # Sample from replay buffer
    s_t, a_t, r_tp1, s_tp1, γ_tp1 = rand(ag._rng, ag._buffer, ag._batch_size)

    if ag._batch_size == 1
        # Unsqueeze batch dimension
        r_tp1 = [r_tp1]
        γ_tp1 = [γ_tp1]
        s_t = unsqueeze(s_t; dims = ndims(s_t) + 1)
        a_t = unsqueeze(a_t; dims = ndims(a_t) + 1)
        s_tp1 = unsqueeze(s_tp1; dims = ndims(s_tp1) + 1)
    end

    # Policy Evaluation
    ag._q̂_update_st, ag._q̂_θ, ag._q̂_st, ag._q̂_target_st = update(
        ag._q̂_update_st, ag._policy, ag._q̂, ag._q̂_model, ag._q̂_θ, ag._q̂_st, ag._q̂_target_θ,
        ag._q̂_target_st, s_t, a_t, r_tp1, s_tp1, γ_tp1,
    )

    # Update target network
    if ag._use_target_nets
        if ag._current_step % ag._q̂_target_refresh_steps == 0
            ag._q̂_target_θ = polyak(ag._q̂_polyak_avg, ag._q̂_target_θ, ag._q̂_θ)
        end
    end

    return nothing
end

function stop!(agent::ValueAgent, r_T, s_T, γ_T)::Nothing
    if !agent._is_training
        @warn "calling stop! on an agent in evaluation mode is a no-op, returning..."
        return nothing
    end

    return nothing
end
