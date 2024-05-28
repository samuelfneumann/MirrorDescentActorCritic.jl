mutable struct CliffWorld{
        T<:Real,
        A<:AbstractSpace,
        O<:AbstractSpace,
        R<:AbstractRNG,
} <: AbstractEnvironment
    const _observationspace::O
    const _actionspace::A
    const _γ::Float32
    _rng::R
    _current_state::Int
    const _rows::Int
    const _cols::Int
    const _int_obs::Bool
    const _exploring_starts::Bool
    _last_transition_off_cliff::Bool

    function CliffWorld{T}(
        rng::R, γ, rows, cols, int_obs, exploring_starts,
    ) where {T<:Real,R<:AbstractRNG}

        if int_obs
            obs_space = Discrete{Int}(rows * cols)
        else
            low = zeros(T, rows * cols)
            high = ones(T, rows * cols)
            obs_space = Box{Int}(low, high)
        end

        action_space = Discrete(4)
        O = typeof(obs_space)
        A = typeof(action_space)

        p = new{T,A,O,R}(
            obs_space, action_space, γ, rng, rows, rows, cols, int_obs, exploring_starts,
            false,
        )

        start!(p)
        return p
    end
end

function CliffWorld(
    rng::AbstractRNG; γ=1f0, rows=4, cols=12, int_obs=true, exploring_starts=false,
)
    CliffWorld{Int}(
        rng; γ=γ, rows=rows, cols=cols, int_obs=int_obs, exploring_starts=exploring_starts,
    )
end

function CliffWorld{T}(
    rng::AbstractRNG; γ=1f0, rows=4, cols=12, int_obs=true, exploring_starts=false,
) where {T<:Real}
    CliffWorld{T}(rng, γ, rows, cols, int_obs, exploring_starts)
end

function _to_grid(c::CliffWorld{T}; vec=false) where {T}#TODO
    if vec
        grid = spzeros(T, c._rows * c._cols)
    else
        grid = spzeros(T, c._rows, c._cols)
    end
    grid[c._current_state] = one(T)
    return grid
end

reward(c::CliffWorld) = c._last_transition_off_cliff ? -100f0 : -1f0
γ(c::CliffWorld) = isterminal(c) ? zero(c._γ) : c._γ
observation_space(c::CliffWorld) = c._observationspace
action_space(c::CliffWorld) = c._actionspace
isterminal(c::CliffWorld) = _at_goal(c)

function start!(c::CliffWorld{T}) where {T}
    if c._exploring_starts
        non_cliff_states = []
        for i in 1:observation_space(c).n[1]
            if !_on_cliff(c, i)
                push!(non_cliff_states, i)
            end
        end
        c._current_state = rand(c._rng, non_cliff_states)
    else
        c._current_state = c._rows
    end
    c._last_transition_off_cliff = false

    return _get_obs(c)
end

function _to_index(c::CliffWorld)
    return _to_index(c, c._current_state)
end

function _to_index(c::CliffWorld, i)
    row = ((i - 1) - c._rows * ((i - 1) ÷ c._rows)) + 1
    col = ((i - 1) ÷ c._rows) + 1

    return (col, row)
end

_up(c::CliffWorld) = -1
_down(c::CliffWorld) = 1
_right(c::CliffWorld) = c._rows
_left(c::CliffWorld) = -c._rows

function envstep!(c::CliffWorld, action)
    check_contains_action(c, action)

    u = _discrete_action(c, action)

    last_state = c._current_state
    if _in_first_col(c) && u == _left(c)
    elseif _in_last_col(c) && u == _right(c)
    elseif _in_first_row(c) && u == _up(c)
    elseif _in_last_row(c) && u == _down(c)
    else
        c._current_state += u
    end

    # Cache whether the agent jumped off the cliff
    c._last_transition_off_cliff = _on_cliff(c)

    # If the agent did jump off the cliff, transition to the start state
    if _on_cliff(c)
        c._current_state = c._rows
    end

    return _get_obs(c), reward(c), isterminal(c), γ(c)
end

function Base.show(io::IO, c::CliffWorld{T}) where {T}
    print(io, "CliffWorld{$T}")
end

function _get_obs(c::CliffWorld{T}) where {T}
    return if c._int_obs
        return [c._current_state]
    else
        _to_grid(c)
    end
end

_at_goal(c::CliffWorld) = c._current_state == c._cols * c._rows
_at_goal(c::CliffWorld, col::Int, row::Int) = col * row == c._cols * c._rows

function _on_cliff(c::CliffWorld)
    col, row = _to_index(c)
    return _on_cliff(c, col, row)
end

function _on_cliff(c::CliffWorld, col::Int, row::Int)
    return 1 < col < c._cols && row == c._rows
end

function _on_cliff(c::CliffWorld, i::Int)
    _on_cliff(c, _to_index(c, i)...)
end

function _discrete_action(c::CliffWorld, action)
    if action isa AbstractArray
        @assert length(action) == 1
        action = first(action)
    end
    actions = [_up, _down, _right, _left]
    return actions[action](c)
end

function _in_first_col(c::CliffWorld)
    return c._current_state <= c._rows
end

function _in_last_col(c::CliffWorld)
    return (
        c._current_state > (c._rows * c._cols) - c._rows
    )
end

function _in_first_row(c::CliffWorld)
    return mod(c._current_state, c._rows) == 1
end

function _in_last_row(c::CliffWorld)
    return mod(c._current_state, c._rows) == 0
end
