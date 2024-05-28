"""
    Gridworld <: AbstractEnvironment

A gridworld environment with a number of starting and goal states.

# Description
In this tabular environment, the agent will start in some (x, y)
position - possibly randomly from the set of all starting positions.
The agent can move left, right, up, and down. Actions that would take
the agent off the grid leave the agent in place. Certain states are
terminal/goal states. Upon entering one of these states, the episode
ends.

# State and Observations
The state/observations returned by the Gridworld can be either onehot
encodings of the (x, y) position of the agent in the gridworld or
the (x, y) positions themselves. Gridworlds are tabular environments,
and so the onehot encoding determines exactly where the agent is in
the environment, and is a somewhat easier problem.

# Actions
Actions are discrete in the set (1, 2, 3, 4) and have the following
interpretations:

Action | Meaning
-------|------------------------
   1   | Move up
   2   | Move right
   3   | Move down
   4   | Move left

# Goal/Rewards
The goal of the agent is to end the episode as quickly as possible by
entering a terminal/goal state. A reward of -1 is given on all
transitions, except the transition to a goal state, when a reward of 0
is given.

# Fields
- `obsspace::AbstractSpace`: the observation space
- `actionspace::AbstractSpace`: the action space
- `rows::Int`:: see constructor
- `cols::Int`:: see constructor
- `onehot::Bool`: see constructor
"""
mutable struct Gridworld{
        S<:Union{AbstractFloat, Bool},  # State observation type: AbstractArray{S}
        A<:AbstractSpace,
        O<:AbstractSpace,
        R<:AbstractRNG,
} <: AbstractEnvironment
    _x::Int
    _y::Int
    _rows::Int
    _cols::Int

    _startxs::Vector{Int}
    _startys::Vector{Int}
    _goalxs::Vector{Int}
    _goalys::Vector{Int}

    _obsspace::O
    _actionspace::A

    _reward::Float32
    _γ::Float32
    _rng::R
    _onehot::Bool # Whether observations should be (x, y) or onehot encodings

    function Gridworld(
        x,
        y,
        rows,
        cols,

        startxs,
        startys,
        goalxs,
        goalys,

        obsspace::O,
        actionspace::A,

        reward,
        γ,
        rng::R,
        onehot,
        use_floating_point_obs,
    ) where {A,O,R}
        S = use_floating_point_obs ? Float32 : Bool

        if S !== eltype(obsspace)
            T = eltype(obsspace)
            error("expected state observation to be $T but got $S")
        end

        return new{S,A,O,R}(
            x,
            y,
            rows,
            cols,

            startxs,
            startys,
            goalxs,
            goalys,

            obsspace,
            actionspace,

            reward,
            γ,
            rng,
            onehot,
        )

    end
end

"""
    Gridworld(rng::AbstractRNG; kwargs...)

Constructor for the gridworld environment.

The gridworld may have many starting positions, determined by `startxs` and `startys`
respectively. These two vectors must have the same length. The starting position is
determine randomly from `(startxs[i], startys[i])` every time `start!` is called.

Similarly, multiple goal positions can be specified with `goalxs` and `goalys`, which also
must be of the same length.
"""
function Gridworld(
    rng::AbstractRNG;
    rows = 10,
    cols = 10,
    startxs = [1],
    startys = [1],
    goalxs = nothing,
    goalys = nothing,
    γ = 1.0,
    onehot = true,
    use_floating_point_obs = true,
)
    # Ensure the number of rows and columns are more than 1
    rows < 1 && error("rows must be larger than 1")
    cols < 1 && error("cols must be larger than 1")

    if goalxs === nothing
        goalxs = [cols]
    end
    if goalys ===  nothing
        goalys = [rows]
    end

    # Check to ensure that the start positions are legal
    if length(startxs) != length(startys)
        error("start x positions should have the same length as start y positions")
    end
    for i = 1:length(startxs)
        startx = startxs[i]
        starty = startys[i]
        startx < 1 && error("startx must be larger than 1")
        startx > cols && error("startx must not exceed cols ($cols)")
        starty < 1 && error("starty must be larger than 1")
        starty > rows && error("starty must not exceed rows ($rows)")
    end

    # Check to ensure that the goal positions are legal
    if length(goalxs) != length(goalys)
        error("goal x positions should have the same length as goal y positions")
    end
    for i = 1:length(goalxs)
        goalx = goalxs[i]
        goaly = goalys[i]
        goalx < 1 && error("goalx must be larger than 1")
        goalx > cols && error("goalx must not exceed cols ($cols)")
        goaly < 1 && error("goaly must be larger than 1")
        goaly > rows && error("goaly must not exceed rows ($rows)")
    end

    # Create observation and action spaces
    S = use_floating_point_obs ? Float32 : Bool
    low = zeros(rows * cols)
    high = ones(rows * cols)
    obsspace = Box{S}(low, high)
    actionspace = Discrete(4)

    g = Gridworld{F}(
        0,
        0,
        rows,
        cols,
        startxs,
        startys,
        goalxs,
        goalys,
        obsspace,
        actionspace,
        0f0,
        γ,
        rng,
        onehot,
        use_floating_point_obs,
    )
    start!(g)
    return g
end

function start!(g::Gridworld{S})::AbstractArray{S} where {S<:Number}
    ind = rand(g._rng, UInt) % length(g._startxs) + 1
    g._x = g._startxs[ind]
    g._y = g._startys[ind]

    return _obs(g)
end

function envstep!(g::Gridworld, action)
    check_contains_action(g, action)

    if action isa AbstractArray
        action = action[1]
    end

    if action == 1 && g._y > 1
        # Move down
        g._y -= 1
    elseif action == 2 && g._x < g._cols
        # Move right
        g._x += 1
    elseif action == 3 && g._y < g._rows
        # Move up
        g._y += 1
    elseif action == 4 && g._x > 1
        # Move left
        g._x -= 1
    end

    return _obs(g), reward(g), isterminal(g), γ(g)
end

function reward(g::Gridworld)
    return isterminal(g) ? 0f0 : -1f0
end

function isterminal(g::Gridworld)::Bool
    return g._x in g._goalxs && g._y in g._goalys
end

function γ(g::Gridworld)
    isterminal(g) ? 0f0 : g._γ
end

function observation_space(g::Gridworld)
    return g._obsspace
end

function action_space(g::Gridworld)::AbstractSpace{Int,1}
    return g._actionspace
end

function _obs(g::Gridworld{S}) where {S<:Number}
    if !g._onehot
        return [g._x, g._y]
    end
    state = zeros(S, length(g))
    state[(g._x-1)*g._rows+g._y] = one(S)

    return state
end

function Base.length(g::Gridworld)
    return g._rows * g._cols
end

function Base.show(io::IO, g::Gridworld)
    println(io, "Gridworld")
end
