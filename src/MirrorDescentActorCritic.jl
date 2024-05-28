module MirrorDescentActorCritic

import ChoosyDataLoggers
import ChoosyDataLoggers: @data
function __init__()
    ChoosyDataLoggers.@register
end

import Reproduce: @param_from # For construction utils
using ComponentArrays
using StaticArrays
using ChainRulesCore
using AbstractTrees
using ExtendedDistributions
using DistributionsAD
using LinearAlgebra
using Lux
using CUDA
using LuxCUDA
using Random
using Roots
using SparseArrays
using StatsBase
using Tullio
using Zygote
using Adapt
using Optimisers
using Lazy


# Feature constructors
include("util/feature/feature.jl")

# Function Approximators
export Linear, Tabular
include("util/approximator/linear.jl")
include("util/approximator/tabular.jl")
include("util/approximator/lux.jl")
include("util/approximator/util.jl")

# Optimisers
export BackTracking
include("optimisers/backtracking.jl")

# GPU Utilities
include("util/gpu.jl")

include("env/environment.jl")
include("policy/policy.jl")
include("value_function/value_function.jl")
include("update/update.jl")

# Experience Replay
include("util/buffer/buffer.jl")

export
    OnlineQAgent,
    PGFunctionalBaselineAgent,
    PGAgent,
    BatchQAgent,
    ValueAgent,
    UpdateRatio,
    AbstractAgentWrapper,
    AbstractAgentActionWrapper,
    RandomFirstActionAgent

include("agent/abstract_agent.jl")
include("agent/agent_wrapper.jl")
include("agent/update_ratio.jl")
include("agent/online_qagent.jl")
include("agent/pg_agent.jl")
include("agent/pg_functional_baseline_agent.jl")
include("agent/batch_qagent.jl")
include("agent/value_agent.jl")

include("util/episode.jl")

# General construction utilities: updates, optimisers, buffers, policies, value functions
include("util/construct.jl")

# Experiment Utils
include("util/exp.jl")
include("util/exp/brax_experiment.jl")
include("util/exp/gymnasium_experiment.jl")
include("util/exp/default_experiment.jl")

end # module MirrorDescentActorCritic
