"""
    εGreedyPolicy <: AbstractValuePolicy

εGreedyPolicy implements an ε-greedy policy
"""
struct εGreedyPolicy <: AbstractValuePolicy
    _ε::Float32

    function εGreedyPolicy(ε)
        @assert zero(ε) <= ε <= one(ε) "ε ∉ [0, 1]"
        new(ε)
    end
end

function sample(p::εGreedyPolicy, rng::AbstractRNG, qs::Matrix)
    return dropdims(mapslices(x -> sample(p, rng, x), qs; dims=1); dims=1)
end

function sample(p::εGreedyPolicy, rng::AbstractRNG, qs::Vector; num_samples=1)
    u = rand(typeof(p._ε))
    if u < p._ε
        action = rand(rng, [i for i in 1:size(qs, 1)])
        return action
    else
        return argmax(qs)
    end
end

logprob(p::εGreedyPolicy, qs) = log.(prob(p, qs))

function prob(p::εGreedyPolicy, qs::Matrix)
    return mapslices(x -> prob(p, x), qs; dims=1)
end

function prob(p::εGreedyPolicy, qs::Vector)
    ε = p._ε
    probs = zeros(length(qs))
    probs[:] .= ε / length(qs)

    idx = findall(==(maximum(qs)), qs)
    probs[idx] .+= (1 - ε) / size(idx)[1]

    return probs
end

function mode(p::εGreedyPolicy, qs::Matrix)
    return dropdims(mapslices(x -> mode(p, x), qs; dims=1); dims=1)
end

function mode(p::εGreedyPolicy, qs::Vector)
    return argmax(qs)
end
