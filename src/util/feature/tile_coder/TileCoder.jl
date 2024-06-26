struct TileCoder{T<:AbstractTiling{<:Number,<:Number}} <: AbstractFeatureCreator
    _tilings::Vector{T}
    _bias::Bool
    _sum_to_one::Bool

    function TileCoder(
        tilings::Vector{T}, bias::Bool, sum_to_one::Bool,
    ) where {T<:AbstractTiling{<:Number,<:Number}}
        return new{T}(tilings, bias, sum_to_one)
    end
end

function GridTileCoder(
    OUT::Type,
    mindims::Vector{IN},
    maxdims::Vector{IN},
    bins::AbstractMatrix{<:Integer},
    seed::Int;
    bias::Bool=true,
    max_offset::Float32=0.67f0,
    sum_to_one::Bool=true,
) where {IN<:Number}
    # Create each tiling
    num = size(bins, 2)

    tilings::Vector{GridTiling{IN,OUT}} = []
    rng = Xoshiro(convert(UInt, seed))

    for i = 1:num
        # Create the tiling. Each tiling will be given a different seed, which determines
        # the offset from the origin.
        next_seed = convert(UInt, seed + abs(rand(rng, Int)))
        tiling = GridTiling{OUT}(
            mindims, maxdims, bins[:, i], next_seed; max_offset=max_offset,
        )
        append!(tilings, [tiling])
    end

    return TileCoder(tilings, bias, sum_to_one)
end

function (t::TileCoder)(state; use_onehot=true)
    return use_onehot ? onehot(t, state) : index(t, state)
end
nonzero(t::TileCoder)::Int = t._bias + length(t._tilings)
Base.getindex(t::TileCoder, i) = index(t, i)
Base.size(t::TileCoder) = size(t._tilings)
include_bias(t::TileCoder)::Bool = t._bias

function features(t::TileCoder)::Int
    total = 0
    for tiling in t._tilings
        total += features(tiling)
    end
    return total + t._bias
end

"""
    _features_before(t::TileCoder, i::Integer)::Int

Return the number of features generated by tilings before tiling `i` in
the `TileCoder`.
"""
function _features_before(t::TileCoder, i::Integer)::Int
    if i > length(t._tilings)
        return features(t)
    end

    total = 0
    for tiling = 1:i-1
        # Calculate the number of features before tiling i
        total += features(t._tilings[tiling])
    end
    return total + t._bias
end

function index(
    t::TileCoder{<:AbstractTiling{T}}, v::Vector{T}, tiling::Integer,
)::Int where {T<:Number}
    if !(0 < tiling <= length(t._tilings))
        error("no such tiing $tiling ∉ [1, $(length(t._tilings))]")
    end
    offset = _features_before(t, tiling)
    ind = index(t._tilings[tiling], v)
    return ind .+ offset
end

function index(
    t::TileCoder{<:AbstractTiling{T}}, v::Vector{T},
)::Vector{Int} where {T<:Number}
    indices::Vector{Int} = []
    if t._bias
        push!(indices, 1)
    end

    for i = 1:length(t._tilings)
        push!(indices, index(t, v, i))
    end
    return indices
end

function index(
    t::TileCoder{<:AbstractTiling{T}}, v::AbstractArray{T,N}, tiling::Integer,
)::Array{Int,N - 1} where {T<:Number,N}
    if !(0 < tiling <= length(t._tilings))
        error("no such tiing $tiling ∉ [1, $(length(t._tilings))]")
    end
    offset = _features_before(t, tiling)
    indices = index(t._tilings[tiling], v)
    return indices .+ offset
end

function index(t::TileCoder{<:AbstractTiling{T}}, v::AbstractArray{T,N}) where {T<:Number,N}
    indices = []

    for i = 1:length(t._tilings)
        # Get the index of the batch into the tilings of the tile coder
        idx = index(t, v, i)
        push!(indices, idx)
    end

    # Add bias unit if applicable
    if t._bias
        bias = one.(indices[end])
        push!(indices, bias)
    end

    return stack(indices; dims=1)
end

function onehot(
    t::TileCoder{<:AbstractTiling{IN,OUT}}, v::AbstractVector{IN}, tiling::Int,
)::Vector{OUT} where {IN<:Number,OUT<:Number}
    if !(0 < tiling <= length(t._tilings))
        error("no such tiing $tiling ∉ [1, $(length(t._tilings))]")
    end

    onehot = zeros(OUT, features(t))
    elem = t._sum_to_one ? oftype(onehot[1], inv(nonzero(t))) : one(OUT)
    setindex!(onehot, elem, index(t, v, tiling))

    if t._bias
        onehot[1] = elem
    end
    return onehot
end

function onehot(
    t::TileCoder{<:AbstractTiling{IN,OUT}}, v::AbstractVector{IN},
)::Vector{OUT} where {IN<:Number,OUT<:Number}
    onehot = zeros(OUT, features(t))
    indices = index(t, v)
    onehot[indices] .= t._sum_to_one ? oftype(onehot[1], inv(nonzero(t))) : one(OUT)
    return onehot
end

function onehot(
    t::TileCoder{<:AbstractTiling{IN,OUT}}, v::AbstractArray{IN,N}, tiling::Integer,
)::Array{OUT,N} where {IN<:Number,OUT<:Number,N}
    if !(0 < tiling <= length(t._tilings))
        error("no such tiing $tiling ∉ [1, $(length(t._tilings))]")
    end

    indices = index(t, v, tiling)
    s = size(indices)
    indices = reshape(indices, prod(s))

    onehot = zeros(OUT, size(indices)[1], features(t))
    elem = t._sum_to_one ? oftype(onehot[1], inv(nonzero(t))) : one(OUT)
    onehot[[CartesianIndex(i, indices[i]) for i = 1:size(indices)[1]]] .= elem

    if t._bias
        onehot[1, :] .= elem
    end

    reshape(onehot, s..., features(t))
end

function onehot(
    t::TileCoder{<:AbstractTiling{IN,OUT}}, v::AbstractArray{IN,N},
)::Array{OUT,N} where {IN<:Number,OUT<:Number,N}
    indices = index(t, v)
    s = size(indices)
    indices = reshape(indices, prod(s[1:end-1]), s[end]) # Reshape to matrix

    onehot = zeros(OUT, features(t), size(indices, 2))
    elem = t._sum_to_one ? oftype(onehot[1], inv(nonzero(t))) : one(OUT)
    for col = 1:size(indices, 2)
        onehot[indices[:, col], col] .= elem
    end

    reshape(onehot, features(t), s[begin+1:end]...)
end
