#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Dates
using JLD2
using Comonicon
using Statistics

function Statistics.middle(m1::Millisecond, m2::Millisecond)
    return Millisecond((m1.value + m2.value) ÷ 2)
end

"""
get runtime statistics

# Args

- `dir`: directory which holds all the files to search through
"""
@main function time_statistics(dir)
    dirs = readdir(dir)
    dirs = filter(x -> occursin("RP_", x), dirs)
    dirs = map(d -> joinpath(dir, d, "results.jld2"), dirs)
    times = []
    for d in dirs
        @load d runtime
        push!(times, only(runtime))
    end

    max_time = canonicalize(maximum(times))
    min_time = canonicalize(minimum(times))
    mean_time = canonicalize(sum(times) ÷ length(times))
    median_time = canonicalize(median(times))

    println()
    println("Min time:\t", min_time)
    println("Max time:\t", max_time)
    println("Median time:\t", median_time)
    println("Mean time:\t", mean_time)
end
