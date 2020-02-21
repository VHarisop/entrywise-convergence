#!/usr/bin/env julia

using ArgParse
using Arpack
using CSV
using DataFrames
using LinearAlgebra
using Logging
using Printf
using Random
using Statistics

include("Utils.jl")
include("IOUtils.jl")


# compute the oracle gap between the first and second eigenvalues
# as well as the leading eigenvector
oracle(A) = begin
    evals, evecs, _ = eigs(A, nev=2, which=:LM, ritzvec=true)
    # make sure we get all-positive vec.
    if minimum(evecs[:, 1]) < 0
        evecs[:, 1] = -evecs[:, 1]
    end
    return abs(evals[1] - evals[2]), evecs[:, 1]
end

# pick loading function from IOUtils
pickFun(data) = begin
    if (data == :DBLP)
        IOUtils.genDBLPMatrix
    elseif (data == :LIVEJOURNAL)
        IOUtils.genLJMatrix
    elseif (data == :GEMSEC)
        IOUtils.genGemsecMatrix
    elseif (data == :HEP)
        IOUtils.genHepMatrix
    elseif (data == :ASTRO)
        IOUtils.genAstroMatrix
    else
        throw(ErrorException("Option $(data) not recognized!"))
    end
end

function eigCentralityAll(data, maxExp, normalized, useTheo)
    adjMat, dG = pickFun(data)(normalized)
    # compute leading eigenvector - start from all ones vector
    n, _  = size(adjMat)
    v₁, μ = ones(n) ./ sqrt(n), 1.0
    gap, vTrue = oracle(adjMat)
    @info("Oracle gap - $(gap)")
    # compute smallest distance between ranked nodes' centrality
    nComp = trunc(Int, sqrt(n) / 5)
    vRanked = sort(vTrue, rev=true)[1:nComp]
    vecDiff = abs.(diff(vRanked))
    @info("diffs - mean: $(mean(vecDiff)), median: $(median(vecDiff)), min: $(minimum(vecDiff))")
    ϵ = parse(Float64, "1e-$(maxExp)")
    # compute usual metrics and ranking distance
    v, d₂, dI, res₂, resI, (idxNaive, idxComp), τs =
        Utils.powerMethod(adjMat, v₁, ϵ=ϵ, use_naive=true, μ=μ, gap=gap,
                          use_theo=false, comp_kendall=true,
                          nComp=nComp, vTrue=vTrue)
    # compute ratios
    maxIt = max(idxNaive, idxComp)
    replFun = x -> (x == nothing) ? maxIt : x
    indsNaive = replace(replFun,
                        findfirst.([res₂ .< parse(Float64, "1e-$(i)") for i in 1:maxExp]))
    indsComp  = replace(replFun,
                        findfirst.([resI .< parse(Float64, "1e-$(i)") for i in 1:maxExp]))
    ratios = indsComp ./ indsNaive
    # in the normalized case, use true ratios due to underestimation
    trueIndsNaive = replace(replFun,
                            findfirst.([d₂ .< parse(Float64, "1e-$(i)") for i in 1:maxExp]))
    trueIndsCompo = replace(replFun,
                            findfirst.([dI .< parse(Float64, "1e-$(i)") for i in 1:maxExp]))
    trueRatios = trueIndsCompo ./ trueIndsNaive
    @info("Ratios - $(ratios)")
    @info("True Ratios - $(trueRatios)")
    # save dataframe
    df       = DataFrame(k=1:length(d₂), d2=d₂, dI=dI,
                         r2=res₂, rI=resI, taus=accumulate(min, abs.(τs)))
    if normalized
        dfStop   = DataFrame(exp=(1:maxExp), ratio=trueRatios)
        CSV.write("$(data)-nrm_$(normalized)-theo_$(useTheo)-maxExp_$(maxExp)-ratios.csv", dfStop)
    else
        dfStop   = DataFrame(exp=(1:maxExp), ratio=ratios)
        CSV.write("$(data)-nrm_$(normalized)-theo_$(useTheo)-maxExp_$(maxExp)-ratios.csv", dfStop)
    end
    CSV.write("$(data)-nrm_$(normalized)-theo_$(useTheo)-maxExp_$(maxExp).csv", df)
end


s = ArgParseSettings(
    description="Compare the performance of the two stopping criteria for the " *
                "computation of eigenvector centrality in two large networks.")
datasets = ["dblp", "livejournal", "gemsec", "hep", "astro"]
@add_arg_table s begin
    "--dataset"
        arg_type     = String
        range_tester = (x -> lowercase(x) in datasets)
        help         = "dataset; must be one of $(join(datasets, ", "))"
    "--normalized"
        help   = "Set to use the normalized adjacency matrix"
        action = :store_true
    "--max_exp"
        help     = "The largest exponent in the accuracy levels 10^(-i)"
        arg_type = Int
        default  = 6
    "--use_theo"
        help     = "Set to use provable version of stopping crit."
        action   = :store_true
    "--seed"
        help     = "The random seed to set"
        arg_type = Int
        default  = 999
end
parsed  = parse_args(s); Random.seed!(parsed["seed"])
data = Symbol(uppercase(parsed["dataset"]))
nrm  = parsed["normalized"]
mExp = parsed["max_exp"]
theo = parsed["use_theo"]
# compute centralities and other metrics
df   = eigCentralityAll(data, mExp, nrm, theo)
