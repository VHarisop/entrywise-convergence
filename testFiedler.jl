#!/usr/bin/env julia

using ArgParse
using Arpack
using CSV
using DataFrames
using LinearAlgebra
using LinearMaps
using LightGraphs
using Logging
using MatrixNetworks
using Printf
using Random
using Statistics

include("Utils.jl")
include("IOUtils.jl")

# compute the oracle gap between the first and second eigenvalues
# as well as the leading eigenvector
oracle(A) = begin
    evals, evecs, _ = eigs(A, nev=2, ritzvec=true)
    return abs(evals[1] - evals[2]), evecs[:, 1]
end

# leading vector of normalized adjacency matrix
leadVec(g) = begin
    _, d = IOUtils.adjMatrix(g, true)
    return normalize(sqrt.(d))
end

# pick loading function from IOUtils
pickFun(data) = begin
    if (data == :DBLP)
        IOUtils.genDBLPGraph()
    elseif (data == :YOUTUBE)
        IOUtils.genYoutubeGraph()
    elseif (data == :LIVEJOURNAL)
        IOUtils.genLJGraph()
    elseif (data == :GEMSEC)
        IOUtils.genGemsecGraph()
    elseif (data == :ASYS)
        IOUtils.genAsysGraph()
    elseif (data == :GOOGLE)
        IOUtils.genGoogleGraph()
    elseif (data == :ACTOR)
        IOUtils.genActorGraph()
    elseif (data == :HEP)
        IOUtils.genHepGraph()
    elseif (data == :ASTRO)
        IOUtils.genAstroGraph()
    else
        throw(ErrorException("Option $(data) not recognized!"))
    end
end


function compareCuts(data, maxExp, τ)
    g = pickFun(data); adj = adjacency_matrix(g); ttVol = sum(adj)
    A = IOUtils.adjMatrixReg(g, τ) + 1.0I; v₁ = leadVec(g); dg = degree(g)
    # A, dg = IOUtils.adjMatrix(g, true); v₁ = normalize(sqrt.(dg))
    Aop = LinearMap(X -> A * X - v₁ .* (v₁' * (A * X)), size(A)..., issymmetric=true)
    n, _  = size(A)
    v₀    = normalize(rand(n))
    gap, vTrue = oracle(Aop); μ = maximum(abs.(vTrue)) * sqrt(n)  # true incoh.
    bestCut  = sweepcut(adj, normalize((1 ./ sqrt.(dg)) .* vTrue), ttVol)
    bestCond = minimum(bestCut.conductance[:])
    @info("gap: $(gap) - μ: $(μ) - bestCond: $(bestCond)")
    v = copy(v₀); ϵCurr₂ = 1; ϵCurrI = 1; ϵCurr₂True = 1; itIdx = 0
    df = DataFrame(cIdx=(1:(n-1)))
    ind₂ = fill(Inf, maxExp)
    indI = fill(Inf, maxExp)
    ind₂True = fill(Inf, maxExp);
    pend₂ = pendI = pendT = true;
    while (pendI || pend₂ || pendT)
        itIdx += 1
        v[:]   = normalize(Aop * v)
        λ      = v' * (Aop * v)
        res    = Aop * v - λ .* v
        ϵ₂, ϵI = norm(res), maximum(abs.(res))
        ϵ₂     = ϵ₂ / gap
        ϵFac   = ϵ₂
        # residuals
        vv2E  = res .- v * (v'res)
        nrm0  = maximum(abs.(v))
        nrm1  = maximum(abs.(vv2E))
        ϵComp = min(ϵFac, 8 * nrm0 * (ϵFac^2) + 2 * (nrm1 / gap) * (1 + ϵFac))
        @debug("it: $(itIdx) - ϵ₂: $(ϵ₂) - ϵInf: $(ϵComp)")
        # more positive signs to make process deterministic
        nPos  = sum(v .>= 0); nNeg = sum(v .< 0)
        vFied = (nPos > nNeg) ? (1 ./ sqrt.(dg)) .* copy(v) : (1 ./ sqrt.(dg)) .* copy(-v)
        # get approx fiedler vec
        if pendI && (ϵComp ≤ 10.0^(-ϵCurrI))
            cut = sweepcut(adj, normalize(vFied), ttVol)
            df[Symbol("cI_$(ϵCurrI)")] = cut.conductance[:]
            d₂ = min(norm(v - vTrue), norm(v + vTrue))
            dI = min(norm(v - vTrue, Inf), norm(v + vTrue, Inf))
            @info("[ϵI = 1e-$(ϵCurrI)] min. ϕ(S): $(minimum(cut.conductance[:]))")
            @info("[dI = $(dI), d₂ = $(d₂)")
            indI[ϵCurrI] = itIdx
            if ϵCurrI == maxExp
                pendI = false
            end
            ϵCurrI += 1
        end
        if pend₂ && (ϵ₂ ≤ 10.0^(-ϵCurr₂) * sqrt(n))
            cut = sweepcut(adj, vFied, ttVol)
            df[Symbol("c2_$(ϵCurr₂)")] = cut.conductance[:]
            @info("[ϵ₂ = 1e-$(ϵCurr₂) × √n] min. ϕ(S): $(minimum(cut.conductance[:]))")
            # increase to next level, so that we don't trigger twice
            ind₂[ϵCurr₂] = itIdx
            if ϵCurr₂ == maxExp
                pend₂ = false
            end
            ϵCurr₂ += 1
        end
        if pendT && (ϵ₂ ≤ 10.0^(-ϵCurr₂True))
            ind₂True[ϵCurr₂True] = min(ind₂True[ϵCurr₂True], itIdx)
            # store cut conductance too
            cut = sweepcut(adj, vFied, ttVol)
            df[Symbol("c2T_$(ϵCurr₂True)")] = cut.conductance[:]
            if ϵCurr₂True == maxExp
                pendT = false
            end
            ϵCurr₂True += 1
        end
    end
    @info("indsI: $(indI) - inds₂: $(ind₂) - inds₂T: $(ind₂True)")
    return df, indI ./ ind₂, indI ./ ind₂True
end

s = ArgParseSettings(
    description="Compare the performance of the two stopping criteria for " *
                "generating the sweep cut approximation of a community plot, " *
                "using the Fiedler vector for biclustering.")
datasets = ["dblp", "livejournal", "gemsec", "hep", "astro"]
@add_arg_table s begin
    "--dataset"
        arg_type     = String
        range_tester = (x -> lowercase(x) in datasets)
        help         = "dataset; must be one of $(join(datasets, ", "))"
    "--max_exp"
        help     = "The largest exponent in the accuracy levels 10^(-i)"
        arg_type = Int
        default  = 6
    "--tau"
        help     = "The regularization parameter for the normalized adj. matrix"
        arg_type = Float64
        default  = 0.0
    "--seed"
        help     = "The random seed to set"
        arg_type = Int
        default  = 999
end
parsed  = parse_args(s); Random.seed!(parsed["seed"])
data    = Symbol(uppercase(parsed["dataset"]))
τ, mExp = parsed["tau"], parsed["max_exp"]
df, ratioNaive, ratioTrue = compareCuts(data, mExp, τ)
CSV.write("normCut_$(data)_$(mExp).csv", df)
CSV.write("normCut_ratios_$(data)_$(mExp).csv", DataFrame(k=1:mExp, ratio=ratioTrue))
