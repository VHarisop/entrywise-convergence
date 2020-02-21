#!/usr/bin/env julia

using ArgParse
using Arpack
using CSV
using Clustering
using DataFrames
using Distributions
using LinearAlgebra
using LightGraphs
using Logging
using Printf
using Random
using Statistics

include("Utils.jl")
include("IOUtils.jl")

logger = SimpleLogger(stdout, Logging.Debug)


"""
    multiwayCut(A, k, inds)

Compute the value of the multiway cut induced by the cluster assignment
`inds` using the adjacency matrix `A` on k clusters in total.
"""
function multiwayCut(A, k, inds)
    cutVal = 0.0
    @inbounds for idx in 1:k
        idPick = (inds .== idx)
        eCross = sum(A[idPick, .!idPick])
        idxVol = sum(A[idPick, :])
        cutVal += (eCross / idxVol)
    end
    return cutVal
end

"""
    qrClustering(U; soft=true) -> P

Compute a 'soft' clustering assignment using the column-pivoted QR method given
the `n x r` eigenvector matrix `U`. A hard assignment can then be obtained via
the rowwise `argmax` of the resulting object. If `soft == false`, return the
resulting assignment as a vector.
Return:
- `P`: a clustering assignment, in the form of a vector of indices if `soft == false`.
"""
function qrClustering(U)
    _, r = size(U)
    _, _, p = qr(U', Val(true))  # get pivot vector
    piv = p[1:r]
    svdObj = svd(U[piv, :]')
    return vec(mapslices(argmax, abs.(U * (svdObj.U * svdObj.Vt)), dims=2))
end


# compute the oracle gap between the first and second eigenvalues
# as well as the leading eigenvector
oracle(A, k) = begin
    evals, evecs, _ = eigs(A, nev=k+1, which=:LM, ritzvec=true)
    return evals, abs(evals[k] - evals[k+1]), evecs[:, 1:k]
end


# count elements of each cluster
function countElts(p)
    return [count(x -> x == i, p) for i in unique(p)]
end


"""
    genSBM(n, nComms; p=(10 * log(n) / n), q=(log(n) / n), balanced=false) -> (g, perCom)

Generate a sample from the Stochastic Block Model on `n` nodes with `nComms`
communities, with intra-community edge probability `p` and inter-community edge
probability `q`.

Returns:
- `g`: the undirected graph corresponding to the SBM
- `perCom`: a vector of length `nComm` containing the number of nodes in each
   community
"""
function genSBM(n, nComms; p=(10 * log(n) / n), q=(log(n) / n), balanced=false)
    if balanced
        perCom = repeat([n ÷ nComms], nComms)
    else
        perCom = rand(Multinomial(n, nComms))
    end
    indSums = cumsum(perCom)
    g       = Graph(n)
    for i in 1:n
        ind_i = searchsortedfirst(indSums, i)
        for j in (i+1):n
            ind_j = searchsortedfirst(indSums, j)
            (ind_i == ind_j) && (rand() ≤ p) && add_edge!(g, i, j)
            (ind_i ≠  ind_j) && (rand() ≤ q) && add_edge!(g, i, j)
        end
    end
    # return the graph and the number of nodes per community (for labels)
    return g, perCom
end

function genSBM(perCom::AbstractArray{<:Real, 1}, p::Float64, q::Float64)
	n = sum(perCom); indSums = cumsum(perCom)
	g = Graph(n)
    for i in 1:n
        ind_i = searchsortedfirst(indSums, i)
        for j in (i+1):n
            ind_j = searchsortedfirst(indSums, j)
            (ind_i == ind_j) && (rand() ≤ p) && add_edge!(g, i, j)
            (ind_i ≠  ind_j) && (rand() ≤ q) && add_edge!(g, i, j)
        end
    end
	return g
end

# pick loading function from IOUtils
pickFun(data) = begin
    if (data == :DBLP)
        IOUtils.genDBLPGraph()
    elseif (data == :LIVEJOURNAL)
        IOUtils.genLJGraph()
    elseif (data == :GEMSEC)
        IOUtils.genGemsecGraph()
    elseif (data == :HEP)
        IOUtils.genHepGraph()
    elseif (data == :ASTRO)
        IOUtils.genAstroGraph()
    else
        throw(ErrorException("Option $(data) not recognized!"))
    end
end

function evalPerf(data, maxExp, k, τ)
    g = pickFun(data); adj = adjacency_matrix(g)
    A = IOUtils.adjMatrixReg(g, τ) + 1.0I
    n, _   = size(A)
    V₀, _  = Utils.genIncoherent(n, k); V = copy(V₀)
    Λ, gap, vTrue = oracle(A, k)
    μ = maximum(mapslices(norm, vTrue, dims=2)) * sqrt(n / k)
    p = qrClustering(vTrue)
    cVal = multiwayCut(adj, k, p)
    cValOracle = cVal
    @info("cVal: $(cVal)")
    @info("elts: $(countElts(p))")
    @info("Λ: $(Λ) - gap: $(gap) - μ: $(μ)")
    pendI  = pend₂ = true
    ϵCurrI = ϵCurr₂ = 1
    itIdx  = 0
    cVal   = Inf
    cVals  = fill(0.0, maxExp)
    indI = fill(Inf, maxExp)
    ind₂ = fill(Inf, maxExp)
    disI = fill(0.0, maxExp)
    dis₂ = fill(0.0, maxExp)
    while (pend₂ || pendI)
        itIdx += 1
        V = Utils._qrDirect!(Matrix(A * V))
        ϵ₂, ϵComp = Utils.residual(A, V, μ=μ, gap=gap)
        ϵComp = min(ϵ₂, ϵComp)
        @debug("it: $(itIdx) - ϵ₂: $(ϵ₂) - ϵInf: $(ϵComp)")
        if pendI && (ϵComp ≤ 2.0^(-ϵCurrI))
            # compute multiway cut metric
            p = qrClustering(V)
            cVal = min(cVal, multiwayCut(adj, k, p))
            cVals[ϵCurrI] = cVal
            # compute subspace distances
            d₂   = opnorm(vTrue - V * (V'vTrue))
            dI   = Utils.ℓ₂infProxy(V, vTrue)
            disI[ϵCurrI] = dI
            dis₂[ϵCurrI] = d₂
            # report stuff
            @info("[ϵI = 2^$(-ϵCurrI)] cVal: $(cVal) - [dI: $(dI)] - [d₂: $(d₂)]")
            # increase to next level, so that we don't trigger twice
            indI[ϵCurrI] = itIdx
            if ϵCurrI == maxExp
                pendI = false
            end
            ϵCurrI += 1
        end
        if pend₂ && (ϵ₂ ≤ 2.0^(-ϵCurr₂))
            ind₂[ϵCurr₂] = min(itIdx)
            if ϵCurr₂ == maxExp
                pend₂ = false
            end
            ϵCurr₂ += 1
        end
    end
    return DataFrame(i=1:maxExp, cvals=cVals, disI=disI, dis2=dis₂, ratio=indI ./ ind₂,
                     cvalsOracle=fill(cValOracle, maxExp))
end

s = ArgParseSettings(
    description="Examine the efficiency of the proposed stopping criterion " *
    "for clustering real-world graphs.")
datasets = ["dblp", "livejournal", "gemsec", "hep", "astro"]
@add_arg_table s begin
    "--dataset"
        arg_type     = String
        range_tester = (x -> lowercase(x) in datasets)
        help         = "dataset; must be one of $(join(datasets, ", "))"
    "--max_exp"
        help     = "The max. index of the residual levels, 2^(-max_exp)"
        arg_type = Int
        default  = 10
    "--k"
        help     = "The number of clusters to use"
        arg_type = Int
        default  = 20
    "--tau"
        help     = "The regularization parameter to use, if any"
        arg_type = Float64
        default  = 5.0
    "--seed"
        help     = "The random seed to set"
        arg_type = Int
        default  = 999
end
parsed  = parse_args(s); Random.seed!(parsed["seed"])
data, maxExp = parsed["dataset"], parsed["max_exp"]
k, τ = parsed["k"], parsed["tau"]
df = evalPerf(Symbol(uppercase(data)), maxExp, k, τ)
CSV.write("clustering_$(data)_$(k)-$(@sprintf("%.1f", τ)).csv", df)
