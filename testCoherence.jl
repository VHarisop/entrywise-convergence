#!/usr/bin/env julia

using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Logging
using Random
using Printf

include("Utils.jl")


function compareCoherence(n, r, ϵ, ρ, cCompl, num_logged)
    V₁, μ₁ = Utils.genCoherent(n, r)
    V₂, μ₂ = Utils.genIncoherent(n, r)
    V₀     = Matrix(qr(randn(n, r)).Q)  # init. guess
    Vp, _  = cCompl ? Utils.genCoherent(n, n - r) : Utils.genIncoherent(n, n - r)
	Vp₁    = Matrix(qr(Vp - V₁ * (V₁'Vp)).Q)  # subspace 1
	Vp₂    = Matrix(qr(Vp - V₂ * (V₂'Vp)).Q)  # subspace 2
    Λ      = ρ .^ (0:n - 1)  # eigvals
    gap    = Λ[r] - Λ[r+1]
    @info("eigengap - $(gap)")
    # generate matrices
    A₁  = V₁ * Diagonal(Λ[1:r]) * V₁' + Vp₁ * Diagonal(Λ[(r+1):end]) * Vp₁'
    A₂  = V₂ * Diagonal(Λ[1:r]) * V₂' + Vp₂ * Diagonal(Λ[(r+1):end]) * Vp₂'
    # check subspace iteration for both versions
    _, d₂C, dIC, r₂C, rIC, (idxNaiveC, idxCompC) =
        Utils.subspaceIteration(A₁, V₀, n, ϵ=ϵ, use_naive=true, use_theo=false,
								μ=μ₁, gap=gap, num_logged=num_logged, Vtrue=V₁)
    @info("[COHERENT] stopping - $(idxCompC) vs. $(idxNaiveC) (naive)")
    _, d₂I, dII, r₂I, rII, (idxNaiveI, idxCompI) =
        Utils.subspaceIteration(A₂, V₀, n, ϵ=ϵ, use_naive=true, use_theo=false,
								μ=μ₂, gap=gap, num_logged=num_logged, Vtrue=V₂)
    @info("[INCOHERENT] stopping - $(idxCompI) vs. $(idxNaiveI) (naive)")
    # theor. conv.
    d₁ = opnorm(V₀'Vp₁); tanθ₁ = d₁ / sqrt(1 - d₁^2)
    d₂ = opnorm(V₀'Vp₂); tanθ₂ = d₂ / sqrt(1 - d₂^2)
    step = n ÷ num_logged
    rsC  = (1:length(d₂C)) .* step
    rsI  = (1:length(d₂I)) .* step
    # create a data frame for each case
    dfC = DataFrame(k=(1:length(d₂C)) .* step, d2=d₂C, dI=dIC, r2=r₂C, rI=rIC,
                    dTheo=(ρ.^(rsC)) .* tanθ₁)
    dfI = DataFrame(k=(1:length(d₂I)) .* step, d2=d₂I, dI=dII, r2=r₂I, rI=rII,
                    dTheo=(ρ.^(rsI)) .* tanθ₂)
    dig = -log10(ϵ)
    CSV.write("synthetic_$(n)_$(r)-cc-accuracy_$(@sprintf("%d", dig)).csv", dfC)
    CSV.write("synthetic_$(n)_$(r)-ii-accuracy_$(@sprintf("%d", dig)).csv", dfI)
    return dfC, dfI
end

s = ArgParseSettings(
    description="Compare the performance of subspace iteration for coherent " *
                "and incoherent matrices on synthetic examples.")
@add_arg_table s begin
    "--n"
        help     = "Problem dimension"
        arg_type = Int
        default  = 1000
    "--r"
        help     = "Target subspace dimension"
        arg_type = Int
        default  = 25
    "--eps"
        help     = "Target accuracy"
        arg_type = Float64
        default  = 1e-4
    "--rho"
        help     = "Target ratio between successive eigenvalues"
        arg_type = Float64
        default  = 0.95
    "--seed"
        help     = "The random seed to set"
        arg_type = Int
        default  = 999
    "--num_logged"
        help     = "The number of logged data points"
        arg_type = Int
        default  = 50
    "--coherent_compl"
        help     = "Set to true to make complementary subspace coherent"
        action   = :store_true
end
parsed     = parse_args(s); Random.seed!(parsed["seed"])
n, r, ϵ, ρ = parsed["n"], parsed["r"], parsed["eps"], parsed["rho"]
cCompl, nLogged = parsed["coherent_compl"], parsed["num_logged"]
# get data frames from running experiment
dfC, dfI = compareCoherence(n, r, ϵ, ρ, cCompl, nLogged)
