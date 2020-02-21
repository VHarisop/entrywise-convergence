#!/usr/bin/env julia

using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Logging
using Random
using Printf

include("Utils.jl")


# classical rate
function standardRate(ρ, d₀, its)
    return ρ.^(its) .* d₀ / sqrt(1 - d₀^2)
end

# "ideal" rate
function standard2InfRate(ρ, d₀, dInf₀, its)
    return ρ.^(its) .* (dInf₀ / sqrt(1 - d₀^2))
end

# first derived rate
function thInfRate1(ρ₁, ρ₂, n, r, μ₁, μ₂, d₀, dInf₀, its)
    t₀ = d₀ / sqrt(1 - d₀^2)
    return ρ₁.^(its) .* (t₀ * μ₁ * sqrt(r / n) + μ₂ * dInf₀ / sqrt(1 - d₀^2)) +
        ρ₂.^(its) .* t₀
end

# second derived rate
function thInfRate2(ρ, n, r, μ, C, d₀, dInf₀, its)
    c₀ = sqrt(1 - d₀^2)
    return ρ.^(its) * (μ * sqrt(r / n) * (d₀ / c₀) + C * dInf₀ / c₀)
end


"""
compareConvergence(n, r, ϵ, ρ, cCompl, rDecay)

Compares the ℓ2→∞ distance between the leading `r`-dimensional subspace and the
subspace iteration estimates for a synthetically generated `n x n` matrix with
the theoretical bound involving the ∞-∞ operator norm of the `r+1`-st
eigenvector. The algorithm is run until reaching residual `ϵ`, while the matrix
generated has ratio of successive eigenvalues equal to `ρ`.
Return a dataframe containing the following columns:
- `d2`: the spectral distance between iterates
- `dInf`: the ℓ2→∞ distance between iterates
- `r2`: the theoretical convergence rate for the spectral distance
- `rInf`: the theoretical convergence rate for the ℓ2→∞ distance
"""
function compareCoherence(n, r, ϵ, ρ, cCompl, rDecay)
    # set num_logged
    num_logged = (n ÷ 5)
    V₁, μ₁ = Utils.genIncoherent(n, r)
    V₂, _  = cCompl ? Utils.genCoherent(n, n - r) : Utils.genIncoherent(n, n - r)
	V₂     = Matrix(qr(V₂ - V₁ * (V₁'V₂)).Q)  # orthogonalize subspace 2
    V₀     = Matrix(qr(randn(n, r)).Q)  # init. guess
    Λ      = ρ .^ (0:n - 1)  # eigvals
	if rDecay  # rapid decay after (r+1)-st eigenvalue
		@info("Adjusting eigenvalues for rapid decay")
		Λ[r + 2] = 0.05 * Λ[r]
		Λ[(r+3):end] = Λ[r+2] .* (ρ.^(1:(n - (r + 2))))
	end
    gap    = Λ[r] - Λ[r+1]
    # generate matrix
    A   = V₁ * Diagonal(Λ[1:r]) * V₁' + V₂ * Diagonal(Λ[(r+1):end]) * V₂'
    # get infnorm of v_{r+1} and ratio between ∞-operator norms
    C₁ = opnorm(V₂[:, 1] .* V₂[:, 1]', Inf)
    C₂ = opnorm(V₂ * Diagonal(Λ[(r+1):end]) * V₂', Inf) / (Λ[r+1] * opnorm(V₂ * V₂', Inf))
	@info("eigengap - $(gap) - C₁: $(C₁) - C₂: $(C₂)")
    _, d₂, dI, _, _, _ = Utils.subspaceIteration(A, V₀, n, ϵ=ϵ, use_naive=true,
                                                 use_theo=false, μ=μ₁, gap=gap,
                                                 num_logged=num_logged, Vtrue=V₁)
    λ₁, λ₂, λ₃ = Λ[[r, r+1, r+2]]  # eigs controlling convergence
    ρ₁, ρ₂ = ρ, (λ₃ / λ₁)
	@info("ρ₁: $(ρ₁) - ρ₂: $(ρ₂)")
    # theor. conv.
    d₀   = opnorm(V₀'V₂); dInf₀ = Utils.ℓ₂infProxy(V₁, V₀)
    step = n ÷ num_logged
    rsC  = (1:length(d₂)) .* step
    rsI  = (1:length(dI)) .* step
    # get rates
    rateClassic2 = standardRate(ρ₁, d₀, rsC)
    rateClassicI = standard2InfRate(ρ₁, d₀, dInf₀, rsC)
    rateTheory1  = thInfRate1(ρ₁, ρ₂, n, r, μ₁, C₁, d₀, dInf₀, rsC)
    rateTheory2  = thInfRate2(ρ₁, n, r, μ₁, C₂, d₀, dInf₀, rsC)
    # create a data frame holding the results
    df  = DataFrame(k=((1:length(d₂)) .* step), rateC=rateClassic2, rateI=rateClassicI,
                    rateT1=rateTheory1, rateT2=rateTheory2, dist=dI)
    dig = -log10(ϵ)
	CSV.write("conv_synth_$(n)-$(r)-$(cCompl)-acc_$(@sprintf("%d", dig))-rho_$(@sprintf("%.2f", ρ))-rDecay_$(rDecay).csv", df)
    return df
end


s = ArgParseSettings(
    description="Compare the performance of subspace iteration with the " *
                "theoretically prescribed convergence rate for synthetic " *
                "coherent and incoherent examples.")
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
        help     = "Target between successive eigenvalues"
        arg_type = Float64
        default  = 0.95
    "--seed"
        help     = "The random seed to set"
        arg_type = Int
        default  = 999
    "--coherent_compl"
        help     = "Set to true to make complementary subspace coherent"
        action   = :store_true
	"--rdecay"
	    help     = "Set to have rapid decay after (r+1)-st eigenvalue"
		action   = :store_true
end
parsed     = parse_args(s); Random.seed!(parsed["seed"])
n, r, ϵ, ρ = parsed["n"], parsed["r"], parsed["eps"], parsed["rho"]
cCompl     = parsed["coherent_compl"]
rDecay     = parsed["rdecay"]
# get data frames from running experiment
df = compareCoherence(n, r, ϵ, ρ, cCompl, rDecay)
