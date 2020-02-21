#!/usr/bin/env julia

module Utils

    using Arpack
    using LinearAlgebra
    using LinearMaps
    using Logging
    using Printf
    using Random
    using SparseArrays
    using StatsBase


    """
        _qrDirect!(A)

    Obtain and return the Q factor from the reduced QR factorization of `A`,
    overwriting A in place.
    """
    function _qrDirect!(A)
        return LAPACK.orgqr!(LAPACK.geqrf!(A)...)
    end


    # aux fun for printing floats
    _fmtFloat(x) = @sprintf("%.12f", x)


    """
        subspaceIteration(A::Array{<:Number, 2}, V::Array{<:Number, 2}, K;
                          ϵ=1e-16, use_naive=false, num_logged=100,
                          gap=1.0, μ=1.0) -> (V, dist₂, distInf, res₂, resComp, (idxNaive, idxComp))

    Run subspace iteration to find the eigenvectors/eigenvalues of the symmetric
    matrix `A` with initial guess `V` for `K` iterations or up until achieving
    residual smaller than `ϵ`. If `use_naive == true`, stops when the 2-residual is
    less than `ϵ`. The parameter `num_logged` controls the number of samples from
    the convergence history returned (default: 100). `gap` and `μ` are the eigengap
    and incoherence parameter, if known (default: 1.0).

    Returns:
    - `V`: an orthogonal matrix with the approximate eigenvectors found
    - `dist₂`: a vector containing the history of distances to the true subspace
    - `distInf`: a vector containing the history of rowwise distances to the true subspace
    - `res₂`: the history of the 2-norm of the residuals
    - `resComp`: the history of the composite residuals
    - `idxNaive, idxComp`: the first iterations satisfying the residual stopping criteria
    """
    function subspaceIteration(A::Union{SparseMatrixCSC,Array,LinearMap}, V::Array{<:Number, 2}, K;
                               ϵ=1e-16, use_naive=false, num_logged=100,
                               gap=1.0, μ=1.0, use_theo=false, Vtrue=nothing)
        n, r    = size(V)
        freq    = n ÷ num_logged  # frequency so that we return `num_logged` samples
        dist₂   = []
        distInf = []
        res₂    = []
        resComp = []
        # get true subspace if needed
        Vtrue   = (Vtrue === nothing) ? eigs(A, nev=r, which=:LM)[2] : Vtrue
        # iteration satisfying criterion
        idxNaive = idxComp = K
        for t = 1:K
            V = _qrDirect!(Matrix(A * V))
            # log statistics at sampling intervals
            if (t % freq) == 0
                if use_theo
                    ϵ₂, ϵComp = residual(A, V, μ=μ, gap=gap)
                else
                    ϵ₂, ϵComp = ℓinfPert(A, V, gap=gap)
                end
                # store stopping
                if ϵ₂ ≤ ϵ
                    idxNaive = min(idxNaive, t)
                end
                if ϵComp ≤ ϵ
                    idxComp = min(idxComp, t)
                end
                push!(res₂, ϵ₂); push!(resComp, ϵComp)
                push!(dist₂, first(svds(V - Vtrue * (Vtrue'V),
                                        nsv=1, tol=1e-5, ritzvec=false)[1].S))
                push!(distInf, ℓ₂infProxy(V, Vtrue))
                # debugging output
                @debug("it: $(t), res₂: $(_fmtFloat(ϵ₂)), resI: $(_fmtFloat(ϵComp)), d₂: $(dist₂[end]), dI: $(distInf[end])")
                if use_naive
                    (ϵ₂ ≤ ϵ) && (ϵComp ≤ ϵ) && break
                else
                    (ϵComp ≤ ϵ) && break
                end
            end
        end
        return V, dist₂, distInf, res₂, resComp, (idxNaive, idxComp)
    end


    """
        subspaceIteration(A::Union{SparseMatrixCSC, Array, LinearMap}, V::Array{<:Number, 2};
                          ϵ=1e-16, use_naive=false, gap=1.0, μ=1.0)

    If no iteration number has been specified, run until residual is satisfied.
    """
    function subspaceIteration(A::Union{SparseMatrixCSC, Array, LinearMap}, V::Array{<:Number, 2};
                               ϵ=1e-16, use_naive=false, gap=1.0, μ=1.0)
        if ϵ ≤ 1e-6
            fmtFn = x -> @sprintf("%.6f", x)
        else
            fmtFn = x -> @sprintf("%.12f", x)
        end
        n, r = size(V); idx = 0
        while true
            idx += 1
            V = _qrDirect!(Matrix(A * V))
            ϵ₂, ϵComp = ℓinfPert(A, V, gap=gap)
            if use_naive
                (ϵ₂ ≤ ϵ) && (ϵComp ≤ ϵ) && break
            else
                (ϵComp ≤ ϵ) && break
            end
            @debug("it: $(fmtFn(idx)) - res₂: $(fmtFn(ϵ₂)), resI: $(fmtFn(ϵComp))")
        end
        return V
    end


    """
        powerMethod(A::Union{SparseMatrixCSC, Array}, v₀::Array{<:Number, 1};
                    ϵ=1e-16, use_naive=false, gap=1.0, μ=1.0, use_theo=false,
                    comp_kendall=false, nComp=length(v₀), vTrue=nothing) -> (v, d₂, dI, r₂, rI, (idxNaive, idxComp), τs)

    Run the power method to compute the leading eigenvector of the matrix `A`
    with initial guess `v₀`, with stopping criterion achieving accuracy `ϵ` in
    the 2-norm (if `use_naive == true`), otherwise in the ∞-norm. Optionally,
    can specify the eigenvalue `gap` and the incoherence parameter `μ` of the
    leading eigenvector. If `comp_kendal == true`, computes the kendall rank
    correlation between the true eigenvector `vTrue` and the estimates at each
    step, for the top `nComp` elements. If `vTrue == nothing`, it is computed
    from scratch using `Arpack.eigs`.
    Returns:
    - `v`: the approximate eigenvector
    - `d₂, dI`: a distance history in the 2- and ∞-norms, respectively
    - `r₂, rI`: a residual history, similarly
    - `idxNaive, idxComp`: the first iteration satisfying residual stopping criteria
    - `τs`: a history of kendall τ distances, if `comp_kendall == true`.
    """
    function powerMethod(A::Union{SparseMatrixCSC, Array, LinearMap}, v₀::Array{<:Number, 1};
                         ϵ=1e-16, use_naive=false, gap=1.0, μ=1.0, use_theo=false,
                         comp_kendall=false, nComp=length(v₀), vTrue=nothing)
        dist₂ = []; distI = []; res₂ = []; resI = []
        vTrue = (vTrue == nothing) ? vec(eigs(A, nev=1, which=:LM)[2]) : vTrue
        v     = v₀[:]
        n     = length(v)
        τs    = (comp_kendall) ? [] : nothing
        # stopping indices
        idxNaive = idxComp = n
        for i = 1:n
            v[:]   = A * v; normalize!(v)
            λ      = v' * (A * v)
            res    = A * v - λ .* v
            ϵ₂, ϵI = norm(res), maximum(abs.(res))
            ϵFac   = ϵ₂ / gap
            if use_theo
                ϵComp  = 8 * μ * sqrt(1 / n) * (ϵFac)^2 +
                    2 * ((1 + μ) / gap) * (ϵI + 2 * ϵFac * ϵI)
            else
                vv2E  = res .- v * (v'res)
                nrm0  = maximum(abs.(v))
                nrm1  = maximum(abs.(vv2E))
                ϵComp = min(nrm0 * (ϵFac)^2 + nrm1 * (1 + ϵFac),
                            μ * sqrt(1 / n) * (ϵFac)^2 +
                            ((1 + μ) / gap) * (ϵI + ϵFac * ϵI))
                ϵComp = min(ϵComp, ϵFac)
            end
            # update metrics
            push!(dist₂, min(norm(v - vTrue), norm(v + vTrue)))
            push!(distI, min(maximum(abs.(v - vTrue)), maximum(abs.(v + vTrue))))
            push!(res₂, ϵFac)
            push!(resI, ϵComp)
            (comp_kendall) && push!(τs, min(rankDist(v, vTrue, nComp), rankDist(-v, vTrue, nComp)))
            @debug("it: $(i), res₂: $(_fmtFloat(ϵFac)), resI = $(_fmtFloat(ϵComp)), " *
                   "d₂: $(_fmtFloat(dist₂[end])), dI: $(_fmtFloat(distI[end]))")
            if (ϵFac ≤ ϵ)
                idxNaive = min(idxNaive, i)
            end
            if (ϵComp ≤ ϵ)
                idxComp = min(idxComp, i)
            end
            if use_naive
                (ϵComp ≤ ϵ) && (ϵFac ≤ ϵ) && break
            else
                (ϵComp ≤ ϵ) && break
            end
        end
        return v, dist₂, distI, res₂, resI, (idxNaive, idxComp), τs
    end


    """
        residual(A, V; μ=1.0, gap=1.0) -> (ϵ₂, ϵComp)

    Compute the residual of the approximate subspace `V` with respect to the matrix
    `A`. Optionally, specify the incoherence parameter `μ` (default: `1.0`) and the
    `gap` parameter (default: `1.0`).
    Returns:
    - ϵ₂: the 2-norm of the residual
    - ϵComp: the minimum of the 2-norm and the composite residuals, tailored for 2-∞
      norm convergence
    """
    function residual(A, V; μ=1.0, gap=1.0)
        n, r = size(V)
        # Rayleigh-Ritz
        _, Q, D = schur(V' * Matrix(A * V)); V[:] = V * Q
        E = Matrix(A * V) - D' .* V
        # compute residuals
        ϵ₂    = opnorm(E)
        # ϵ₂    = first(svds(E, nsv=1, ritzvec=false)[1].S)
        ϵInf  = sqrt(maximum(sum(E.^2, dims=2)))
        ϵFac  = (ϵ₂ / gap)
        ϵComp = 8 * μ * sqrt(r / n) * (ϵFac)^2 +
            2 * ((1 + μ * sqrt(r)) / gap) * (ϵInf + 2 * ϵFac * ϵInf)
        return ϵFac, min(ϵComp, ϵ₂)
    end


    """
        ℓinfPert(A, V; gap=1.0)

    Compute the 2→∞ residual, substituting `V` for the true eigenvector
    matrix.
    """
    function ℓinfPert(A, V; gap=1.0)
        # Rayleigh-Ritz
        _, Q, D = schur(V' * Matrix(A * V)); V[:] = V * Q
        E = Matrix(A * V) - D' .* V
        # compute residuals
        ϵ₂    = opnorm(E)
        # ϵ₂    = first(svds(E, nsv=1, ritzvec=false)[1].S)
        ϵInf  = sqrt(maximum(sum(E.^2, dims=2)))
        ϵFac  = (ϵ₂ / gap)
        vv2E  = E - V * (V'E)
        nrm0  = sqrt(maximum(sum(V.^2, dims=2)))
        nrm1  = sqrt(maximum(sum(vv2E.^2, dims=2)))
        ϵComp = 8 * nrm0 * ϵFac^2 + 2 * (nrm1 / gap) * (1 + 2 * ϵFac)
        return ϵFac, min(ϵComp, ϵ₂)
    end


    """
        ℓ₂infProxy(V0, V1)

    Return an upper bound for the `2 -> ∞` distance between `V0` and `V1` using
    the Procrustes solution for the Frobenius distance.
    """
    function ℓ₂infProxy(V0, V1)
        Op = procrustesRot(V0, V1)
        return sqrt(maximum(sum((V0 - V1 * Op).^2, dims=2)))
    end


    """
        procrustesRot(V0, V1) -> O

    Return the orthogonal matrix minimizing
    ``O \\mapsto \\|V_0 - V_1 O\\|_F``, via the closed-form solution using the
    SVD of ``V_1^T V_0``.
    Returns:
    - `O`: the minimizer of the above problem
    """
    function procrustesRot(V0, V1)
        svdObj = svd(V1'V0)
        return svdObj.U * svdObj.Vt
    end


    """
        procrustesDist(V0, V1)

    Return the distance found by ``\\min \\|V_0 - V_1 O\\|_F`` with ``O`` ranging
    over the set of orthogonal matrices.
    """
    function procrustesDist(V0, V1)
        svdObj = svd(V1'V0)
        return norm(V0 - V1 * procrustesRot(V0, V1))
    end


    """
        genIncoherent(n, r) -> (V, μ)

    Generate a `n × r` incoherent subspace `V`. Returns:
    - `V`: the generated subspace
    - `μ`: the incoherence constant
    """
    function genIncoherent(n, r)
        V = Matrix(qr(randn(n, r)).Q)
        μ = sqrt(n / r) * sqrt(maximum(sum(V.^2, dims=2)))
        return V, μ
    end


    """
        genCoherent(n, r) -> (V, μ)

    Generate a `n x r` coherent subspace `V` by sampling `r` canonical basis
    elements uniformly at random. Returns:
    - `V`: the generated subspace
    - `μ`: the incoherence constant
    """
    function genCoherent(n, r)
        V = Matrix{Float64}(I, n, n)
        V = V[:, randperm(n)[1:r]]
        μ = sqrt(n / r) * sqrt(maximum(sum(V.^2, dims=2)))
        return V, μ
    end


    """
        rankDist(v₁, v₂) -> τ

    Compute Kendall's tau-distance for comparing rankings, when `v₁, v₂` are
    real-valued vectors inducing a ranking via their sorted indices, e.g. in
    eigenvector centrality. The distance is adjusted in the scale [0, 1], so
    that `0` and `1` indicate complete agreement and disagreement, resp.
    Returns:
    - `τ`: the ranking distance computed
    """
    function rankDist(v₁, v₂, nComp=length(v₁))
        τ = corkendall(sortperm(v₁, rev=true)[1:nComp], sortperm(v₂, rev=true)[1:nComp])
        return (1 - τ) / 2
    end

end
