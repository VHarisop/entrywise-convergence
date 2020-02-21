#!/usr/bin/env julia

module IOUtils

    using LightGraphs
    using LinearAlgebra
    using LinearMaps
    using DelimitedFiles

    export readGraph, genDBLPMatrix, genLJMatrix, genYoutubeMatrix, genGemsecMatrix


    """
        readGraph(fname::String; comment="#") -> SimpleGraph{Int64}

    Read an undirected graph in vertex pair format from the file `fname`, with
    lines containing comments designated by `comment`. Return a `LightGraph.Graph`
    object containing the read graph without self-loops.
    """
    function readGraph(fname; comment='#', delim=nothing)
        if delim == nothing
            df  = readdlm(fname, Int, comments=true, comment_char=comment)
        else
            df  = readdlm(fname, delim, Int, comments=true, comment_char=comment)
        end
        # get minimum index - if 0, add 1 to every index because we are in Julia
        off = (minimum(df) == 0) ? 1 : 0
        n   = maximum(df) + off
        g   = Graph(n)
        for (i, j) in zip(df[:, 1], df[:, 2])
            # add all except self-loops
            (i ≠ j) && add_edge!(g, i+off, j+off)
        end
        return g[sort(connected_components(g), rev=true, by=length)[1]]
    end


    """
        readGemsecGraph(fname) -> g

    Read a graph in vertex pair format from a Gemsec data file.
    """
    function readGemsecGraph(fname)
        df = readdlm(fname, ',', Int, skipstart=1)
        # get minimum index - if 0, add 1 to every index because we are in Julia
        off = (minimum(df) == 0) ? 1 : 0
        n   = maximum(df) + off
        g   = Graph(n)
        for (i, j) in zip(df[:, 1], df[:, 2])
            # add all except self-loops
            (i ≠ j) && add_edge!(g, i+off, j+off)
        end
        return g[sort(connected_components(g), rev=true, by=length)[1]]
    end


    genGemsecGraph() = readGemsecGraph("data/artist_edges.csv")
    genLJGraph() = readGraph("data/com-lj.ungraph.txt")
    genDBLPGraph() = readGraph("data/com-dblp.ungraph.txt")
    genHepGraph() = readGraph("data/ca-HepPh.txt")
    genAstroGraph() = readGraph("data/ca-AstroPh.txt")


    """
        adjMatrix(g::LightGraphs.Graph, normalized=true) -> (adjMat, d)

    Return the adjacency matrix of `g`, as well as a vector of vertex degrees.
    If `normalized == true`, return the symmetrically normalized adjacency
    matrix, shifted by `+1.0I` to ensure algebraically largest eigenvalues are
    also the largest in magnitude.
    Returns:
    - `adjMat`: the adjacency matrix
    - `d`: a vector containing the degrees of each vertex
    """
    function adjMatrix(g::LightGraphs.Graph, normalized=true)
        g = g[sort(connected_components(g), rev=true, by=length)[1]]  # coerce first conn. component
        D = normalized ? Diagonal(1 ./ sqrt.(degree(g))) : 1.0I
        if normalized  # return shifted matrix for identif. issues
            return D * adjacency_matrix(g) * D + 1.0I, degree(g)
        else
            return D * adjacency_matrix(g) * D, degree(g)
        end
    end


    function adjMatrixReg(g::LightGraphs.Graph, τ)
        g = g[connected_components(g)[1]]  # coerce first conn. component
        A = adjacency_matrix(g)
        D = Diagonal(1 ./ sqrt.(degree(g) .+ τ))
        Aop = LinearMap(X -> A * X .+ (τ / nv(g)) * sum(X), size(A)..., issymmetric=true)
        return D * Aop * D
    end


    genGemsecMatrix(normalized) = adjMatrix(genGemsecGraph(), normalized)
    genLJMatrix(normalized) = adjMatrix(genLJGraph(), normalized)
    genDBLPMatrix(normalized) = adjMatrix(genDBLPGraph(), normalized)
    genHepMatrix(normalized) = adjMatrix(genHepGraph(), normalized)
    genAstroMatrix(normalized) = adjMatrix(genAstroGraph(), normalized)
end
