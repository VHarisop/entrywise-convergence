# Instructions
This is the code accompanying the paper "Entrywise convergence of iterative
methods for eigenproblems" \[1\]. All code has been tested in Julia v1.1. The
core utilities files, `Utils.jl` and `IOUtils.jl`, rely on the following
libraries:

* Arpack
* LightGraphs
* LinearMaps
* StatsBase

The various experiments (following the naming convention `test*.jl`) require:

* `ArgParse`
* `CSV`
* `DataFrames`
* `Distributions`
* `MatrixNetworks` (only for the sweep cut experiment)

To get documentation and instructions on available parameters for each
experiment, run

```bash
julia <script.jl> --help
```

All of the experiments set up a random seed of 999 to ensure reproducibility,
and the default values for arguments are set to the ones used in the
experiments reported in the manuscript, with the exception of dataset-specific
parameters - please refer to the manuscript and supplementary material for
these parameters.

### Obtaining the datasets
All of the datasets used in our experiments are made available via the
[SNAP](https://snap.stanford.edu/data/) network collection. To run the
experiments, you need to create a folder called "data/" in your working
directory containing the following files:

- `com-lj.ungraph.txt` ([Livejournal][livejournal])
- `com-dblp.ungraph.txt` ([Dblp][dblp])
- `ca-HepPh.txt` ([HepPh][hep])
- `ca-AstroPh.txt` ([AstroPh][astro])
- `artist_edges.csv` ([Gemsec][gemsec])


### Enabling debugging output
Our subspace iteration implementation allows debugging output for inspecting
its progress. To enable such output when running `script.jl`, use the following
command:

```bash
JULIA_DEBUG=Utils julia <script.jl> ...
```

### References
\[1\]: Vasileios Charisopoulos, Austin R. Benson and Anil Damle. "Entrywise
convergence of iterative methods for eigenproblems". arXiv preprint
[abs/2002.08491](https://arxiv.org/abs/2002.08491).


[livejournal]: https://snap.stanford.edu/data/com-LiveJournal.html
[dblp]: https://snap.stanford.edu/data/com-DBLP.html
[hep]: https://snap.stanford.edu/data/ca-HepPh.html
[astro]: https://snap.stanford.edu/data/ca-AstroPh.html
[gemsec]: https://snap.stanford.edu/data/gemsec-Facebook.html
