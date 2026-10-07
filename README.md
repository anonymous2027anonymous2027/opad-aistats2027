# Optimal Reweighting for Discrete MCMC

Anonymous code accompanying the AISTATS submission. This release contains the
models, estimators, data columns, and commands needed to run the submission's
experiments, except the LSAC experiment. No previously generated results are
needed.

## Installation

Use Python 3.10 or 3.11. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --list
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Plots are saved as PDF and PNG; no display server or LaTeX installation is needed.
The pinned numerical packages match the tested release environment.

## Quick start

Check every experiment with small budgets:

```bash
python reproduce.py all --quick
```

This uses two replicates and short runs to verify installation and generate
example outputs. Use the commands below for the submission's budgets.
Every run creates a new timestamped directory under `results/`, so earlier
results remain available. `--output-dir PATH` changes that parent directory.

## Reproduce the submission

| Paper experiment | Command | Default setting |
| --- | --- | --- |
| Figure 1: OPAD versus IS | `python reproduce.py illustration` | 101 states; 500 burn-in steps; 1,800 retained draws; seed 7 |
| Figure 2(a,b): Ising15 KL | `python reproduce.py kl --model ising15` | DPVI K=10 and K=100; shared sampling curves |
| Figure 2(c): BVS KL | `python reproduce.py kl --model bvs` | DPVI K=100 |
| Figure 2(d): BslEr KL | `python reproduce.py kl --model bsler` | DPVI K=100; five-node DAGs |
| Figure 3(a,e,i): BVS trade-off | `python reproduce.py tradeoff --model bvs` | 12 seconds per method |
| Figure 3(b,f,j): Ising15 trade-off | `python reproduce.py tradeoff --model ising15` | 12 seconds per method |
| Figure 3(c,g,k): Ising30 trade-off | `python reproduce.py tradeoff --model ising30` | 12 seconds per method |
| Figure 3(d,h,l): Ising40 trade-off | `python reproduce.py tradeoff --model ising40` | 24 seconds per method |
| Appendix D: Ising R-hat | `python reproduce.py rhat` | Ising15/30/40; 12 seconds; 20 replicates |

All KL comparisons use 20,000 budget units and 20 independent replicates.
All trade-off and R-hat comparisons also use 20 replicates. To run the entire
suite with these defaults:

```bash
python reproduce.py all
```

Run a family on all its supported models with `kl`, `tradeoff`, or `rhat`
without `--model`. The shell wrapper provides the same interface:

```bash
./run.sh kl --model ising15 --particles 10 100
```

Useful overrides:

```bash
python reproduce.py tradeoff --model ising40 --chains 3 --seconds 24
python reproduce.py kl --model bvs --chains 3 --evaluations 20000 --particles 100
python reproduce.py rhat --model ising30 --chains 3 --seconds 2 --records 20
```

These commands create new runs; they do not resume or overwrite old runs.
The complete suite is a research experiment, not the quick installation check:
the sampling clocks alone accumulate many minutes across the independent runs.
Wall-clock curves depend on hardware, numerical libraries, and system load.

## Regenerate plots without rerunning experiments

Each completed KL, trade-off, or R-hat run contains `metadata.json` and raw
per-replicate results. Pass that directory to:

```bash
python reproduce.py plot results/RUN_DIRECTORY
```

Replace `RUN_DIRECTORY` with an actual directory printed by the runner.
This regenerates plots in that run directory. It does not repeat sampling.
The illustration can be regenerated quickly using its command and fixed seed.

## What is computed

- **MC:** empirical visitation frequencies.
- **RB/WR:** acceptance-weighted current/proposed-state contributions.
- **OPAD:** target-proportional weights on distinct visited states.
- **OPAD+:** the same weights on distinct visited and evaluated proposal states.
- **NWSS:** neighbor-weighted stochastic search, with 2,000 candidates pruned to
  1,000 when the limit is exceeded.
- **Clyde–Ghosh ratio HT:** every eighth primary state is retained, paired with
  one transition from an independently seeded auxiliary chain. Inclusion
  probabilities use their plug-in normalization estimate. One retained draw
  costs eight primary plus one auxiliary transition. Undefined checkpoints
  are recorded; aggregate curves use checkpoints defined for every replicate.
- **DPVI / DPVI OPAD+:** coordinate particle optimization and its accumulated
  evaluated support, respectively. The KL runner performs the full requested
  number of coordinate-update budget units, including updates after convergence.

There is no burn-in in the KL, trade-off, or R-hat runs. The illustration uses
its specified burn-in. Primary seeds are `0,...,chains-1`; HT auxiliary seeds
are `100000,...,100000+chains-1`. BslEr data use seed 0 and are generated locally.
The independent streams and settings are recorded with each run.

A KL budget unit is a target-evaluation/proposal step for Ising and BVS, and a
proposal attempt for BslEr. Each DPVI coordinate update costs K units; each HT
retained draw costs nine. HT's final checkpoint at budget 20,000 is 19,998.
The exact target reference is used to evaluate KL and errors, not to initialize
particles or calculate HT inclusion corrections.

Trade-off runtimes include the sampler and each estimator's updates, with
primary, auxiliary, and weighting times summed for HT. The Appendix D runner
retains the original R-hat routines' recorded sampler/search-evolution clocks.
BVS target-table construction and exact-reference calculations are shared setup.
The BVS script also produces oracle-C HT diagnostics separately from the main
comparison. Ising evaluation-axis plots reuse their time-run trajectories.

MSE is the mean squared error across replicates; absolute bias is the absolute
mean error; variance is the sample variance with denominator `chains-1`.
MSE bands are empirical 5th–95th percentiles of squared errors. KL bands are
empirical 5th–95th percentiles of KL values. Bias and variance have no bands;
no bootstrap is used. Generalized R-hat is
`sqrt(1 + variance_of_run_means / mean_within_run_variance)`.

## Models and data

- **Ising15:** a periodic 15-spin model, beta=0.5, J=1, h=0.1, each bond counted once.
- **Ising30/40:** periodic zero-field models, beta=0.5. Their score sums both
  neighbors at each site, equivalent to J=2 in a single-counted-bond convention.
- **BVS:** 337 complete observations, 13 selectable predictors and a fixed
  intercept; standardized predictors, uncentered response in weeks; g=337,
  a=3, b=1 and inclusion prior 0.5. The target uses the same pseudoinverse
  computation as the submission. See [data documentation](DATA.md).
- **BslEr:** a five-node random-order ER DAG with expected total degree 2,
  200 observations, coefficient magnitudes uniform on [0.5,2.5] with random
  signs, and noise standard deviation 1. The BGe prior has alpha_mu=1,
  alpha_w=7 and scale 0.5 I. Structure MCMC uses valid edge additions,
  deletions, reversals and a self move, with the Hastings correction.

Exact references enumerate 8,192 BVS models, 32,768 Ising15 states, and
29,281 five-node DAGs. Ising30/40 use the analytic zero expectation of the
sum of spins; their state spaces are not enumerated.

## Organization and checks

- `reproduce.py`, `run.sh`: public command-line interface.
- `om/models/`: the four target families and BVS input columns.
- `om/samplers/`: MCMC, DPVI and stochastic search.
- `om/tools/`: weighted distributions, RB/WR and ratio HT.
- `om/experiments/`: experiment runners and plot generation.
- `tests/`: numerical and accounting checks.

```bash
python -m unittest discover -s tests -v
```

The LSAC data, code, and digitized figures are not included in this release.
