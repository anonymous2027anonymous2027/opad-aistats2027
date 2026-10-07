#!/usr/bin/env python3
"""Reproduce the IS/OPAD notebook experiment with a symmetric proposal.

Dependencies: numpy, matplotlib.
Run from any directory; figures are saved beside this script by default:
    python is_vs_opad_from_q_draws.py
    python is_vs_opad_from_q_draws.py --output-dir /path/to/figures

At either endpoint, an outward proposal stays at the current state. Thus
Q(y | x) = Q(x | y), including the boundary edges, and the ordinary target
ratio is the correct Metropolis acceptance ratio. Samples are correlated
MCMC output targeting q, not independent draws from q.

The probability-vector error is mean_x[(estimated_p(x) - p(x))**2], not
squared error for an expectation of a particular observable f.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def symmetric_proposal(x, num_states, rng):
    """Choose left/right with probability 1/2; reject steps outside the grid."""
    candidate = x + (1 if rng.random() < 0.5 else -1)
    return candidate if 0 <= candidate < num_states else x


def mh_sample(target_probs, n_steps, start=0, burn_in=0, thin=1, seed=0):
    """Return retained states and acceptance rate (including self-proposals)."""
    target_probs = np.asarray(target_probs, dtype=float)
    if target_probs.ndim != 1 or len(target_probs) < 2:
        raise ValueError("target_probs must be a vector with at least two states")
    if not np.all(np.isfinite(target_probs)) or np.any(target_probs <= 0):
        raise ValueError("Target weights must be finite and strictly positive")
    if not 0 <= start < len(target_probs):
        raise ValueError("start must be a valid state index")
    if not 0 <= burn_in < n_steps or thin < 1:
        raise ValueError("Require 0 <= burn_in < n_steps and thin >= 1")

    rng = np.random.default_rng(seed)
    x = int(start)
    samples = []
    accept_count = 0
    for t in range(n_steps):
        proposal = symmetric_proposal(x, len(target_probs), rng)
        accept_prob = min(1.0, target_probs[proposal] / target_probs[x])
        if rng.random() < accept_prob:
            x = proposal
            accept_count += 1
        if t >= burn_in and (t - burn_in) % thin == 0:
            samples.append(x)
    return np.asarray(samples, dtype=int), accept_count / n_steps


def kl_divergence(approx_dist, target_dist):
    mask = approx_dist > 0
    return float(np.sum(approx_dist[mask] * np.log(approx_dist[mask] / target_dist[mask])))


def probability_vector_mse(approx_dist, target_dist):
    return float(np.mean((approx_dist - target_dist) ** 2))


def is_distribution_from_q_draws(samples, p, q):
    weights = p[samples] / q[samples]
    weights /= weights.sum()
    return np.bincount(samples, weights=weights, minlength=len(p))


def opad_distribution_from_draws(samples, p):
    visited = np.unique(samples)
    approx = np.zeros_like(p)
    approx[visited] = p[visited] / p[visited].sum()
    return approx


def compare_distributions(samples, p, q):
    is_hat = is_distribution_from_q_draws(samples, p, q)
    opad_hat = opad_distribution_from_draws(samples, p)
    metrics = {
        "is_kl": kl_divergence(is_hat, p),
        "opad_kl": kl_divergence(opad_hat, p),
        "is_probability_vector_mse": probability_vector_mse(is_hat, p),
        "opad_probability_vector_mse": probability_vector_mse(opad_hat, p),
    }
    return is_hat, opad_hat, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "is_vs_opad_plots",
        help="Directory for three figures (PNG and PDF) and numerical results",
    )
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    # No interactive display is required; the saved figures work in headless runs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    # Remain legible when the 10-inch figures are reduced to paper minipages.
    plt.rcParams.update({
        # Computer Modern matches the paper's default LaTeX serif family.
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, name, fixed_layout=False):
        # Preserve identical canvas and axes dimensions for the paired plots.
        if not fixed_layout:
            fig.tight_layout()
        bbox = None if fixed_layout else "tight"
        fig.savefig(output_dir / (name + ".png"), dpi=200, bbox_inches=bbox)
        fig.savefig(output_dir / (name + ".pdf"), bbox_inches=bbox)
        plt.close(fig)

    # Model and sampling settings match the notebook.
    states = np.arange(101)
    sigma, rho = 6.0, 2.0
    p = (np.exp(-0.5 * ((states - 25) / sigma) ** 2)
         + np.exp(-0.5 * ((states - 75) / sigma) ** 2))
    p /= p.sum()
    q = p ** (1.0 / rho)
    q /= q.sum()
    start_state, burn_in = 5, 500
    single_length = 1800
    chain_lengths = [50, 100, 200, 500, 1000]

    # The histogram and both estimators use exactly the same retained draws.
    samples, _ = mh_sample(q, single_length + burn_in, start=start_state,
                           burn_in=burn_in, seed=7)
    is_hat, opad_hat, single_metrics = compare_distributions(samples, p, q)
    histogram_counts = np.bincount(samples, minlength=len(states))
    empirical_q = histogram_counts / len(samples)
    ymax = np.ceil(1.08 * max(p.max(), q.max(), empirical_q.max(),
                              is_hat.max(), opad_hat.max()) / 0.01) * 0.01

    def mass_plot(title):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.24, top=0.86)
        ax.set(xlabel="State", ylabel="Probability mass", title=title,
               xlim=(-2, 102), ylim=(-0.025 * ymax, ymax))
        return fig, ax

    fig, ax = mass_plot("Target, auxiliary and observed frequencies")
    ax.bar(states, empirical_q, width=0.8, color="0.55", alpha=0.45,
           label="Empirical auxiliary draws", zorder=2)
    ax.plot(states, p, label=r"Target $p(x)$", linestyle="none",
            marker="o", markersize=4, color="black", markerfacecolor="black", zorder=3)
    ax.plot(states, q, label=r"Auxiliary $q(x)\propto\sqrt{p(x)}$",
            linestyle="none", marker="^", markersize=4, color="tab:orange",
            markerfacecolor="none", zorder=3)
    ax.legend(loc="upper right")
    save_figure(fig, "01_target_and_auxiliary", fixed_layout=True)

    fig, ax = mass_plot("IS and OPAD from the same auxiliary draws")
    ax.plot(states, p, label="True target", linestyle="none",
            marker="o", markersize=4, color="black", markerfacecolor="black")
    ax.plot(states, is_hat, label="IS", linestyle="none",
            marker="x", markersize=4, color="green")
    ax.plot(states, opad_hat, label="OPAD", linestyle="none",
            marker="+", markersize=5, color="red")
    ax.legend(loc="upper right")
    save_figure(fig, "03_single_chain_comparison", fixed_layout=True)

    rows, summaries = [], []
    for n in chain_lengths:
        current_rows = []
        for seed in range(args.repetitions):
            samples, _ = mh_sample(q, n + burn_in, start=start_state,
                                   burn_in=burn_in, seed=seed)
            _, _, metrics = compare_distributions(samples, p, q)
            row = {"n": n, "seed": seed, **metrics}
            rows.append(row)
            current_rows.append(row)
        summaries.append({"n": n, **{
            key: float(np.mean([row[key] for row in current_rows]))
            for key in single_metrics
        }})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(
        axes, ("kl", "probability_vector_mse"),
        (r"$D_{\mathrm{KL}}(\hat p\Vert p)$", "Mean squared probability error"),
    ):
        for method in ("is", "opad"):
            ax.plot(chain_lengths, [row[f"{method}_{metric}"] for row in summaries],
                    marker="o", label=method.upper(),
                    color="green" if method == "is" else "red")
        ax.set(xlabel="Retained states from chain targeting q", ylabel=ylabel,
               title=f"Average over {args.repetitions} independent runs")
        ax.legend()
    save_figure(fig, "04_repeated_run_errors")

    with (output_dir / "repeated_run_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = {
        "settings": {
            "num_states": len(states), "sigma": sigma, "rho": rho,
            "start_state": start_state, "burn_in": burn_in,
            "single_length": single_length, "single_seed": 7,
            "chain_lengths": chain_lengths, "repetitions": args.repetitions,
            "repetition_seeds": list(range(args.repetitions)),
            "proposal": "left/right equiprobable; outward steps stay at endpoint",
            "probability_vector_mse": "mean_x[(estimated_p(x)-p(x))**2]",
        },
        "single_chain": single_metrics,
        "auxiliary_histogram": {
            "counts": histogram_counts.tolist(),
            "probability_mass": empirical_q.tolist(),
        },
        "repeated_run_means": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Single-chain metrics:", single_metrics)
    print(f"Saved three plots in PNG/PDF format and numerical results to {output_dir}")


if __name__ == "__main__":
    main()
