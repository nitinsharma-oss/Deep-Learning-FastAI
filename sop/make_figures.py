"""make_figures.py -- generates ALL result graphs (PNG for viewing + PDF for
LaTeX) from the training artifacts. Called automatically by train_all.py.

Figures
-------
fig1_convergence : DRL convergence curves vs grid reference (Mbit/s).
fig2_final_adr   : bar chart of final expected mean ADR per algorithm.
fig3_fairness    : per-user E[ADR] and HARQ-IR AoI, ADR-optimal vs AoI-optimal
                   orientation (the throughput/fairness trade-off).
fig4_outage      : outage probability vs transmit SNR at the learned
                   orientation -- analytical Gamma (Meijer-G reduction) vs
                   Monte Carlo, truncated-Laplace vs uniform orientation,
                   with the zero-outage SNR cliff.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammainc

from oirs_vlc_env import (OIRSVLCEnv, trunc_laplace_moments,
                          sample_trunc_laplace, XI_MAX)

RATE_R = 2.0


def _smooth(x, w=150):
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def _save(fig, stem):
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=160)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"saved {stem}.png / .pdf")


def fig_convergence(curves, ref_mbps):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = {"DDPG": "#1f77b4", "SAC": "#d62728", "PPO": "#2ca02c",
              "CAPG": "#9467bd", "ECDPO": "#ff7f0e"}
    for name in ["DDPG", "SAC", "PPO", "CAPG", "ECDPO"]:
        y = _smooth(curves[name] / 1e6)
        ax.plot(np.arange(len(y)) + 150, y, label=name, color=colors[name], lw=1.5)
    ax.axhline(ref_mbps, color="k", ls="--", lw=1.1,
               label=f"Grid reference ({ref_mbps:.2f} Mbit/s)")
    ax.set_xlabel("Environment interaction")
    ax.set_ylabel("Sampled mean ADR (Mbit/s, moving avg.)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, "fig1_convergence")


def fig_final_adr(results, ref_mbps):
    names = ["DDPG", "SAC", "PPO", "CAPG", "ECDPO"]
    vals = [results[n]["mean_adr_mbps"] for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(names, vals, color=["#1f77b4", "#d62728", "#2ca02c",
                                      "#9467bd", "#ff7f0e"])
    ax.axhline(ref_mbps, color="k", ls="--", lw=1.1,
               label=f"Grid reference ({ref_mbps:.2f})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Final expected mean ADR (Mbit/s)")
    ax.set_ylim(0, ref_mbps * 1.15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig2_final_adr")


def fig_fairness(results):
    r_adr = results["ECDPO"]          # ADR-optimal (narrow beam)
    r_aoi = results["ECDPO-AoI"]      # AoI-optimal (fairness config)
    K = len(r_adr["adr_k_mbps"])
    x = np.arange(K)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(x - 0.18, r_adr["adr_k_mbps"], 0.36, label="ADR-optimal",
                color="#ff7f0e")
    axes[0].bar(x + 0.18, r_aoi["adr_k_mbps"], 0.36, label="AoI-optimal",
                color="#2ca02c")
    axes[0].set_xticks(x, [f"UE {k+1}" for k in range(K)])
    axes[0].set_ylabel("Expected ADR (Mbit/s)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x - 0.18, r_adr["aoi_k"], 0.36, label="ADR-optimal",
                color="#ff7f0e")
    axes[1].bar(x + 0.18, r_aoi["aoi_k"], 0.36, label="AoI-optimal",
                color="#2ca02c")
    axes[1].set_xticks(x, [f"UE {k+1}" for k in range(K)])
    axes[1].set_ylabel(r"Average AoI $\overline{\Delta}_{k,IR}$ (rounds)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    _save(fig, "fig3_fairness")


def fig_outage(results, xi_min_fig=0.20, n_mc=200_000, seed=1):
    """Outage vs SNR at the ECDPO-learned orientation, best-served user.
    Truncated-Laplace vs uniform orientation; analytical Gamma vs MC; cliff."""
    rng = np.random.default_rng(seed)
    env = OIRSVLCEnv(seed=0)
    g = np.deg2rad(results["ECDPO"]["gamma_deg"])
    w = np.deg2rad(results["ECDPO"]["omega_deg"])
    ak = env.a_k(g, w)
    k_best = int(np.argmax(ak))
    a = float(ak[k_best]); mu = float(env.mu_xi[k_best]); MN = env.MN

    # ---- truncated-Laplace statistics with FoV bound xi_min_fig ----
    e_xi, var_xi = trunc_laplace_moments(mu, lo=xi_min_fig, hi=XI_MAX)
    kg = MN * e_xi ** 2 / var_xi
    th = a * var_xi / e_xi
    LB = MN * a * xi_min_fig

    # ---- uniform orientation baseline on the same support ----
    e_u = 0.5 * (xi_min_fig + XI_MAX)
    var_u = (XI_MAX - xi_min_fig) ** 2 / 12.0
    kg_u = MN * e_u ** 2 / var_u
    th_u = a * var_u / e_u

    # MC samples of Z (common for all SNR points)
    S_lap = sample_trunc_laplace(mu, (n_mc, MN), rng,
                                 lo=xi_min_fig, hi=XI_MAX).sum(axis=1) * a
    S_uni = rng.uniform(xi_min_fig, XI_MAX, (n_mc, MN)).sum(axis=1) * a
    # conventional uniform assumption with support pulled to zero (no cliff)
    e_u0 = 0.5 * XI_MAX
    var_u0 = XI_MAX ** 2 / 12.0
    kg_u0 = MN * e_u0 ** 2 / var_u0
    th_u0 = a * var_u0 / e_u0
    S_u0 = rng.uniform(0.0, XI_MAX, (n_mc, MN)).sum(axis=1) * a

    gdB = np.linspace(112, 152, 200)
    gbar = 10 ** (gdB / 10)
    zth = np.sqrt((2 * np.pi / np.e) * (2 ** (2 * RATE_R) - 1) / gbar)

    p_lap = np.clip(gammainc(kg, zth / th) - gammainc(kg, LB / th), 0, 1)
    p_lap[zth <= LB] = 0.0                       # zero-outage regime (exact)
    p_uni = np.clip(gammainc(kg_u, zth / th_u) - gammainc(kg_u, LB / th_u), 0, 1)
    p_uni[zth <= LB] = 0.0
    p_u0 = np.clip(gammainc(kg_u0, zth / th_u0), 0, 1)   # LB = 0: never zero

    idx = np.linspace(0, len(gdB) - 1, 14).astype(int)
    mc_lap = [(S_lap < z).mean() for z in zth[idx]]
    mc_uni = [(S_uni < z).mean() for z in zth[idx]]
    mc_u0 = [(S_u0 < z).mean() for z in zth[idx]]

    cliff = (2 * np.pi / np.e) * (2 ** (2 * RATE_R) - 1) / LB ** 2
    cliff_dB = 10 * np.log10(cliff)

    fl = 1e-6                                    # semilogy floor
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(gdB, np.maximum(p_lap, fl), "b-", lw=1.8,
                label="Trunc.-Laplace (analytical)")
    ax.semilogy(gdB[idx], np.maximum(mc_lap, fl), "bo", ms=5, mfc="none",
                label="Trunc.-Laplace (Monte Carlo)")
    ax.semilogy(gdB, np.maximum(p_uni, fl), "r--", lw=1.8,
                label="Uniform (analytical)")
    ax.semilogy(gdB[idx], np.maximum(mc_uni, fl), "rs", ms=5, mfc="none",
                label="Uniform (Monte Carlo)")
    ax.semilogy(gdB, np.maximum(p_u0, fl), "g-.", lw=1.6,
                label=r"Uniform, $\xi_{\min}\to 0$ (no cliff)")
    ax.semilogy(gdB[idx], np.maximum(mc_u0, fl), "g^", ms=5, mfc="none")
    ax.axvline(cliff_dB, color="k", ls=":", lw=1.4,
               label=f"Zero-outage cliff ({cliff_dB:.1f} dB)")
    ax.set_xlabel(r"Average transmit SNR $\bar{\gamma}$ (dB)")
    ax.set_ylabel(r"Outage probability $P_{out}$")
    ax.set_ylim(fl, 1.5)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    _save(fig, "fig4_outage")
    return cliff_dB


def main():
    curves = dict(np.load("curves.npz"))
    results = json.load(open("results.json"))
    ref = results["Grid reference"]["mean_adr_mbps"]
    fig_convergence(curves, ref)
    fig_final_adr(results, ref)
    if "ECDPO-AoI" in results:
        fig_fairness(results)
    cliff_dB = fig_outage(results)
    print(f"zero-outage cliff at {cliff_dB:.2f} dB")


if __name__ == "__main__":
    main()
