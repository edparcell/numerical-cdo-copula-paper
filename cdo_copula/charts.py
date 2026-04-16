"""Charting functions for calibration results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .distributions import ANTDistribution, Normal
from .mathutils import norm_pdf


def plot_ant_calibration(cal: dict, date: str, output_path: str | None = None):
    """Full calibration chart: ANT functions, densities, and fit summary."""
    dist_m = cal["dist_m"]
    dist_eps = cal["dist_eps"]
    rho = cal["rho"]

    x = np.linspace(-5, 5, 500)

    # For h_M plot: extend x-range to cover knots if needed
    if hasattr(dist_m, '_knot_x'):
        kx_std = (dist_m._knot_x - dist_m.mu) / dist_m.sigma
        x_hm_lo = min(-5, np.min(kx_std) - 1)
        x_hm_hi = max(5, np.max(kx_std) + 1)
    else:
        x_hm_lo, x_hm_hi = -5, 5
    x_hm = np.linspace(x_hm_lo, x_hm_hi, 500)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle(f"ANT Copula Calibration — {date}\n$\\rho$ = {rho:.4f}", fontsize=14)

    # 1. h_M function
    ax1 = fig.add_subplot(gs[0, 0])
    h_m_vals = np.array([dist_m._h_raw(dist_m._to_raw(xi)) for xi in x_hm])
    ax1.plot(x_hm, h_m_vals, "C0", linewidth=2, label="$h_M$")
    ax1.plot(x_hm, x_hm, "--", color="gray", alpha=0.5, label="$y=x$")
    # Mark control points
    if hasattr(dist_m, '_knot_x'):
        kx = (dist_m._knot_x - dist_m.mu) / dist_m.sigma
        ky = dist_m._knot_y
        ax1.plot(kx, ky, "o", color="C0", markersize=6)
    ax1.set_title("Market factor $h_M$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$h_M(x)$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. h_eps function (or note if Normal)
    ax2 = fig.add_subplot(gs[0, 1])
    if hasattr(dist_eps, '_h_raw'):
        h_e_vals = np.array([dist_eps._h_raw(dist_eps._to_raw(xi)) for xi in x])
        ax2.plot(x, h_e_vals, "C1", linewidth=2, label="$h_\\varepsilon$")
        ax2.plot(x, x, "--", color="gray", alpha=0.5, label="$y=x$")
        if hasattr(dist_eps, '_knot_x'):
            kx = (dist_eps._knot_x - dist_eps.mu) / dist_eps.sigma
            ky = dist_eps._knot_y
            ax2.plot(kx, ky, "o", color="C1", markersize=6)
    else:
        ax2.plot(x, x, "C1", linewidth=2, label="$h_\\varepsilon = x$ (Normal)")
        ax2.plot(x, x, "--", color="gray", alpha=0.5)
        h_e_vals = x
    ax2.set_title("Idiosyncratic $h_\\varepsilon$")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$h_\\varepsilon(x)$")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. h_M - x (deviation from normal)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x_hm, h_m_vals - x_hm, "C0", linewidth=2, label="$h_M(x) - x$")
    if hasattr(dist_eps, '_h_raw'):
        ax3.plot(x, h_e_vals - x, "C1", linewidth=2, label="$h_\\varepsilon(x) - x$")
    ax3.axhline(0, color="gray", alpha=0.5, linestyle="--")
    ax3.set_title("Deviation from normal")
    ax3.set_xlabel("$x$")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Market factor density
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(x, dist_m.pdf(x), "C0", linewidth=2, label="$f_M$")
    ax4.plot(x, norm_pdf(x), "--", color="gray", alpha=0.5, label="Normal")
    ax4.set_title("Market factor density")
    ax4.set_xlabel("$x$")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Idiosyncratic density
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(x, dist_eps.pdf(x), "C1", linewidth=2, label="$f_\\varepsilon$")
    ax5.plot(x, norm_pdf(x), "--", color="gray", alpha=0.5, label="Normal")
    ax5.set_title("Idiosyncratic density")
    ax5.set_xlabel("$x$")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Log densities (to see tail behaviour)
    ax6 = fig.add_subplot(gs[1, 2])
    pdf_m = np.asarray(dist_m.pdf(x), dtype=float)
    pdf_n = norm_pdf(x)
    ax6.plot(x, np.log10(np.maximum(pdf_m, 1e-20)), "C0", linewidth=2, label="$f_M$")
    ax6.plot(x, np.log10(np.maximum(pdf_n, 1e-20)), "--", color="gray", alpha=0.5, label="Normal")
    ax6.set_title("Log$_{10}$ densities (tail behaviour)")
    ax6.set_xlabel("$x$")
    ax6.set_ylim(-8, 0)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.savefig(f"ant_calibration_{date}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: ant_calibration_{date}.png")
    plt.close()


def plot_base_correlations(
    base_corrs: list[tuple[float, float]],
    date: str,
    output_path: str | None = None,
):
    """Plot base correlation curve."""
    dets = [d * 100 for d, _ in base_corrs]
    corrs = [c * 100 for _, c in base_corrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dets, corrs, "o-", color="C0", linewidth=2, markersize=8)
    ax.set_title(f"Base Correlation Curve — {date}", fontsize=13)
    ax.set_xlabel("Detachment Point (%)")
    ax.set_ylabel("Base Correlation (%)")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.savefig(f"base_corr_{date}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: base_corr_{date}.png")
    plt.close()


def plot_base_correlations_multi(
    all_base_corrs: dict[str, list[tuple[float, float]]],
    output_path: str | None = None,
):
    """Plot base correlation curves for multiple dates on one chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (date, bc) in enumerate(sorted(all_base_corrs.items())):
        dets = [d * 100 for d, _ in bc]
        corrs = [c * 100 for _, c in bc]
        ax.plot(dets, corrs, "o-", color=f"C{i}", linewidth=2, markersize=6, label=date)

    ax.set_title("Base Correlation Curves", fontsize=13)
    ax.set_xlabel("Detachment Point (%)")
    ax.set_ylabel("Base Correlation (%)")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 80)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_path or "base_corr_all.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_tranche_fit(
    tranches,
    results,
    date: str,
    model_name: str = "ANT",
    output_path: str | None = None,
):
    """Bar chart comparing market vs model tranche spreads."""
    labels = []
    market_vals = []
    model_vals = []

    for tr, r in zip(tranches, results):
        label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
        labels.append(label)
        if tr.quote_type == "upfront_pct":
            market_vals.append(tr.quote_value)
            # Convert model fair spread to approximate upfront
            # (rough: PV_def - PV_prem(500bp)) / tranche_width * 100
            model_upfront = (r.pv_default_leg - 0.05 * r.risky_annuity) / (tr.detachment - tr.attachment) * 100
            model_vals.append(model_upfront)
        else:
            market_vals.append(tr.quote_value)
            model_vals.append(r.fair_spread_bps)

    x_pos = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width/2, market_vals, width, label="Market", color="C0", alpha=0.8)
    ax.bar(x_pos + width/2, model_vals, width, label=model_name, color="C1", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(f"Tranche Spreads: Market vs {model_name} — {date}", fontsize=13)
    ax.set_ylabel("Spread (bp) / Upfront (%)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    path = output_path or f"tranche_fit_{date}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
