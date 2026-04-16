"""Reporting functions for fit comparison tables."""

import json
from pathlib import Path

import numpy as np

from .cdo import Tranche, price_all_tranches
from .copula import GaussianCopula, ANTCopula
from .distributions import ANTDistribution
from .hazard_rates import FlatHazardRate
from .interest_rates import FlatForwardCurve


def _format_tranche(tr: Tranche, result) -> tuple[str, str, str]:
    """Return (label, market_str, model_str) for a tranche."""
    label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
    if tr.quote_type == "upfront_pct":
        mkt = f"{tr.quote_value:.1f}%uf"
        model_uf = (result.pv_default_leg - 0.05 * result.risky_annuity) / (tr.detachment - tr.attachment) * 100
        mdl = f"{model_uf:.1f}%uf"
        err = model_uf - tr.quote_value
    else:
        mkt = f"{tr.quote_value:.1f}bp"
        mdl = f"{result.fair_spread_bps:.1f}bp"
        err = result.fair_spread_bps - tr.quote_value
    return label, mkt, mdl, f"{err:+.1f}"


def fit_comparison_table(
    date: str,
    tranches: list[Tranche],
    hazard_rate: FlatHazardRate,
    rate_curve: FlatForwardCurve,
    ant_runs: dict[str, str] | None = None,
    gaussian_rhos: list[float] | None = None,
):
    """Print a comparison table of market vs model tranche quotes.

    ant_runs: dict of {label: path_to_ant_json}
    gaussian_rhos: list of flat rho values to include
    """
    n_names, maturity, coupon_freq, recovery = 125, 5.0, 4, 0.40

    if gaussian_rhos is None:
        gaussian_rhos = []

    # Build all model results
    models = {}

    for rho in gaussian_rhos:
        gc = GaussianCopula(rho)
        results = price_all_tranches(
            tranches, gc, hazard_rate, rate_curve,
            n_names, maturity, coupon_freq, recovery,
        )
        models[f"Gauss {rho:.2f}"] = (results, rho)

    if ant_runs:
        for label, json_path in ant_runs.items():
            with open(json_path) as f:
                cal = json.load(f)
            dist_m = ANTDistribution.from_unconstrained(
                np.array(cal["m_params"]), cal["n_knots_m"]
            )
            dist_eps = ANTDistribution.from_unconstrained(
                np.array(cal["eps_params"]), cal["n_knots_eps"]
            )
            copula = ANTCopula(cal["rho"], dist_m, dist_eps)
            results = price_all_tranches(
                tranches, copula, hazard_rate, rate_curve,
                n_names, maturity, coupon_freq, recovery,
            )
            models[label] = (results, cal["rho"])

    # Print table
    model_names = list(models.keys())
    col_width = max(14, max(len(n) for n in model_names) + 2)

    header = f"{'Tranche':<10} {'Market':>{col_width}}"
    for name in model_names:
        header += f" {name:>{col_width}}"
    print(f"\nCDX IG 5Y — {date}")
    print(header)
    print("-" * len(header))

    for i, tr in enumerate(tranches):
        label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
        if tr.quote_type == "upfront_pct":
            mkt = f"{tr.quote_value:.1f}%uf"
        else:
            mkt = f"{tr.quote_value:.1f}bp"

        row = f"{label:<10} {mkt:>{col_width}}"
        for name in model_names:
            results, rho = models[name]
            r = results[i]
            if tr.quote_type == "upfront_pct":
                model_uf = (r.pv_default_leg - 0.05 * r.risky_annuity) / (tr.detachment - tr.attachment) * 100
                cell = f"{model_uf:.1f}%uf"
            else:
                cell = f"{r.fair_spread_bps:.1f}bp"
            row += f" {cell:>{col_width}}"
        print(row)

    # Print rho values
    row = f"{'rho':<10} {'':>{col_width}}"
    for name in model_names:
        _, rho = models[name]
        row += f" {rho:>{col_width}.4f}"
    print(row)
    print()
