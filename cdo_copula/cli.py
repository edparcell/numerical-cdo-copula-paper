"""Command-line interface."""

import json
from pathlib import Path

import click
import numpy as np
import yaml

from .calibration import calibrate_base_correlation, calibrate_ant
from .cdo import Tranche, price_all_tranches
from .copula import GaussianCopula, ANTCopula
from .distributions import ANTDistribution
from .hazard_rates import FlatHazardRate
from .interest_rates import FlatForwardCurve


def _load_input(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_from_input(data: dict):
    tranches = [
        Tranche(t["attachment"], t["detachment"], t["quote_type"], t["quote_value"])
        for t in data["tranches"]
    ]
    hazard_rate = FlatHazardRate(
        spread=data["index_spread_bps"] / 10000.0,
        recovery=data["recovery_rate"],
    )
    rate_curve = FlatForwardCurve(data["swap_rate_pct"] / 100.0)
    return tranches, hazard_rate, rate_curve


@click.group()
def main():
    """CDO copula model: pricing and calibration."""
    pass


@main.command()
@click.option("--input", "input_path", required=True, help="Path to YAML input file")
@click.option("--rho", type=float, required=True, help="Correlation parameter")
def price(input_path: str, rho: float):
    """Price tranches with Gaussian copula at a given correlation."""
    data = _load_input(input_path)
    tranches, hazard_rate, rate_curve = _build_from_input(data)
    copula = GaussianCopula(rho)

    results = price_all_tranches(
        tranches, copula, hazard_rate, rate_curve,
        data["num_names"], data["maturity_years"], data["coupon_frequency"],
        data["recovery_rate"],
    )

    print(f"\nPricing results for {data['date']} with rho={rho:.4f}")
    print(f"{'Tranche':<12} {'PV_def':>10} {'PV_prem':>10} {'Fair(bp)':>10} {'PV@mkt':>12}")
    print("-" * 58)
    for tr, r in zip(tranches, results):
        label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
        print(f"{label:<12} {r.pv_default_leg:>10.6f} {r.pv_premium_leg:>10.6f} "
              f"{r.fair_spread_bps:>10.1f} {r.pv_at_market:>12.6f}")


@main.command("base-corr")
@click.option("--input", "input_path", required=True, help="Path to YAML input file")
@click.option("--output", "output_path", default=None, help="Output JSON path")
def base_corr(input_path: str, output_path: str | None):
    """Calibrate base correlations."""
    data = _load_input(input_path)
    tranches, hazard_rate, rate_curve = _build_from_input(data)

    print(f"Calibrating base correlations for {data['date']}...")

    bc = calibrate_base_correlation(
        tranches, hazard_rate, rate_curve,
        data["num_names"], data["maturity_years"], data["coupon_frequency"],
        data["recovery_rate"],
    )

    print(f"\n{'Detachment':>12} {'Base Corr':>12}")
    print("-" * 26)
    for det, corr in bc:
        print(f"{det*100:>11.0f}% {corr:>12.4f}")

    from .charts import plot_base_correlations
    plot_base_correlations(bc, data["date"])

    if output_path:
        out = {
            "date": data["date"],
            "base_correlations": [{"detachment": d, "correlation": c} for d, c in bc],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {output_path}")


@main.command("calibrate-ant")
@click.option("--input", "input_path", required=True, help="Path to YAML input file")
@click.option("--n-knots", type=int, default=20, help="Number of ANT knots for h_M")
@click.option("--reg-lambda", type=float, default=0.01, help="Regularisation strength")
@click.option("--rho-init", type=float, default=0.2, help="Initial correlation")
@click.option("--maxiter", type=int, default=200, help="Max generations")
@click.option("--workers", type=int, default=-1, help="Parallel workers (-1 = all cores)")
@click.option("--output", "output_path", default=None, help="Output JSON path")
@click.option("--charts", is_flag=True, help="Produce charts")
def calibrate_ant_cmd(
    input_path: str,
    n_knots: int,
    reg_lambda: float,
    rho_init: float,
    maxiter: int,
    workers: int,
    output_path: str | None,
    charts: bool,
):
    """Calibrate ANT copula model."""
    data = _load_input(input_path)
    tranches, hazard_rate, rate_curve = _build_from_input(data)

    cal = calibrate_ant(
        tranches, hazard_rate, rate_curve,
        data["num_names"], data["maturity_years"], data["coupon_frequency"],
        data["recovery_rate"],
        n_knots_m=n_knots,
        reg_lambda=reg_lambda, rho_init=rho_init, maxiter=maxiter,
        workers=workers,
    )

    # Price at calibrated parameters
    copula = ANTCopula(cal["rho"], cal["dist_m"], cal["dist_eps"])
    results = price_all_tranches(
        tranches, copula, hazard_rate, rate_curve,
        data["num_names"], data["maturity_years"], data["coupon_frequency"],
        data["recovery_rate"],
    )

    print(f"\nANT calibration results for {data['date']}")
    print(f"rho = {cal['rho']:.4f}")
    print(f"\n{'Tranche':<12} {'Market':>10} {'Model':>10} {'PV@mkt':>12}")
    print("-" * 46)
    for tr, r in zip(tranches, results):
        label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
        if tr.quote_type == "upfront_pct":
            mkt = f"{tr.quote_value:.1f}%uf"
            model_upfront = (r.pv_default_leg - 0.05 * r.risky_annuity) / (tr.detachment - tr.attachment) * 100
            mdl = f"{model_upfront:.1f}%uf"
        else:
            mkt = f"{tr.quote_value:.1f}bp"
            mdl = f"{r.fair_spread_bps:.1f}bp"
        print(f"{label:<12} {mkt:>10} {mdl:>10} {r.pv_at_market:>12.8f}")

    if output_path:
        out = {
            "date": data["date"],
            "rho": cal["rho"],
            "n_knots_m": cal["n_knots_m"],
            "m_params": cal["m_params"].tolist(),
            "objective": cal["objective"],
            "converged": cal["converged"],
            "reg_lambda": reg_lambda,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {output_path}")

        # Write fit report
        report_path = str(Path(output_path).parent / "fit_report.txt")
        with open(report_path, "w") as f:
            f.write(f"ANT Calibration Report — {data['date']}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Parameters: n_knots={n_knots}, lambda={reg_lambda}, rho_init={rho_init}\n")
            f.write(f"Calibrated rho: {cal['rho']:.4f}\n")
            f.write(f"Objective: {cal['objective']:.6e}\n")
            f.write(f"Func evals: {cal.get('n_func_evals', 'N/A')}\n\n")
            f.write(f"{'Tranche':<10} {'Market':>10} {'Model':>10} {'Error':>8} {'PV@mkt':>12}\n")
            f.write(f"{'-'*52}\n")
            for tr, r in zip(tranches, results):
                label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
                if tr.quote_type == "upfront_pct":
                    mkt = f"{tr.quote_value:.1f}%uf"
                    model_uf = (r.pv_default_leg - 0.05 * r.risky_annuity) / (tr.detachment - tr.attachment) * 100
                    mdl = f"{model_uf:.1f}%uf"
                    err = f"{model_uf - tr.quote_value:+.1f}"
                else:
                    mkt = f"{tr.quote_value:.1f}bp"
                    mdl = f"{r.fair_spread_bps:.1f}bp"
                    err = f"{r.fair_spread_bps - tr.quote_value:+.1f}"
                f.write(f"{label:<10} {mkt:>10} {mdl:>10} {err:>8} {r.pv_at_market:>12.8f}\n")
            f.write(f"\ndist_m: mu={cal['dist_m'].mu:.4f}, sigma={cal['dist_m'].sigma:.4f}\n")
            f.write(f"dist_eps: Normal(0, 1)\n")
        print(f"Saved report: {report_path}")

    if charts:
        from .charts import plot_ant_calibration, plot_tranche_fit
        chart_dir = str(Path(output_path).parent) + "/" if output_path else ""
        plot_ant_calibration(cal, data["date"], f"{chart_dir}ant_calibration_{data['date']}.png")
        plot_tranche_fit(tranches, results, data["date"], output_path=f"{chart_dir}tranche_fit_{data['date']}.png")


if __name__ == "__main__":
    main()
