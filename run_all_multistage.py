"""Run multi-stage calibration on both dates."""

import json
import numpy as np
from pathlib import Path

from cdo_copula.distributions import ANTDistribution, Normal
from cdo_copula.copula import ANTCopula
from cdo_copula.cdo import price_all_tranches
from cdo_copula.calibration import calibrate_ant
from cdo_copula.charts import plot_ant_calibration, plot_tranche_fit
from cdo_copula.cli import _load_input, _build_from_input


def run_date(input_path, output_dir):
    data = _load_input(input_path)
    tranches, hazard, curve = _build_from_input(data)

    print(f"\n{'='*60}", flush=True)
    print(f"Calibrating {data['date']}", flush=True)
    print(f"{'='*60}", flush=True)

    cal = calibrate_ant(
        tranches, hazard, curve,
        data["num_names"], data["maturity_years"], data["coupon_frequency"],
        data["recovery_rate"],
        n_knots_m=20, reg_lambda=0.01, rho_init=0.3, maxiter=500, workers=8,
    )

    copula = ANTCopula(cal["rho"], cal["dist_m"], cal["dist_eps"])
    results = price_all_tranches(
        tranches, copula, hazard, curve,
        data["num_names"], data["maturity_years"], data["coupon_frequency"],
        data["recovery_rate"],
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out = {
        "date": data["date"],
        "rho": cal["rho"],
        "n_knots_m": cal["n_knots_m"],
        "m_params": cal["m_params"].tolist(),
        "objective": cal["objective"],
        "converged": cal["converged"],
        "reg_lambda": 0.01,
    }
    with open(f"{output_dir}/ant.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  {'Tranche':<10} {'Market':>10} {'Model':>10} {'Error':>8}")
    print(f"  {'-'*40}")
    for tr, r in zip(tranches, results):
        label = f"{tr.attachment*100:.0f}-{tr.detachment*100:.0f}%"
        if tr.quote_type == "upfront_pct":
            model_uf = (r.pv_default_leg - 0.05 * r.risky_annuity) / (tr.detachment - tr.attachment) * 100
            print(f"  {label:<10} {tr.quote_value:.1f}%uf {model_uf:>8.1f}%uf {model_uf - tr.quote_value:>+7.1f}")
        else:
            print(f"  {label:<10} {tr.quote_value:.1f}bp {r.fair_spread_bps:>8.1f}bp {r.fair_spread_bps - tr.quote_value:>+7.1f}")

    cal_dict = {"rho": cal["rho"], "dist_m": cal["dist_m"], "dist_eps": cal["dist_eps"]}
    plot_ant_calibration(cal_dict, data["date"], f"{output_dir}/ant_calibration_{data['date']}.png")
    plot_tranche_fit(tranches, results, data["date"], output_path=f"{output_dir}/tranche_fit_{data['date']}.png")


if __name__ == "__main__":
    run_date("data/cdx_ig_2005_08_30.yaml", "results/2005-08-30")
    run_date("data/cdx_ig_2004_08_04.yaml", "results/2004-08-04")
