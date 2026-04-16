"""Calibration routines for base correlation and ANT copula."""

import numpy as np

from .cdo import Tranche, price_tranche, price_all_tranches
from .copula import GaussianCopula, ANTCopula
from .distributions import ANTDistribution, Normal
from .hazard_rates import FlatHazardRate
from .interest_rates import FlatForwardCurve


def calibrate_base_correlation(
    tranches: list[Tranche],
    hazard_rate: FlatHazardRate,
    rate_curve: FlatForwardCurve,
    n_names: int,
    maturity: float,
    coupon_freq: int,
    recovery: float,
    n_quad: int = 100,
) -> list[tuple[float, float]]:
    """Bootstrap base correlations.

    Returns list of (detachment_point, base_correlation) pairs.
    """
    from scipy.optimize import brentq

    base_corrs = []

    for idx, tranche in enumerate(tranches):
        if idx == 0:
            # Equity tranche: solve directly
            def objective(rho):
                copula = GaussianCopula(rho)
                result = price_tranche(
                    tranche, copula, hazard_rate, rate_curve,
                    n_names, maturity, coupon_freq, recovery, n_quad,
                )
                return result.pv_at_market

            rho = brentq(objective, 0.001, 0.999, xtol=1e-8)
            base_corrs.append((tranche.detachment, rho))
        else:
            # Non-equity: the [a,d] tranche PV = PV(0,d,rho_d) - PV(0,a,rho_a)
            # We already have rho_a from the previous step
            a = tranche.attachment
            d = tranche.detachment
            rho_a = base_corrs[-1][1]  # might not be the right one if a != prev detachment

            # Find the base correlation for the detachment point
            # by finding rho_d such that tranche PV at market spread = 0
            base_tranche_a = Tranche(0.0, a, "spread_bps", 0.0)
            base_tranche_d = Tranche(0.0, d, "spread_bps", 0.0)

            # Precompute the 0-a leg PVs at rho_a
            copula_a = GaussianCopula(rho_a)
            result_a = price_tranche(
                base_tranche_a, copula_a, hazard_rate, rate_curve,
                n_names, maturity, coupon_freq, recovery, n_quad,
            )

            def objective(rho_d):
                copula_d = GaussianCopula(rho_d)
                result_d = price_tranche(
                    base_tranche_d, copula_d, hazard_rate, rate_curve,
                    n_names, maturity, coupon_freq, recovery, n_quad,
                )
                # PV_def of [a,d] tranche
                pv_def_ad = result_d.pv_default_leg - result_a.pv_default_leg
                # Risky annuity of [a,d] tranche
                ra_ad = result_d.risky_annuity - result_a.risky_annuity

                if tranche.quote_type == "upfront_pct":
                    market_spread = 0.05
                    tranche_width = d - a
                    pv_prem = market_spread * ra_ad
                    return pv_def_ad - pv_prem - (tranche.quote_value / 100.0) * tranche_width
                else:
                    market_spread = tranche.quote_value / 10000.0
                    return pv_def_ad - market_spread * ra_ad

            rho_d = brentq(objective, 0.001, 0.999, xtol=1e-8)
            base_corrs.append((d, rho_d))

    return base_corrs


def _ant_objective(x, tranches, hazard_rate, rate_curve, n_names, maturity,
                    coupon_freq, recovery, n_knots_m,
                    reg_lambda, n_quad, bw_scale=1.0):
    """Objective function for ANT calibration.

    Top-level function (not a closure) so it can be pickled for multiprocessing.
    Epsilon is fixed as standard normal.
    bw_scale: multiplier on the auto-computed bump bandwidth.
    """
    # x[0] = rho
    # x[1:1+n_knots_m+3] = h_M params (focus, tightness, strength, y-weights)
    rho = x[0]
    m_params = x[1 : 1 + n_knots_m + 3]

    dist_m = ANTDistribution.from_unconstrained(m_params, n_knots_m, bw_scale=bw_scale)
    dist_eps = Normal()
    copula = ANTCopula(rho, dist_m, dist_eps)

    results = price_all_tranches(
        tranches, copula, hazard_rate, rate_curve,
        n_names, maturity, coupon_freq, recovery, n_quad,
    )

    # Scale PVs to $100mm notional for numerical visibility
    pv_error = sum((r.pv_at_market * 1e8) ** 2 for r in results)
    reg = reg_lambda * dist_m.negentropy()

    return pv_error + reg


def calibrate_ant(
    tranches: list[Tranche],
    hazard_rate: FlatHazardRate,
    rate_curve: FlatForwardCurve,
    n_names: int,
    maturity: float,
    coupon_freq: int,
    recovery: float,
    n_knots_m: int = 20,
    reg_lambda: float = 0.01,
    rho_init: float = 0.2,
    n_quad: int = 100,
    maxiter: int = 500,
    workers: int = -1,
) -> dict:
    """Calibrate the ANT copula model using multi-stage differential evolution.

    Three stages with decreasing bump bandwidth:
    1. Coarse (bw_scale=6): broad coupling, find the right region
    2. Medium (bw_scale=2): moderate coupling, refine
    3. Fine (bw_scale=1): tight coupling, final tuning

    Epsilon is fixed as standard normal. Only h_M is calibrated.
    Uses all available CPU cores by default (workers=-1).
    """
    from scipy.optimize import differential_evolution

    lo, hi = ANTDistribution.KNOT_RANGE
    n_m_params = n_knots_m + 3  # focus, tightness, strength, + y-weights
    n_total = 1 + n_m_params

    # Bounds
    bounds = [(0.01, 0.99)]    # rho
    bounds += [(-6, 0)]        # focus (left tail)
    bounds += [(0.5, 10.0)]    # tightness
    bounds += [(0.0, 1.0)]     # strength
    bounds += [(-5, 5)] * n_knots_m  # softmax y-weights

    # Seed population: identity at several rho values
    m_init = ANTDistribution.identity_params(n_knots_m)
    seed_rhos = [0.05, 0.1, 0.2, 0.3, 0.5]
    init_pop = [np.concatenate([[rho], m_init]) for rho in seed_rhos]

    rng = np.random.default_rng(42)
    popsize = max(len(init_pop) + 10, 2 * n_total)
    bounds_arr = np.array(bounds)
    n_random = popsize - len(init_pop)
    random_rows = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1], size=(n_random, n_total))
    pop_array = np.vstack([np.array(init_pop), random_rows])

    base_args = (tranches, hazard_rate, rate_curve, n_names, maturity,
                 coupon_freq, recovery, n_knots_m, reg_lambda, n_quad)

    print(f"Starting ANT calibration with {n_knots_m} M knots, eps=Normal", flush=True)
    print(f"Total parameters: {n_total}, lambda={reg_lambda}, workers={workers}", flush=True)
    init_obj = _ant_objective(init_pop[2], *base_args)
    print(f"Initial objective (identity, rho=0.2): {init_obj:.6e}", flush=True)
    print(f"Population size: {len(pop_array)}", flush=True)

    # Stage schedule: (name, bw_scale, maxiter_fraction)
    stages = [
        ("Coarse (bw=6x)", 6.0, 0.3),
        ("Medium (bw=2x)", 2.0, 0.3),
        ("Fine (bw=1x)",   1.0, 0.4),
    ]

    total_evals = 0
    best_result = None
    prev_bw_scale = None

    for stage_name, bw_scale, iter_frac in stages:
        stage_maxiter = max(10, int(maxiter * iter_frac))
        args = base_args + (bw_scale,)

        print(f"\n--- {stage_name}: {stage_maxiter} generations ---", flush=True)

        # Transform population y-weights to new basis so softmax
        # inputs are preserved across the bandwidth change
        if prev_bw_scale is not None and prev_bw_scale != bw_scale:
            from .bump_basis import make_bump_matrix_from_positions
            from .focused_grid import focused_grid

            transformed_pop = []
            for member in pop_array:
                focus_p, tight_p, strength_p = member[1], member[2], member[3]
                y_weights_old = member[4:]

                # Build x-grid for this member's focus params
                x_interior = focused_grid(n_knots_m - 1, lo, hi, focus_p, tight_p, strength_p)
                x_full = np.concatenate([[lo], x_interior, [hi]])
                x_midpoints = 0.5 * (x_full[:-1] + x_full[1:])

                B_old = make_bump_matrix_from_positions(x_midpoints, bandwidth_scale=prev_bw_scale)
                B_new = make_bump_matrix_from_positions(x_midpoints, bandwidth_scale=bw_scale)

                # Solve B_new @ w_new = B_old @ w_old
                softmax_inputs = B_old @ y_weights_old
                y_weights_new = np.linalg.solve(B_new, softmax_inputs)

                new_member = member.copy()
                new_member[4:] = y_weights_new
                transformed_pop.append(new_member)

            pop_array = np.array(transformed_pop)
            print(f"  Transformed {len(pop_array)} population members to new basis", flush=True)

        gen_count = [0]
        def make_callback(bw_s):
            def callback(xk, convergence):
                gen_count[0] += 1
                if gen_count[0] <= 3 or gen_count[0] % 10 == 0:
                    obj = _ant_objective(xk, *base_args, bw_s)
                    print(f"  gen {gen_count[0]}: obj={obj:.6e}, rho={xk[0]:.4f}", flush=True)
            return callback

        result = differential_evolution(
            _ant_objective,
            bounds,
            args=args,
            init=pop_array,
            maxiter=stage_maxiter,
            tol=1e-6,
            seed=42 + int(bw_scale * 10),
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
            callback=make_callback(bw_scale),
            polish=False,
        )

        total_evals += result.nfev
        print(f"  Stage result: obj={result.fun:.6e}, rho={result.x[0]:.4f}, evals={result.nfev}", flush=True)

        # Seed next stage from this result + perturbations
        best_x = result.x.copy()
        prev_bw_scale = bw_scale

        pop_list = [best_x]
        for _ in range(popsize - 1):
            perturbed = best_x + rng.normal(0, 0.1, size=n_total)
            perturbed = np.clip(perturbed, bounds_arr[:, 0], bounds_arr[:, 1])
            pop_list.append(perturbed)
        pop_array = np.array(pop_list)

        best_result = result

    # Final result at bw_scale=1 for consistent reporting
    rho_cal = best_result.x[0]
    m_params_cal = best_result.x[1 : 1 + n_knots_m + 3]
    dist_m_cal = ANTDistribution.from_unconstrained(m_params_cal, n_knots_m, bw_scale=1.0)
    dist_eps_cal = Normal()

    print(f"\nFinal objective: {best_result.fun:.6e}, rho={rho_cal:.4f}", flush=True)
    print(f"Total func evals: {total_evals}", flush=True)

    return {
        "rho": rho_cal,
        "dist_m": dist_m_cal,
        "dist_eps": dist_eps_cal,
        "m_params": m_params_cal,
        "n_knots_m": n_knots_m,
        "objective": result.fun,
        "converged": result.success,
        "n_func_evals": result.nfev,
        "optimizer_result": result,
    }
