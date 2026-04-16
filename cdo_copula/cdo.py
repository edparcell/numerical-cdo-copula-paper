"""CDO tranche pricing."""

from dataclasses import dataclass

import numpy as np

from .copula import CopulaModel
from .hazard_rates import FlatHazardRate
from .interest_rates import FlatForwardCurve


@dataclass
class Tranche:
    attachment: float
    detachment: float
    quote_type: str  # "upfront_pct" or "spread_bps"
    quote_value: float


@dataclass
class TranchePricingResult:
    pv_default_leg: float
    pv_premium_leg: float  # at market-quoted spread (or 500bp for equity)
    fair_spread_bps: float
    pv_at_market: float
    risky_annuity: float


def _loss_distribution(
    n_names: int,
    conditional_default_prob: float,
) -> np.ndarray:
    """Build loss distribution for a homogeneous pool via recursion.

    Returns array of length n_names + 1 where L[j] = P(j defaults | M=m).
    """
    L = np.zeros(n_names + 1)
    L[0] = 1.0
    q = conditional_default_prob

    for _ in range(n_names):
        L[1:] = (1.0 - q) * L[1:] + q * L[:-1]
        L[0] *= (1.0 - q)

    return L


def _expected_tranche_loss(
    loss_dist: np.ndarray,
    attachment: float,
    detachment: float,
    loss_per_unit: float,
) -> float:
    """Expected loss absorbed by a tranche given the loss distribution."""
    etl = 0.0
    for j in range(len(loss_dist)):
        portfolio_loss = j * loss_per_unit
        tranche_loss = min(portfolio_loss, detachment) - min(portfolio_loss, attachment)
        etl += tranche_loss * loss_dist[j]
    return etl


def price_tranche(
    tranche: Tranche,
    copula: CopulaModel,
    hazard_rate: FlatHazardRate,
    rate_curve: FlatForwardCurve,
    n_names: int,
    maturity: float,
    coupon_freq: int,
    recovery: float,
    n_quad: int = 100,
) -> TranchePricingResult:
    """Price a single CDO tranche."""
    loss_per_unit = (1.0 - recovery) / n_names
    coupon_dates = np.arange(1, int(maturity * coupon_freq) + 1) / coupon_freq
    n_periods = len(coupon_dates)
    alpha = 1.0 / coupon_freq  # day count fraction

    m_values, weights = copula.quadrature_points(n_quad)

    # Accumulate unconditional expected tranche losses at each coupon date
    etl = np.zeros(n_periods)

    for m, w in zip(m_values, weights):
        for i, t in enumerate(coupon_dates):
            p_uncond = hazard_rate.default_prob(t)
            p_cond = copula.conditional_default_prob(p_uncond, m)
            loss_dist = _loss_distribution(n_names, p_cond)
            cond_etl = _expected_tranche_loss(
                loss_dist, tranche.attachment, tranche.detachment, loss_per_unit
            )
            etl[i] += w * cond_etl

    # Compute leg PVs
    tranche_width = tranche.detachment - tranche.attachment
    etl_with_zero = np.concatenate([[0.0], etl])

    # Default leg: PV of incremental losses, discounted at midpoints
    pv_def = 0.0
    for i in range(n_periods):
        t_mid = 0.5 * (coupon_dates[i] + (coupon_dates[i - 1] if i > 0 else 0.0))
        pv_def += rate_curve.df(t_mid) * (etl_with_zero[i + 1] - etl_with_zero[i])

    # Premium leg: risky annuity
    risky_annuity = 0.0
    for i in range(n_periods):
        outstanding = tranche_width - 0.5 * (etl_with_zero[i] + etl_with_zero[i + 1])
        risky_annuity += rate_curve.df(coupon_dates[i]) * alpha * outstanding

    fair_spread = pv_def / risky_annuity if risky_annuity > 0 else 0.0

    # PV at market quote
    if tranche.quote_type == "upfront_pct":
        market_spread = 0.05  # 500bp running for equity
        pv_prem = market_spread * risky_annuity
        pv_at_market = pv_def - pv_prem - (tranche.quote_value / 100.0) * tranche_width
    else:
        market_spread = tranche.quote_value / 10000.0
        pv_prem = market_spread * risky_annuity
        pv_at_market = pv_def - pv_prem

    return TranchePricingResult(
        pv_default_leg=pv_def,
        pv_premium_leg=pv_prem,
        fair_spread_bps=fair_spread * 10000.0,
        pv_at_market=pv_at_market,
        risky_annuity=risky_annuity,
    )


def price_all_tranches(
    tranches: list[Tranche],
    copula: CopulaModel,
    hazard_rate: FlatHazardRate,
    rate_curve: FlatForwardCurve,
    n_names: int,
    maturity: float,
    coupon_freq: int,
    recovery: float,
    n_quad: int = 100,
) -> list[TranchePricingResult]:
    """Price all tranches. Reuses the loss distribution computation."""
    loss_per_unit = (1.0 - recovery) / n_names
    coupon_dates = np.arange(1, int(maturity * coupon_freq) + 1) / coupon_freq
    n_periods = len(coupon_dates)
    alpha = 1.0 / coupon_freq
    n_tranches = len(tranches)

    m_values, weights = copula.quadrature_points(n_quad)

    # Accumulate expected tranche losses: shape (n_tranches, n_periods)
    etl = np.zeros((n_tranches, n_periods))

    for m, w in zip(m_values, weights):
        for i, t in enumerate(coupon_dates):
            p_uncond = hazard_rate.default_prob(t)
            p_cond = copula.conditional_default_prob(p_uncond, m)
            loss_dist = _loss_distribution(n_names, p_cond)

            for k, tr in enumerate(tranches):
                cond_etl = _expected_tranche_loss(
                    loss_dist, tr.attachment, tr.detachment, loss_per_unit
                )
                etl[k, i] += w * cond_etl

    results = []
    for k, tr in enumerate(tranches):
        tranche_width = tr.detachment - tr.attachment
        etl_k = np.concatenate([[0.0], etl[k]])

        pv_def = 0.0
        for i in range(n_periods):
            t_mid = 0.5 * (coupon_dates[i] + (coupon_dates[i - 1] if i > 0 else 0.0))
            pv_def += rate_curve.df(t_mid) * (etl_k[i + 1] - etl_k[i])

        risky_annuity = 0.0
        for i in range(n_periods):
            outstanding = tranche_width - 0.5 * (etl_k[i] + etl_k[i + 1])
            risky_annuity += rate_curve.df(coupon_dates[i]) * alpha * outstanding

        fair_spread = pv_def / risky_annuity if risky_annuity > 0 else 0.0

        if tr.quote_type == "upfront_pct":
            market_spread = 0.05
            pv_prem = market_spread * risky_annuity
            pv_at_market = pv_def - pv_prem - (tr.quote_value / 100.0) * tranche_width
        else:
            market_spread = tr.quote_value / 10000.0
            pv_prem = market_spread * risky_annuity
            pv_at_market = pv_def - pv_prem

        results.append(TranchePricingResult(
            pv_default_leg=pv_def,
            pv_premium_leg=pv_prem,
            fair_spread_bps=fair_spread * 10000.0,
            pv_at_market=pv_at_market,
            risky_annuity=risky_annuity,
        ))

    return results
