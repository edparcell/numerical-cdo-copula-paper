# Nonparametric Copula Model for CDO Tranche Pricing

Reference implementation for the paper *A nonparametric copula model for CDO tranche pricing* (Parcell, 2026).

**Paper:** [edparcell.com/files/ant_copula_cdo_tranche_pricing.pdf](https://www.edparcell.com/files/ant_copula_cdo_tranche_pricing.pdf)

## Overview

The arbitrary normal transform (ANT) copula replaces the Gaussian market factor in the standard one-factor copula model with a nonparametric distribution. This fits the CDO correlation smile to within approximately 1 bp on all liquid tranches, while retaining the structural advantages of the Gaussian copula: a single correlation parameter, consistent single-name default probabilities, and the efficient Andersen-Sidenius-Basu recursion for the loss distribution.

## Installation

Requires Python 3.12+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

### CLI

```bash
# Base correlation bootstrapping
uv run cdo-copula base-corr data/cdx_ig_2005_08_30.yaml

# Price tranches with a flat Gaussian copula
uv run cdo-copula price data/cdx_ig_2005_08_30.yaml --rho 0.3

# Calibrate the ANT copula
uv run cdo-copula calibrate-ant data/cdx_ig_2005_08_30.yaml --n-knots 20 --maxiter 500
```

### Reproduce paper results

```bash
uv run run_all_multistage.py
```

This runs the multi-stage differential evolution calibration on both dates from the paper (2004-08-04 and 2005-08-30). Results are written to `results/`.

## Package structure

- `cdo_copula/` - Core library
  - `distributions.py` - Normal and ANT distributions
  - `copula.py` - Gaussian and ANT copula models
  - `cdo.py` - CDO tranche pricing via Andersen-Sidenius-Basu recursion
  - `calibration.py` - Base correlation bootstrap and ANT calibration (multi-stage DE)
  - `steffen.py` - Steffen (1990) monotonic piecewise cubic interpolation
  - `bump_basis.py` - Gaussian kernel basis for coupling softmax parameters
  - `focused_grid.py` - Non-uniform knot placement with configurable focus region
  - `mathutils.py` - Pure numpy normal distribution and Gauss-Hermite quadrature
  - `copula_cheb.py` - Alternative ANT copula using Chebyshev polynomial convolution
- `data/` - CDX IG 5Y market data (YAML)

## License

MIT
