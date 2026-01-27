# Upper Envelope Package

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![PyPI version](https://badge.fury.io/py/upper-envelope.svg)](https://badge.fury.io/py/upper-envelope)
[![Downloads](https://pepy.tech/badge/upper-envelope)](https://pepy.tech/project/upper-envelope)

[![Continuous Integration Workflow](https://github.com/OpenSourceEconomics/upper-envelope/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceEconomics/upper-envelope/actions/workflows/main.yml)
[![Codecov](https://codecov.io/gh/OpenSourceEconomics/upper-envelope/branch/main/graph/badge.svg)](https://app.codecov.io/gh/OpenSourceEconomics/upper-envelope)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package collects several HPC implementations of upper-envelopes used to correct the
value and policy functions in discrete-continuous dynamic programming problems.

The following implementations are available:

- Extension of the Fast Upper-Envelope Scan (FUES) for solving discrete-continuous
  dynamic programming problems based on Dobrescu & Shanker (2022). Both `jax` and
  `numba` versions are available. We provide the original version without endogenous
  jump detection.

- Line segment interpolation and selection of the upper envelope based on Druedahl &
  Jorgensen (2017). Both `jax` and `numba` versions are available.

- Also contained for test reasons is the original upper-envelope implementation from
  Iskhakov et al. (2017). It is not optimized and can not yet be imported when
  installing the package.

## References

1. Dobrescu & Shanker (2022).
   [Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming](https://dx.doi.org/10.2139/ssrn.4181302).

1. Druedahl & Jørgensen (2017).
   [A general endogenous grid method for multi-dimensional models with non-convexities and constraints](https://www.sciencedirect.com/science/article/abs/pii/S0165188916301920).
   *Journal of Economic Dynamics and Control*

1. Iskhakov, Jørgensen, Rust, & Schjerning (2017).
   [The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks](http://onlinelibrary.wiley.com/doi/10.3982/QE643/full).
   *Quantitative Economics*
