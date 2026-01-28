"""Tests for `drued_jorg_numba`.

This test mirrors `tests/test_jorg_drued_jax.py` but exercises the numba
implementation.

We compare against `upenv.fues_jax`, but only on evaluation points that lie on
reference line segments that are not affected by explicit intersection handling.

"""

import os
from itertools import product
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

import upper_envelope as upenv

TEST_DIR = Path(__file__).parent
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def utility_crra(
    consumption: jnp.ndarray, choice: int, params: Dict[str, float]
) -> jnp.ndarray:
    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])
    utility = utility_consumption - (1 - choice) * params["delta"]
    return utility


@njit
def value_func_numba(
    consumption, choice, beta, rho, delta, continuation_at_zero_savings
):
    utility_consumption = (consumption ** (1 - rho) - 1) / (1 - rho)
    utility = utility_consumption - (1 - choice) * delta
    return utility + beta * continuation_at_zero_savings


def interpolate_on_safe_reference_segments(
    ref_m: np.ndarray, ref_y: np.ndarray, m_grid: np.ndarray
):
    dm = ref_m[1:] - ref_m[:-1]
    safe = dm > 0

    weight = (m_grid[None, :] - ref_m[:-1, None]) / (dm[:, None] + 1e-16)
    y_interp = ref_y[:-1, None] + weight * (ref_y[1:] - ref_y[:-1])[:, None]

    outside = (weight < 0.0) | (weight > 1.0)
    y_interp[outside | (~safe[:, None])] = -np.inf

    return np.max(y_interp, axis=0)


@pytest.fixture(autouse=True)
def _jax_x64():
    jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def setup_model():
    params = {"beta": 0.95, "rho": 1.95, "delta": 0.35}
    state_choice_vec = {"lagged_choice": 0, "choice": 0}
    return params, state_choice_vec


@pytest.mark.parametrize(
    "period, numba_enable", product([2, 4, 9, 10, 18], [True, False])
)
def test_drued_jorg_numba_matches_fues_on_safe_segments(
    period, numba_enable, setup_model
):

    # Turn on/off numba JIT compilation as requested
    if numba_enable:
        os.environ["NUMBA_DISABLE_JIT"] = "0"
    else:
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/val{period}.csv",
        delimiter=",",
        dtype=float,
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/pol{period}.csv",
        delimiter=",",
        dtype=float,
    )

    params, state_choice_vec = setup_model

    def value_func_jax(consumption, choice, params):
        return (
            utility_crra(consumption, choice, params) + params["beta"] * value_egm[1, 0]
        )

    ref_m, ref_c, ref_v = upenv.fues_jax(
        endog_grid=jnp.asarray(policy_egm[0, 1:]),
        policy=jnp.asarray(policy_egm[1, 1:]),
        value=jnp.asarray(value_egm[1, 1:]),
        expected_value_zero_savings=value_egm[1, 0],
        value_function=value_func_jax,
        value_function_args=(state_choice_vec["choice"], params),
        n_constrained_points_to_add=len(policy_egm[0, 1:]) // 10,
    )

    ref_m = np.asarray(ref_m)
    ref_v = np.asarray(ref_v)
    valid = ~np.isnan(ref_m)
    ref_m = ref_m[valid]
    ref_v = ref_v[valid]

    m_min = float(np.min(policy_egm[0, 1:]))
    m_max = float(np.max(policy_egm[0, 1:]))
    m_grid = np.linspace(m_min, m_max, 500)

    endog_out, policy_out, value_out = upenv.drued_jorg_numba(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        m_grid=m_grid,
        expected_value_zero_savings=value_egm[1, 0],
        value_function=value_func_numba,
        value_function_args=(
            state_choice_vec["choice"],
            params["beta"],
            params["rho"],
            params["delta"],
            value_egm[1, 0],
        ),
    )

    endog_out = np.asarray(endog_out)
    value_out = np.asarray(value_out)

    assert endog_out[0] == 0.0
    assert value_out[0] == value_egm[1, 0]

    v_ref = interpolate_on_safe_reference_segments(ref_m, ref_v, m_grid)

    good = np.isfinite(v_ref)
    assert good.any(), "No safe reference points found; test setup issue."

    good &= np.isfinite(value_out[1:])

    assert_allclose(value_out[1:][good], v_ref[good], rtol=1e-7, atol=1e-7)
