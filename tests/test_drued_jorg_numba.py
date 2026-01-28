"""Tests for `drued_jorg_numba`.

This test mirrors `tests/test_jorg_drued_jax.py` but exercises the numba
implementation.

We compare against `upenv.fues_jax`, but only on evaluation points that lie on
reference line segments that are not affected by explicit intersection handling.

"""

from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

import upper_envelope as upenv
from tests.utils.comparison_interp import interpolate_on_safe_reference_segments

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


@pytest.fixture()
def setup_model():
    params = {"beta": 0.95, "rho": 1.95, "delta": 0.35}
    state_choice_vec = {"lagged_choice": 0, "choice": 0}
    return params, state_choice_vec


@pytest.mark.parametrize("period", [2, 4, 9, 10, 18])
def test_drued_jorg_numba_matches_fues_on_safe_segments(period, setup_model):

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
    ref_c = np.asarray(ref_c)
    valid = ~np.isnan(ref_m)
    ref_m = ref_m[valid]
    ref_v = ref_v[valid]
    ref_c = ref_c[valid]

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

    v_ref_interp, c_ref_interp = interpolate_on_safe_reference_segments(
        ref_m, ref_v, ref_c, m_grid
    )

    good = ~np.isnan(v_ref_interp)

    # Now the refs live on the same m_grid as outputs. But we cannot compare entries of m_grid which are
    # affected by interpolation
    assert_allclose(value_out[1:][good], v_ref_interp[good], rtol=1e-7, atol=1e-7)
    assert_allclose(policy_out[1:][good], c_ref_interp[good], rtol=1e-7, atol=1e-7)
