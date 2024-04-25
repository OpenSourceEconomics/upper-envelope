"""Test the JAX implementation of the fast upper envelope scan."""
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from upper_envelope.fues_jax.check_and_scan_funcs import back_and_forward_scan_wrapper
from upper_envelope.fues_jax.fues_jax import fast_upper_envelope
from upper_envelope.fues_jax.fues_jax import (
    fast_upper_envelope_wrapper,
)
from upper_envelope.fues_numba import fues_numba as fues_nb

from tests.utils.interpolation import interpolate_policy_and_value_on_wealth_grid
from tests.utils.interpolation import linear_interpolation_with_extrapolation
from tests.utils.upper_envelope_fedor import upper_envelope

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def utility_crra(
    consumption: jnp.array,
    choice: int,
    params: Dict[str, float],
) -> jnp.array:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = utility_consumption - (1 - choice) * params["delta"]

    return utility


@pytest.fixture()
def setup_model():
    max_wealth = 50
    n_grid_wealth = 500
    exog_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    params = {}
    params["beta"] = 0.95  # discount_factor
    params["rho"] = 1.95
    params["delta"] = 0.35

    options = {
        "state_space": {
            "endogenous_states": {
                "period": np.arange(2),
                "lagged_choice": [0, 1],
            },
            "choice": [0, 1],
        },
        "model_params": {"min_age": 50, "max_age": 80, "n_periods": 25, "n_choices": 2},
    }

    state_choice_vars = {"lagged_choice": 0, "choice": 0}

    options["state_space"]["exogenous_states"] = {"exog_state": [0]}

    return params, exog_savings_grid, state_choice_vars


@pytest.mark.parametrize("period", [2, 4, 9, 10, 18])
def test_fast_upper_envelope_wrapper(period, setup_model):
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/val{period}.csv",
        delimiter=",",
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/pol{period}.csv",
        delimiter=",",
    )
    value_refined_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/expec_val{period}.csv",
        delimiter=",",
    )
    policy_refined_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/expec_pol{period}.csv",
        delimiter=",",
    )
    policy_expected = policy_refined_fedor[
        :, ~np.isnan(policy_refined_fedor).any(axis=0)
    ]
    value_expected = value_refined_fedor[
        :,
        ~np.isnan(value_refined_fedor).any(axis=0),
    ]

    params, _exog_savings_grid, state_choice_vars = setup_model

    utility_kwargs = {
        "choice": state_choice_vars["choice"],
        "params": params,
    }
    n_constrained_points_to_add = int(0.1 * len(policy_egm[0]))
    n_final_wealth_grid = int(1.2 * (len(policy_egm[0]) + n_constrained_points_to_add))
    tuning_params = {
        "n_final_wealth_grid": n_final_wealth_grid,
        "jump_thresh": 2,
        "n_constrained_points_to_add": n_constrained_points_to_add,
    }
    (
        endog_grid_refined,
        policy_refined,
        value_refined,
    ) = fast_upper_envelope_wrapper(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        utility_function=utility_crra,
        utility_kwargs=utility_kwargs,
        disc_factor=params["beta"],
        tuning_params=tuning_params,
    )

    wealth_max_to_test = np.max(endog_grid_refined[~np.isnan(endog_grid_refined)]) + 100
    wealth_grid_to_test = jnp.linspace(
        endog_grid_refined[1], wealth_max_to_test, 1000, dtype=float
    )

    value_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=value_expected[0], y=value_expected[1]
    )

    policy_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=policy_expected[0], y=policy_expected[1]
    )

    (
        policy_calc_interp,
        value_calc_interp,
    ) = interpolate_policy_and_value_on_wealth_grid(
        wealth_beginning_of_period=wealth_grid_to_test,
        endog_wealth_grid=endog_grid_refined,
        policy=policy_refined,
        value_grid=value_refined,
    )

    aaae(value_calc_interp, value_expec_interp)
    aaae(policy_calc_interp, policy_expec_interp)


def test_fast_upper_envelope_against_numba(setup_model):
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / "upper_envelope_period_tests/pol10.csv", delimiter=","
    )
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / "upper_envelope_period_tests/val10.csv", delimiter=","
    )
    _params, exog_savings_grid, state_choice_vars = setup_model

    endog_grid_org, value_org, policy_org = fues_nb.fast_upper_envelope(
        endog_grid=policy_egm[0],
        value=value_egm[1],
        policy=policy_egm[1],
        exog_grid=np.append(0, exog_savings_grid),
    )
    n_constrained_points_to_add = int(0.1 * len(policy_egm[0]))
    n_final_wealth_grid = int(1.2 * (len(policy_egm[0]) + n_constrained_points_to_add))
    tuning_params = {
        "n_final_wealth_grid": n_final_wealth_grid,
        "jump_thresh": 2,
        "n_constrained_points_to_add": n_constrained_points_to_add,
    }

    (
        endog_grid_refined,
        value_refined,
        policy_refined,
    ) = fast_upper_envelope(
        endog_grid=policy_egm[0, 1:],
        value=value_egm[1, 1:],
        policy=policy_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        tuning_params=tuning_params,
    )

    wealth_max_to_test = np.max(endog_grid_refined[~np.isnan(endog_grid_refined)]) + 100
    wealth_grid_to_test = jnp.linspace(
        endog_grid_refined[1], wealth_max_to_test, 1000, dtype=float
    )

    (
        policy_calc_interp_calc,
        value_calc_interp_calc,
    ) = interpolate_policy_and_value_on_wealth_grid(
        wealth_beginning_of_period=wealth_grid_to_test,
        endog_wealth_grid=endog_grid_refined,
        policy=policy_refined,
        value_grid=value_refined,
    )

    (
        policy_calc_interp_org,
        value_calc_interp_org,
    ) = interpolate_policy_and_value_on_wealth_grid(
        wealth_beginning_of_period=wealth_grid_to_test,
        endog_wealth_grid=endog_grid_org,
        policy=policy_org,
        value_grid=value_org,
    )

    aaae(value_calc_interp_calc, value_calc_interp_org)
    aaae(policy_calc_interp_calc, policy_calc_interp_org)


@pytest.mark.parametrize("period", [2, 4, 10, 9, 18])
def test_fast_upper_envelope_against_fedor(period, setup_model):
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/val{period}.csv",
        delimiter=",",
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/pol{period}.csv",
        delimiter=",",
    )

    params, exog_savings_grid, state_choice_vec = setup_model

    _policy_fedor, _value_fedor = upper_envelope(
        policy=policy_egm,
        value=value_egm,
        exog_grid=exog_savings_grid,
        state_choice_vec={"choice": state_choice_vec["choice"]},
        params=params,
        compute_utility=utility_crra,
    )
    policy_expected = _policy_fedor[:, ~np.isnan(_policy_fedor).any(axis=0)]
    value_expected = _value_fedor[
        :,
        ~np.isnan(_value_fedor).any(axis=0),
    ]

    utility_kwargs = {
        "choice": state_choice_vec["choice"],
        "params": params,
    }
    n_constrained_points_to_add = int(0.1 * len(policy_egm[0]))
    n_final_wealth_grid = int(1.2 * (len(policy_egm[0]) + n_constrained_points_to_add))
    tuning_params = {
        "n_final_wealth_grid": n_final_wealth_grid,
        "jump_thresh": 2,
        "n_constrained_points_to_add": n_constrained_points_to_add,
    }

    (
        endog_grid_fues,
        policy_fues,
        value_fues,
    ) = fast_upper_envelope_wrapper(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        utility_function=utility_crra,
        utility_kwargs=utility_kwargs,
        disc_factor=params["beta"],
        tuning_params=tuning_params,
    )

    wealth_max_to_test = np.max(endog_grid_fues[~np.isnan(endog_grid_fues)]) + 100
    wealth_grid_to_test = jnp.linspace(endog_grid_fues[1], wealth_max_to_test, 1000)

    value_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=value_expected[0], y=value_expected[1]
    )
    policy_expec_interp = linear_interpolation_with_extrapolation(
        x_new=wealth_grid_to_test, x=policy_expected[0], y=policy_expected[1]
    )

    policy_interp, value_interp = interpolate_policy_and_value_on_wealth_grid(
        wealth_beginning_of_period=wealth_grid_to_test,
        endog_wealth_grid=endog_grid_fues,
        policy=policy_fues,
        value_grid=value_fues,
    )
    aaae(value_interp, value_expec_interp)
    aaae(policy_interp, policy_expec_interp)


def test_back_and_forward_scan_wrapper_direction_flag():
    msg = "Direction must be either 'forward' or 'backward'."

    with pytest.raises(ValueError, match=msg):
        back_and_forward_scan_wrapper(
            endog_grid_to_calculate_gradient=0.6,
            value_to_calculate_gradient=0.5,
            endog_grid_to_scan_from=1.2,
            policy_to_scan_from=0.7,
            endog_grid=1,
            value=np.arange(2, 5),
            policy=np.arange(1, 4),
            idx_to_scan_from=2,
            n_points_to_scan=3,
            is_scan_needed=False,
            jump_thresh=2,
            direction="Don't know",
        )
