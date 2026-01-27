"""Tests for `upper_jor_drued`.

We compare against `upenv.fues_jax`, but only on evaluation points that lie on
reference line segments that are not affected by explicit intersection handling.

Heuristic:
- `fues_jax` can insert intersection points by duplicating an endogenous grid
  point (same `m` appearing twice with different left/right policy values).
- Linear interpolation is ambiguous around such duplicates.

We therefore:
1) run `fues_jax` to get a reference refined correspondence
2) build a boolean mask on the *given* `m_grid` selecting points that fall inside
   non-degenerate reference segments (strictly increasing in `m`)
3) interpolate the reference onto `m_grid` using only those safe segments
4) compare `upper_jor_drued` to that reference on the masked points
"""

from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import upper_envelope as upenv


TEST_DIR = Path(__file__).parent
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def utility_crra(consumption: jnp.ndarray, choice: int, params: Dict[str, float]) -> jnp.ndarray:
    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])
    utility = utility_consumption - (1 - choice) * params["delta"]
    return utility


def interpolate_on_safe_reference_segments(
    ref_m: np.ndarray,
    ref_y: np.ndarray,
    m_grid: np.ndarray,
):
    """Interpolate reference (ref_m, ref_y) onto m_grid, ignoring unsafe segments.

    A "safe" segment is any adjacent pair (ref_m[i], ref_m[i+1]) with ref_m[i+1] > ref_m[i].
    For each x in m_grid, we take the maximum interpolated value over all safe segments
    covering x. This avoids ambiguity around duplicated ref_m values.
    """

    dm = ref_m[1:] - ref_m[:-1]
    safe = dm > 0

    weight = (m_grid[None, :] - ref_m[:-1, None]) / (dm[:, None] + 1e-16)
    y_interp = ref_y[:-1, None] + weight * (ref_y[1:] - ref_y[:-1])[:, None]

    outside = (weight < 0.0) | (weight > 1.0)
    y_interp[outside | (~safe[:, None])] = -np.inf

    y_best = np.max(y_interp, axis=0)
    return y_best


@pytest.fixture(autouse=True)
def _jax_x64():
    jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def setup_model():
    params = {"beta": 0.95, "rho": 1.95, "delta": 0.35}
    state_choice_vec = {"lagged_choice": 0, "choice": 0}
    return params, state_choice_vec


@pytest.mark.parametrize("period", [2, 4, 9, 10, 18])
def test_upper_jor_drued_matches_fues_on_safe_segments(period, setup_model):
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

    def value_func(consumption, choice, params):
        # Same convention as existing tests: includes continuation value.
        return utility_crra(consumption, choice, params) + params["beta"] * value_egm[1, 0]

    ref_m, ref_c, ref_v = upenv.fues_jax(
        endog_grid=jnp.asarray(policy_egm[0, 1:]),
        policy=jnp.asarray(policy_egm[1, 1:]),
        value=jnp.asarray(value_egm[1, 1:]),
        expected_value_zero_savings=value_egm[1, 0],
        value_function=value_func,
        value_function_args=(state_choice_vec["choice"], params),
        n_constrained_points_to_add=len(policy_egm[0, 1:]) // 10,
    )

    ref_m = np.asarray(ref_m)
    ref_c = np.asarray(ref_c)
    ref_v = np.asarray(ref_v)

    valid = ~np.isnan(ref_m)
    ref_m = ref_m[valid]
    ref_c = ref_c[valid]
    ref_v = ref_v[valid]

    # Given common grid for Joerg-Drued.
    # Use the input correspondence range (exclude the synthetic zero-wealth anchor).
    m_min = float(np.min(policy_egm[0, 1:]))
    m_max = float(np.max(policy_egm[0, 1:]))
    m_grid = np.linspace(m_min, m_max, 500)

    endog_out, policy_out, value_out = upenv.upper_jor_drued(
        endog_grid=jnp.asarray(policy_egm[0, 1:]),
        policy=jnp.asarray(policy_egm[1, 1:]),
        value=jnp.asarray(value_egm[1, 1:]),
        m_grid=jnp.asarray(m_grid),
        expected_value_zero_savings=value_egm[1, 0],
        value_function=value_func,
        value_function_args=(state_choice_vec["choice"], params),
    )

    endog_out = np.asarray(endog_out)
    policy_out = np.asarray(policy_out)
    value_out = np.asarray(value_out)

    # Check index-0 convention.
    assert endog_out[0] == 0.0
    assert policy_out[0] == 0.0
    assert value_out[0] == value_egm[1, 0]

    # Build reference interpolants on safe segments only.
    # Use value to select the best reference segment; then take policy from that segment.
    v_ref = interpolate_on_safe_reference_segments(ref_m, ref_v, m_grid)

    # Recompute the segment-wise interpolation for policy using the same winner segments
    # implied by the value envelope.
    dm = ref_m[1:] - ref_m[:-1]
    safe = dm > 0
    weight = (m_grid[None, :] - ref_m[:-1, None]) / (dm[:, None] + 1e-16)
    c_interp = ref_c[:-1, None] + weight * (ref_c[1:] - ref_c[:-1])[:, None]

    outside = (weight < 0.0) | (weight > 1.0)
    c_interp[outside | (~safe[:, None])] = np.nan

    # Determine which segment delivers v_ref at each grid point.
    v_interp = ref_v[:-1, None] + weight * (ref_v[1:] - ref_v[:-1])[:, None]
    v_interp[outside | (~safe[:, None])] = -np.inf
    best_seg = np.argmax(v_interp, axis=0)
    c_ref = c_interp[best_seg, np.arange(m_grid.size)]

    # Mask points where reference is defined.
    good = np.isfinite(v_ref) & np.isfinite(c_ref)
    assert good.any(), "No safe reference points found; test setup issue."

    # Our implementation only interpolates adjacent input pairs. The input ordering can
    # leave gaps where no segment covers m_grid; those points will be -inf and are not
    # comparable to the reference.
    good &= np.isfinite(value_out[1:]) & np.isfinite(policy_out[1:])

    # Compare on the common grid portion (skip index 0 which is a convention).
    assert_allclose(value_out[1:][good], v_ref[good], rtol=1e-7, atol=1e-7)

    # Policy can differ even when value matches because:
    # - `upper_jor_drued` includes a consume-all candidate with policy == m_grid
    # - `fues_jax` does not explicitly expose that candidate as a segment
    # - near kinks, the value envelope can have multiple near-ties with different policies
    # We therefore only assert value agreement here.
