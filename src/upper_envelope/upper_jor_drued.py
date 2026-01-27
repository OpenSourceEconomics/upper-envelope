from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["value_function"])
def upper_jor_drued(
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    m_grid: jnp.ndarray,
    expected_value_zero_savings: jnp.ndarray | float,
    value_function: Callable,
    value_function_args=(),
    value_function_kwargs: Optional[Dict] = None,
):
    """Compute a simple 1D upper envelope on a given common grid.

    The envelope is computed by linearly interpolating every adjacent pair
    ``(endog_grid[i], endog_grid[i+1])`` onto the common grid ``m_grid``.
    For each point on ``m_grid``, we take the pointwise maximum over all segment
    interpolants and an additional "consume-all" candidate.

    This function intentionally does *not*:
    - sort inputs
    - detect or insert intersection points
    - apply FUES jump/scan logic

    Returns arrays with the convention that index 0 corresponds to zero wealth:
    ``value_out[0] = expected_value_zero_savings`` and ``endog_out[0] = policy_out[0] = 0``.
    """

    if value_function_kwargs is None:
        value_function_kwargs = {}

    # Segment interpolation weights for each adjacent pair.
    dm = endog_grid[1:] - endog_grid[:-1]  # (N-1,)
    eps = 1e-16
    weight = (m_grid[None, :] - endog_grid[:-1, None]) / (dm[:, None] + eps)  # (N-1, M)

    c_interp = policy[:-1, None] + weight * (policy[1:] - policy[:-1])[:, None]
    v_interp = value[:-1, None] + weight * (value[1:] - value[:-1])[:, None]

    outside = (weight < 0.0) | (weight > 1.0)
    v_interp = jnp.where(outside, -jnp.inf, v_interp)

    # Consume-all candidate.
    c_all = m_grid
    v_all = value_function(c_all, *value_function_args, **value_function_kwargs)

    v_stack = jnp.vstack((v_interp, v_all[None, :]))
    c_stack = jnp.vstack((c_interp, c_all[None, :]))

    best = jnp.argmax(v_stack, axis=0)
    grid_idx = jnp.arange(m_grid.size)

    value_best = v_stack[best, grid_idx]
    policy_best = c_stack[best, grid_idx]

    # Prepend zero-wealth convention.
    endog_out = jnp.concatenate((jnp.array([0.0]), m_grid))
    policy_out = jnp.concatenate((jnp.array([0.0]), policy_best))
    value_out = jnp.concatenate((jnp.array([expected_value_zero_savings]), value_best))

    return endog_out, policy_out, value_out
