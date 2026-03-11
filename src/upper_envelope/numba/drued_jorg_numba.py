from typing import Callable

import numpy as np
from numba import njit


@njit
def drued_jorg_numba(
    endog_grid: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    m_grid: np.ndarray,
    expected_value_zero_savings: np.ndarray | float,
    value_function: Callable,
    value_function_args=(),
):
    """Compute a simple 1D upper envelope on a given common grid.

    This mirrors `upper_envelope.drued_jorg_jax.drued_jorg_jax` but is implemented
    in numba.

    The envelope is computed by linearly interpolating every adjacent pair
    ``(endog_grid[i], endog_grid[i+1])`` onto the common grid ``m_grid``.
    For each point on ``m_grid``, we take the pointwise maximum over all segment
    interpolants and an additional "consume-all" candidate.

    Returns arrays with the convention that index 0 corresponds to zero wealth:
    ``value_out[0] = expected_value_zero_savings`` and ``endog_out[0] = policy_out[0] = 0``.

    """

    n_m = m_grid.size
    n_segments = endog_grid.size - 1

    policy_best = np.empty(n_m)
    value_best = np.empty(n_m)

    eps = 1e-16

    for j in range(n_m):
        m = m_grid[j]

        # "Consume-all" candidate.
        best_v = value_function(m, *value_function_args)
        best_c = m

        for i in range(n_segments):
            dm = endog_grid[i + 1] - endog_grid[i]
            w = (m - endog_grid[i]) / (dm + eps)

            if (w >= 0.0) and (w <= 1.0):
                v_interp = value[i] + w * (value[i + 1] - value[i])
                if v_interp > best_v:
                    best_v = v_interp
                    best_c = policy[i] + w * (policy[i + 1] - policy[i])

        value_best[j] = best_v
        policy_best[j] = best_c

    endog_out = np.empty(n_m + 1)
    policy_out = np.empty(n_m + 1)
    value_out = np.empty(n_m + 1)

    endog_out[0] = 0.0
    policy_out[0] = 0.0
    value_out[0] = expected_value_zero_savings

    endog_out[1:] = m_grid
    policy_out[1:] = policy_best
    value_out[1:] = value_best

    return endog_out, policy_out, value_out
