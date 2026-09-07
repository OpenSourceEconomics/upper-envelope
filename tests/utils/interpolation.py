from typing import Tuple

import jax.numpy as jnp
import numpy as np


def interpolate_policy_and_value_on_wealth_grid(
    wealth_beginning_of_period: np.ndarray | jnp.ndarray,
    endog_wealth_grid: np.ndarray | jnp.ndarray,
    policy_grid: np.ndarray | jnp.ndarray,
    value_function_grid: np.ndarray | jnp.ndarray,
):
    """Interpolate policy and value functions on the wealth grid.

    Args:
        wealth_beginning_of_period (np.ndarray): 1d array of shape (n,) containing the
            begin of period wealth.
        endog_wealth_grid (np.array): 1d array of shape (n,) containing the endogenous
            wealth grid.
        policy_grid (np.ndarray): 1d array of shape (n,) containing the
            policy function corresponding to the endogenous wealth grid.
        value_function_grid (np.ndarray): 1d array of shape (n,) containing the value
            function corresponding to the endogenous wealth grid.

    Returns:
        tuple:

        - policy_new (np.ndarray): 1d array of shape (n,) containing the interpolated
            policy function values corresponding to the begin of period wealth.
        - value_new (np.ndarray): 1d array of shape (n,) containing the interpolated
            value function values corresponding to the begin of period wealth.

    """
    # Make sure that these are numpy arrays
    wealth_beginning_of_period = np.asarray(wealth_beginning_of_period)
    endog_wealth_grid = np.asarray(endog_wealth_grid)
    policy_grid = np.asarray(policy_grid)
    value_function_grid = np.asarray(value_function_grid)
    ind_high, ind_low = get_index_high_and_low(
        x=endog_wealth_grid, x_new=wealth_beginning_of_period
    )

    wealth_low = np.take(endog_wealth_grid, ind_low)
    wealth_high = np.take(endog_wealth_grid, ind_high)

    policy_new = linear_interpolation_formula(
        y_high=np.take(policy_grid, ind_high),
        y_low=np.take(policy_grid, ind_low),
        x_high=wealth_high,
        x_low=wealth_low,
        x_new=wealth_beginning_of_period,
    )

    value_new = linear_interpolation_formula(
        y_high=np.take(value_function_grid, ind_high),
        y_low=np.take(value_function_grid, ind_low),
        x_high=wealth_high,
        x_low=wealth_low,
        x_new=wealth_beginning_of_period,
    )

    return policy_new, value_new


def interpolate_single_policy_and_value_on_wealth_grid(
    wealth_beginning_of_period: np.ndarray | jnp.ndarray,
    endog_wealth_grid: np.ndarray | jnp.ndarray,
    policy_grid: np.ndarray | jnp.ndarray,
    value_function_grid: np.ndarray | jnp.ndarray,
):
    """Interpolate policy and value functions on the wealth grid.

    This function uses the left and right policy function.
    For a more detailed description, see calc_intersection_and_extrapolate_policy
    in fast_upper_envelope.py.

    Args:
        wealth_beginning_of_period (np.ndarray): 1d array of shape (n,) containing the
            begin of period wealth.
        endog_wealth_grid (np.ndarray): 1d array of shape (n,) containing the endogenous
            wealth grid.
        policy_grid (np.ndarray): 1d array of shape (n,) containing the
            policy function corresponding to the endogenous wealth grid.
        value_function_grid (np.ndarray): 1d array of shape (n,) containing the value
            function corresponding to the endogenous wealth grid.

    Returns:
        tuple:

        - policy_new (np.ndarray): 1d array of shape (n,) containing the interpolated
            policy function values corresponding to the begin of period wealth.
        - value_new (np.ndarray): 1d array of shape (n,) containing the interpolated
            value function values corresponding to the begin of period wealth.

    """
    # Make sure that these are numpy arrays
    wealth_beginning_of_period = np.asarray(wealth_beginning_of_period)
    endog_wealth_grid = np.asarray(endog_wealth_grid)
    policy_grid = np.asarray(policy_grid)
    value_function_grid = np.asarray(value_function_grid)
    ind_high, ind_low = get_index_high_and_low(
        x=endog_wealth_grid, x_new=wealth_beginning_of_period
    )

    wealth_low = np.take(endog_wealth_grid, ind_low)
    wealth_high = np.take(endog_wealth_grid, ind_high)

    policy_new = linear_interpolation_formula(
        y_high=np.take(policy_grid, ind_high),
        y_low=np.take(policy_grid, ind_low),
        x_high=wealth_high,
        x_low=wealth_low,
        x_new=wealth_beginning_of_period,
    )

    value_new = linear_interpolation_formula(
        y_high=np.take(value_function_grid, ind_high),
        y_low=np.take(value_function_grid, ind_low),
        x_high=wealth_high,
        x_low=wealth_low,
        x_new=wealth_beginning_of_period,
    )

    return policy_new, value_new


def linear_interpolation_formula(
    y_high: float | np.ndarray,
    y_low: float | np.ndarray,
    x_high: float | np.ndarray,
    x_low: float | np.ndarray,
    x_new: float | np.ndarray,
):
    """Linear interpolation formula."""
    interpolate_dist = x_new - x_low
    interpolate_slope = (y_high - y_low) / (x_high - x_low)
    interpol_res = (interpolate_slope * interpolate_dist) + y_low

    return interpol_res


def get_index_high_and_low(
    x: np.ndarray, x_new: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Get index of the highest value in x that is smaller than x_new.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        x_new (float): The new x-value at which to evaluate the interpolation function.

    Returns:
        int: Index of the value in the wealth grid which is higher than x_new. Or in
            case of extrapolation last or first index of not nan element.

    """
    ind_high = np.searchsorted(x, x_new).clip(max=(x.shape[0] - 1), min=1)
    ind_high -= np.isnan(x[ind_high]).astype(int)
    return ind_high, ind_high - 1


# =====================================================================================
# Numpy utils
# =====================================================================================


def linear_interpolation_with_extrapolation(x, y, x_new) -> np.ndarray:
    """Linear interpolation with extrapolation.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (np.ndarray or float): 1d array of shape (m,) or float containing
            the new x-values at which to evaluate the interpolation function.

    Returns:
        np.ndarray or float: 1d array of shape (m,) or float containing
            the new y-values corresponding to the new x-values.
            In case x_new contains values outside of the range of x, these
            values are extrapolated.

    """
    # make sure that the function also works for unsorted x-arrays
    # taken from scipy.interpolate.interp1d
    ind = np.argsort(x, kind="mergesort")
    x = x[ind]
    y = np.take(y, ind)

    ind_high = np.searchsorted(x, x_new).clip(max=(x.shape[0] - 1), min=1)
    ind_low = ind_high - 1

    y_high = y[ind_high]
    y_low = y[ind_low]
    x_high = x[ind_high]
    x_low = x[ind_low]

    interpolate_dist = x_new - x_low
    interpolate_slope = (y_high - y_low) / (x_high - x_low)
    interpol_res = (interpolate_slope * interpolate_dist) + y_low

    return interpol_res


def linear_interpolation_with_inserting_missing_values(x, y, x_new, missing_value):
    """Linear interpolation with inserting missing values.

    Args:
        x (np.ndarray): 1d array of shape (n,) containing the x-values.
        y (np.ndarray): 1d array of shape (n,) containing the y-values
            corresponding to the x-values.
        x_new (np.ndarray or float): 1d array of shape (m,) or float containing
            the new x-values at which to evaluate the interpolation function.
        missing_value (np.ndarray or float): Flat array of shape (1,) or float
            to set for values of x_new outside of the range of x.

    Returns:
        np.ndarray or float: 1d array of shape (m,) or float containing the
            new y-values corresponding to the new x-values.
            In case x_new contains values outside of the range of x, these
            values are set equal to missing_value.

    """
    interpol_res = linear_interpolation_with_extrapolation(x, y, x_new)
    where_to_miss = (x_new < x.min()) | (x_new > x.max())
    interpol_res[where_to_miss] = missing_value

    return interpol_res
