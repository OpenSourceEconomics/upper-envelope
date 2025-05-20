"""Jax implementation of the extended Fast Upper-Envelope Scan (FUES).

The original FUES algorithm is based on Loretti I. Dobrescu and Akshay Shanker (2022)
'Fast Upper-Envelope Scan for Solving Dynamic Optimization Problems',
https://dx.doi.org/10.2139/ssrn.4181302

"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from upper_envelope.fues_jax.check_and_scan_funcs import (
    determine_cases_and_conduct_necessary_scans,
)
from upper_envelope.math_funcs import calc_intersection_and_extrapolate_policy


@partial(
    jax.jit,
    static_argnames=[
        "value_function",
        "n_constrained_points_to_add",
        "n_final_wealth_grid",
        "jump_thresh",
        "n_points_to_scan",
    ],
)
def fues_jax(
    endog_grid: jnp.ndarray,
    policy: jnp.ndarray,
    value: jnp.ndarray,
    expected_value_zero_savings: jnp.ndarray | float,
    value_function: Callable,
    value_function_args: Optional[Tuple] = (),
    value_function_kwargs: Optional[Dict] = {},
    n_constrained_points_to_add=None,
    n_final_wealth_grid=None,
    jump_thresh=2,
    n_points_to_scan=10,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Drop suboptimal points and refines the endogenous grid, policy, and value.

    Computes the upper envelope over the overlapping segments of the
    decision-specific value functions, which in fact are value "correspondences"
    in this case, where multiple solutions are detected. The dominated grid
    points are then eliminated from the endogenous wealth grid.
    Discrete choices introduce kinks and non-concave regions in the value
    function that lead to discontinuities in the policy function of the
    continuous (consumption) choice. In particular, the value function has a
    non-concave region where the decision-specific values of the
    alternative discrete choices (e.g. continued work or retirement) cross.
    These are referred to as "primary" kinks.
    As a result, multiple local optima for consumption emerge and the Euler
    equation has multiple solutions.
    Moreover, these "primary" kinks propagate back in time and manifest
    themselves in an accumulation of "secondary" kinks in the choice-specific
    value functions in earlier time periods, which, in turn, also produce an
    increasing number of discontinuities in the consumption functions
    in earlier periods of the life cycle.
    These discontinuities in consumption rules in period t are caused by the
    worker's anticipation of landing exactly at the kink points in the
    subsequent periods t + 1, t + 2, ..., T under the optimal consumption policy.

    Args:
        endog_grid (jnp.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific endogenous grid.
        policy (jnp.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific policy function.
        value (jnp.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific value function.
        expected_value_zero_savings (jnp.ndarray | float): The agent's expected value
            given that she saves zero.
        value_function (callable): The value function for calculating the value if
            nothing is saved.
        value_function_args (Tuple): The positional arguments to be passed to the value
            function.
        value_function_kwargs (dict): The keyword arguments to be passed to the value
            function.
        n_constrained_points_to_add (int): Number of constrained points to add to the
            left of the first grid point if there is an area with credit-constrain.
        n_final_wealth_grid (int): Size of final function grid. Determines number of
            iterations for the scan in the fues_jax.
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        tuple:

        - endog_grid_refined (jnp.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing the refined endogenous wealth grid.
        - policy_refined (jnp.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing refined consumption policy.
        - value_refined (jnp.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing refined value function.

    """
    # Set default of n_constrained_points_to_add to 10% of the grid size
    n_constrained_points_to_add = (
        endog_grid.shape[0] // 10
        if n_constrained_points_to_add is None
        else n_constrained_points_to_add
    )

    # Set default value of final grid size to 1.2 times current if not defined
    n_final_wealth_grid = (
        int(1.2 * endog_grid.shape[0])
        if n_final_wealth_grid is None
        else n_final_wealth_grid
    )

    # Check if a non-concave region coincides with the credit constrained region.
    # This happens when there is a non-monotonicity in the endogenous wealth grid
    # that goes below the first point (the minimal wealth, below it is optimal to
    # consume everything).

    # If there is such a non-concave region, we extend the value function to the left
    # of the first point and calculate the value function there with the supplied value
    # function.

    # Because of jax, we always need to perform the same set of computations. Hence,
    # if there is no wealth grid point below the first, we just add nans thereafter.
    min_id = np.argmin(endog_grid)
    min_wealth_grid = endog_grid[min_id]

    # This is the condition, which we do not use at the moment.
    # closed_form_cond = min_wealth_grid < endog_grid[0]
    grid_points_to_add = jnp.linspace(
        min_wealth_grid, endog_grid[0], n_constrained_points_to_add + 1
    )[:-1]
    # Compute closed form values
    values_to_add = vmap(_compute_value, in_axes=(0, None, None, None))(
        grid_points_to_add, value_function, value_function_args, value_function_kwargs
    )

    # Now determine if we actually had to extend the grid.
    # If not, we just add nans.
    no_need_to_add = min_id == 0
    multiplikator = jax.lax.select(no_need_to_add, jnp.nan, 1.0)
    grid_points_to_add *= multiplikator
    values_to_add *= multiplikator

    grid_augmented = jnp.append(grid_points_to_add, endog_grid)
    value_augmented = jnp.append(values_to_add, value)
    policy_augmented = jnp.append(grid_points_to_add, policy)

    (
        endog_grid_refined,
        value_refined,
        policy_refined,
    ) = fues_jax_unconstrained(
        grid_augmented,
        value_augmented,
        policy_augmented,
        expected_value_zero_savings,
        n_final_wealth_grid=n_final_wealth_grid,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
    )
    return (
        endog_grid_refined,
        policy_refined,
        value_refined,
    )


@partial(
    jax.jit, static_argnames=["n_final_wealth_grid", "jump_thresh", "n_points_to_scan"]
)
def fues_jax_unconstrained(
    endog_grid: jnp.ndarray,
    value: jnp.ndarray,
    policy: jnp.ndarray,
    expected_value_zero_savings: float,
    n_final_wealth_grid=None,
    jump_thresh=2,
    n_points_to_scan=10,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Remove suboptimal points from the endogenous grid, policy, and value function.

    Args:
        endog_grid (jnp.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific endogenous grid.
        policy (jnp.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific policy function.
        value (jnp.ndarray): 1d array of shape (n_grid_wealth + 1,)
            containing the current state- and choice-specific value function.
        expected_value_zero_savings (jnp.ndarray | float): The agent's expected value
            given that she saves zero.
        n_final_wealth_grid (int): Size of final function grid. Determines number of
            iterations for the scan in the fues_jax.
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        tuple:

        - endog_grid_refined (jnp.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing the refined endogenous wealth grid.
        - policy_refined (jnp.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing refined consumption policy.
        - value_refined (jnp.ndarray): 1d array of shape (n_final_wealth_grid,)
            containing refined value function.

    """
    # Set default value of final grid size to 1.2 times current if not defined
    n_final_wealth_grid = (
        int(1.2 * endog_grid.shape[0])
        if n_final_wealth_grid is None
        else n_final_wealth_grid
    )

    idx_sort = jnp.argsort(endog_grid)
    value = jnp.take(value, idx_sort)
    policy = jnp.take(policy, idx_sort)
    endog_grid = jnp.take(endog_grid, idx_sort)

    (
        value_refined,
        policy_refined,
        endog_grid_refined,
    ) = scan_value_function(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        expected_value_zero_savings=expected_value_zero_savings,
        n_final_wealth_grid=n_final_wealth_grid,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
    )

    return endog_grid_refined, value_refined, policy_refined


def scan_value_function(
    endog_grid: jnp.ndarray,
    value: jnp.ndarray,
    policy: jnp.ndarray,
    expected_value_zero_savings,
    n_final_wealth_grid: int,
    jump_thresh: float,
    n_points_to_scan: Optional[int] = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan the value function to remove suboptimal points and add intersection points.

    Args:
        endog_grid (np.ndarray): 1d array containing the unrefined endogenous wealth
            grid of shape (n_grid_wealth + 1,).
        value (np.ndarray): 1d array containing the unrefined value correspondence
            of shape (n_grid_wealth + 1,).
        policy (np.ndarray): 1d array containing the unrefined policy correspondence
            of shape (n_grid_wealth + 1,).
        expected_value_zero_savings (float): The agent's expected value given that she
            saves zero.
        n_final_wealth_grid (int): Size of final grid. Determines number of
            iterations for the scan in the fues_jax.
        jump_thresh (float): Jump detection threshold.
        n_points_to_scan (int): Number of points to scan for suboptimal points.

    Returns:
        tuple:

        - (np.ndarray): 1d array of shape (n_grid_clean,) containing the refined
            value function. Overlapping segments have been removed and only
            the optimal points are kept.

    """
    value_k_and_j = expected_value_zero_savings, value[0]
    endog_grid_k_and_j = 0, endog_grid[0]
    policy_k_and_j = 0, policy[0]
    vars_j_and_k_inital = (value_k_and_j, policy_k_and_j, endog_grid_k_and_j)

    to_be_saved_inital = (expected_value_zero_savings, 0.0, 0.0)
    last_point_in_grid = jnp.array([value[-1], policy[-1], endog_grid[-1]])
    dummy_points_grid = jnp.array([jnp.nan, jnp.nan, jnp.nan])

    idx_to_inspect = 0
    idx_to_be_saved_next = 0
    # These get update before assigned to be saved next.
    idx_result_array_valid = 0
    idx_result_array_invalid = 0
    last_point_was_intersect = jnp.array(False)
    second_last_point_was_intersect = jnp.array(False)
    saved_last_point_already = jnp.array(False)

    carry_init = (
        vars_j_and_k_inital,
        to_be_saved_inital,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
        second_last_point_was_intersect,
        idx_result_array_valid,
        idx_result_array_invalid,
        idx_to_be_saved_next,
    )
    partial_body = partial(
        scan_body,
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        last_point_in_grid=last_point_in_grid,
        dummy_points_grid=dummy_points_grid,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
    )

    _final_carry, result = jax.lax.scan(
        partial_body,
        carry_init,
        xs=None,
        length=n_final_wealth_grid,
    )
    result_arrays, sort_index = result
    value, policy, endog_grid = result_arrays
    value = value.at[sort_index].set(value)
    policy = policy.at[sort_index].set(policy)
    endog_grid = endog_grid.at[sort_index].set(endog_grid)
    return value, policy, endog_grid


def scan_body(
    carry,
    _iter_step,
    value,
    policy,
    endog_grid,
    last_point_in_grid,
    dummy_points_grid,
    jump_thresh,
    n_points_to_scan,
):
    """The scan body to be executed at each iteration of the scan function.

    Depending on the idx_to_inspect of the carry value, either a new value is scanned
    or the value from the last period is saved.
    The carry value is updated in each iteration and passed to the next iteration.
    This scan body returns value, policy and endogenous wealth grid.

        Args:
            carry (tuple): The carry value passed from the previous iteration. This is a
                tuple containing the variables that are updated in each iteration.
                Including the current two optimal points j and k, the points to be saved
                this iteration as well as the indexes of the point to be inspected, the
                indicator of case 2 and the indicator if the last point was an
                intersection point.
            _iter_step (int): The current iteration number.
            value (jnp.ndarray): 1d array containing the unrefined value correspondence
                of shape (n_grid_wealth,).
            policy (jnp.ndarray): 1d array containing the unrefined policy
                correspondence of shape (n_grid_wealth,).
            endog_grid (jnp.ndarray): 1d array containing the unrefined endogenous
                wealth grid of shape (n_grid_wealth,).
            jump_thresh (float): Jump detection threshold.
            n_points_to_scan (int): Number of points to scan in the forward and backward
                scan.

    Returns:
        tuple:

        - carry (tuple): The updated carry value passed to the next iteration.
        - result (tuple): The result of this iteration. This is a tuple containing four
            elements to be saved in this iteration:
            - value
            - policy
            - endogenous grid point

    """
    (
        points_j_and_k,
        planned_to_be_saved_this_iter,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
        second_last_point_was_intersect,
        idx_result_array_valid,
        idx_result_array_invalid,
        idx_to_be_saved_this_period,
    ) = carry

    point_to_inspect = (
        value[idx_to_inspect],
        policy[idx_to_inspect],
        endog_grid[idx_to_inspect],
    )

    is_final_point_on_grid = idx_to_inspect == len(endog_grid) - 1
    end_of_valid_points = jnp.isnan(value[idx_to_inspect])
    is_final_point_on_grid = is_final_point_on_grid | end_of_valid_points

    (
        cases,
        update_idx,
        idx_next_on_lower_curve,
        idx_before_on_upper_curve,
    ) = determine_cases_and_conduct_necessary_scans(
        point_to_inspect=point_to_inspect,
        points_j_and_k=points_j_and_k,
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        idx_to_scan_from=idx_to_inspect,
        n_points_to_scan=n_points_to_scan,
        last_point_was_intersect=last_point_was_intersect,
        second_last_point_was_intersect=second_last_point_was_intersect,
        is_final_point_on_grid=is_final_point_on_grid,
        jump_thresh=jump_thresh,
    )

    intersection_point = select_and_calculate_intersection(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        points_j_and_k=points_j_and_k,
        idx_next_on_lower_curve=idx_next_on_lower_curve,
        idx_before_on_upper_curve=idx_before_on_upper_curve,
        idx_to_inspect=idx_to_inspect,
        case_5=cases[5],
        case_6=cases[6],
    )

    # Select the values we want to save this iteration
    point_to_save_this_iteration = select_point_to_save_this_iteration(
        intersection_point=intersection_point,
        planned_to_be_saved_this_iter=planned_to_be_saved_this_iter,
        case_6=cases[6],
    )

    point_case_2 = jax.lax.select(
        saved_last_point_already, dummy_points_grid, last_point_in_grid
    )

    point_to_be_saved_next_iteration = select_points_to_be_saved_next_iteration(
        point_to_inspect=point_to_inspect,
        point_case_2=point_case_2,
        intersection_point=intersection_point,
        points_j_and_k=points_j_and_k,
        planned_to_be_saved_this_iter=planned_to_be_saved_this_iter,
        cases=cases,
    )

    points_j_and_k = update_values_j_and_k(
        point_to_inspect=point_to_inspect,
        intersection_point=intersection_point,
        points_j_and_k=points_j_and_k,
        cases=cases,
    )

    (
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
        second_last_point_was_intersect,
        idx_to_save_next,
        idx_result_array_valid,
        idx_result_array_invalid,
    ) = update_bools_and_idx(
        idx_to_inspect=idx_to_inspect,
        update_idx=update_idx,
        idx_result_array_valid=idx_result_array_valid,
        idx_result_array_invalid=idx_result_array_invalid,
        cases=cases,
    )

    carry = (
        points_j_and_k,
        point_to_be_saved_next_iteration,
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
        second_last_point_was_intersect,
        idx_result_array_valid,
        idx_result_array_invalid,
        idx_to_save_next,
    )

    return carry, (point_to_save_this_iteration, idx_to_be_saved_this_period)


def update_bools_and_idx(
    idx_to_inspect, update_idx, idx_result_array_valid, idx_result_array_invalid, cases
):
    """Update indicators and index of the point to be inspected in the next period.

    The indicators are booleans that capture cases where
    - we have saved the last point we checked already, and
    - the last point we inspected is an intersection point.

    Args:
        idx_to_inspect (int): Index of the point to be inspected in the current
            iteration.
        update_idx (bool): Indicator if the index should be updated.
        cases (tuple): Tuple containing the indicators for the different cases.
    Returns:
        tuple:

        - idx_to_inspect (int): Index of the point to inspect in the next
            iteration.
        - saved_last_point_already (bool): Indicator if we have saved the previous
            point already.
        - last_point_was_intersect (bool): Indicator if the most recent point was an
            intersection point.

    """
    case_0, case_1, case_2, case_3, case_4, case_5, case_6 = cases
    idx_to_inspect += update_idx

    idx_result_array_valid += ~case_3
    idx_result_array_invalid -= case_3
    idx_to_save_next = idx_result_array_invalid * case_3 + idx_result_array_valid * (
        1 - case_3
    )

    # In the iteration where case_2 is True for the first time, the last point we
    # checked is selected and afterwards only nans are added.
    saved_last_point_already = case_2
    last_point_was_intersect = case_5 | case_6 | case_0
    second_last_point_was_intersect = case_5

    return (
        idx_to_inspect,
        saved_last_point_already,
        last_point_was_intersect,
        second_last_point_was_intersect,
        idx_to_save_next,
        idx_result_array_valid,
        idx_result_array_invalid,
    )


def update_values_j_and_k(point_to_inspect, intersection_point, points_j_and_k, cases):
    """Update point j and k, i.e. the two most recent points on the upper envelope.

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy, and
            endogenous grid point of to the point to be inspected.
        intersection_point (tuple): Tuple containing the value, policy, and endogenous
            grid points of the intersection point.
        points_j_and_k (tuple): Tuple containing the value, policy, and endogenous grid
            points of the most recent point on the upper envelope (j) and the
            point before (k).
        cases (tuple): Tuple containing the indicators for the different cases.

    Returns:
        tuple:

        - points_j_and_k (tuple): Tuple containing the updated value, policy, and
            endogenous grid point of the most recent point on the upper envelope (j)
            and the point before (k).

    """
    (
        intersect_grid,
        intersect_value,
        _intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point

    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    case_0, case_1, case_2, case_3, case_4, case_5, case_6 = cases
    in_case_123 = case_1 | case_2 | case_3
    in_case_01236 = case_0 | case_1 | case_2 | case_3 | case_6
    in_case_45 = case_4 | case_5

    # In case 1, 2, 3 the old value remains as value_j, in 4, 5, value_j is former
    # value k and in 6 the old value_j is overwritten

    # Value function update
    value_j_new = (
        case_0 * value_to_inspect
        + in_case_123 * value_k_and_j[1]
        + case_4 * value_to_inspect
        + case_5 * intersect_value
        + case_6 * intersect_value
    )
    value_k_new = in_case_01236 * value_k_and_j[0] + in_case_45 * value_k_and_j[1]
    value_k_and_j = value_k_new, value_j_new

    # Policy function update
    policy_j_new = (
        case_0 * policy_to_inspect
        + in_case_123 * policy_k_and_j[1]
        + case_4 * policy_to_inspect
        + case_5 * intersect_policy_right
        + case_6 * intersect_policy_right
    )
    policy_k_new = in_case_01236 * policy_k_and_j[0] + in_case_45 * policy_k_and_j[1]
    policy_k_and_j = policy_k_new, policy_j_new

    # Endog grid update
    endog_grid_j_new = (
        case_0 * endog_grid_to_inspect
        + in_case_123 * endog_grid_k_and_j[1]
        + case_4 * endog_grid_to_inspect
        + case_5 * intersect_grid
        + case_6 * intersect_grid
    )
    endog_grid_k_new = (
        in_case_01236 * endog_grid_k_and_j[0] + in_case_45 * endog_grid_k_and_j[1]
    )
    endog_grid_k_and_j = endog_grid_k_new, endog_grid_j_new

    return value_k_and_j, policy_k_and_j, endog_grid_k_and_j


def select_points_to_be_saved_next_iteration(
    point_to_inspect,
    point_case_2,
    points_j_and_k,
    intersection_point,
    planned_to_be_saved_this_iter,
    cases,
):
    """Select the points to be saved in the next iteration, depending on the case.

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy, and endogenous
            grid points of the point to be inspected.
        point_case_2 (jnp.ndarray): Tuple containing the value, policy, and endogenous
            grid points of the point to be saved in case 2.
        intersection_point (tuple): Tuple containing the value, policy, and endogenous
            grid points of the intersection point.
        planned_to_be_saved_this_iter (tuple): Tuple containing the value, policy, and
            endogenous grid point of the point to be saved in this iteration.
        cases (tuple): Tuple containing the indicators for the different cases.

    Returns:
        tuple:

        - point_to_be_saved_next_iteration (tuple): Tuple containing the value,
            policy and endogenous grid point of the point to be
            saved in the next iteration.

    """
    case_0, case_1, case_2, case_3, case_4, case_5, case_6 = cases
    value_case_2, policy_case_2, endog_grid_case_2 = point_case_2
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k

    (
        planned_value,
        planned_policy,
        planned_endog_grid,
    ) = planned_to_be_saved_this_iter

    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point

    value_to_be_saved_next = (
        case_0 * value_k_and_j[1]
        + case_1 * value_to_inspect
        + case_2 * value_case_2
        + jax.lax.select(case_3, on_true=jnp.nan, on_false=0.0)
        + case_4 * value_to_inspect
        + case_5 * intersect_value
        + case_6 * intersect_value
    )

    policy_to_be_saved_next = (
        case_0 * policy_k_and_j[1]
        + case_1 * policy_to_inspect
        + case_2 * policy_case_2
        + jax.lax.select(case_3, on_true=jnp.nan, on_false=0.0)
        + case_4 * policy_to_inspect
        + case_5 * intersect_policy_left
        + case_6 * intersect_policy_right
    )
    endog_grid_to_be_saved_next = (
        case_0 * endog_grid_k_and_j[1]
        + case_1 * endog_grid_to_inspect
        + case_2 * endog_grid_case_2
        + jax.lax.select(case_3, on_true=jnp.nan, on_false=0.0)
        + case_4 * endog_grid_to_inspect
        + case_5 * intersect_grid
        + case_6 * intersect_grid
    )

    return (
        value_to_be_saved_next,
        policy_to_be_saved_next,
        endog_grid_to_be_saved_next,
    )


def select_point_to_save_this_iteration(
    intersection_point,
    planned_to_be_saved_this_iter,
    case_6,
):
    """Select point on the value function to be saved, depending on the case.

    This is the point which, in the previous iteration, was marked to be saved
    this iteration. Except in case 6, where we realize that this point actually
    needs to be disregarded.

    Args:
        intersection_point (tuple): Tuple containing the value, policy, and endogenous
            grid point of the intersection point.
        planned_to_be_saved_this_iter (tuple): Tuple containing the value, policy, and
            endogenous grid point of the point to be saved this iteration.
        case_6 (bool): Indicator if we are in case 6.


    Returns:
        tuple:

        - point_to_save_this_iteration (tuple): Tuple containing the value, policy left,
            policy right, and endogenous grid point of the point to be saved this
            iteration.

    """
    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = intersection_point
    (
        planned_value,
        planned_policy,
        planned_endog_grid,
    ) = planned_to_be_saved_this_iter

    # Determine variables to save this iteration. This is always the variables
    # carried from last iteration. Except in case 6.
    value_to_save = planned_value * (1 - case_6) + intersect_value * case_6
    policy_to_save = planned_policy * (1 - case_6) + intersect_policy_left * case_6
    endog_grid_to_save = planned_endog_grid * (1 - case_6) + intersect_grid * case_6

    return value_to_save, policy_to_save, endog_grid_to_save


def select_and_calculate_intersection(
    endog_grid,
    policy,
    value,
    points_j_and_k,
    idx_next_on_lower_curve,
    idx_before_on_upper_curve,
    idx_to_inspect,
    case_5,
    case_6,
):
    """Select points we use to compute the intersection points.

    This functions maps very nicely into Figure 5 of the paper.
    In case 5, we use the next point (q in the graph) we found on the value function
    segment of point j (i in graph) and intersect it with the idx_to_check
    (i + 1 in the graph).
    In case 6, we are in the situation on the right-hand side in Figure 5.
    Here, we intersect the line of j and k (i and i-1 in the graph) with the line of
    idx_to_check (i+1 in the graph) and the point before on the same value function
    segment.

    Args:
        endog_grid (jnp.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        policy (jnp.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        value (jnp.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        points_j_and_k (tuple): A tuple containing the value, policy, and endogenous
            wealth grid point of the two most recent points on the upper envelope.
        idx_next_on_lower_curve (int): The index of the next point on the lower curve.
        idx_before_on_upper_curve (int): The index of the point before on the upper
            curve.
        idx_to_inspect (int): The index of the point to inspect.
        case_5 (bool): Indicator if we are in case 5.
        case_6 (bool): Indicator if we are in case 6.

    Returns:
        intersect_grid (float): The endogenous grid point of the intersection point.
        intersect_value (float): The value of the value function at the intersection.
        intersect_policy_left (float): The value of the left-continuous policy function
            at the intersection point.
        intersect_policy_right (float): The value of the right-continuous policy
            function at the intersection point.

    """
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k

    wealth_1_on_lower_curve = (
        endog_grid[idx_next_on_lower_curve] * case_5 + endog_grid_k_and_j[0] * case_6
    )
    value_1_on_lower_curve = (
        value[idx_next_on_lower_curve] * case_5 + value_k_and_j[0] * case_6
    )
    policy_1_on_lower_curve = (
        policy[idx_next_on_lower_curve] * case_5 + policy_k_and_j[0] * case_6
    )
    (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    ) = calc_intersection_and_extrapolate_policy(
        wealth_1_lower_curve=wealth_1_on_lower_curve,
        value_1_lower_curve=value_1_on_lower_curve,
        policy_1_lower_curve=policy_1_on_lower_curve,
        wealth_2_lower_curve=endog_grid_k_and_j[1],
        value_2_lower_curve=value_k_and_j[1],
        policy_2_lower_curve=policy_k_and_j[1],
        wealth_1_upper_curve=endog_grid[idx_to_inspect],
        value_1_upper_curve=value[idx_to_inspect],
        policy_1_upper_curve=policy[idx_to_inspect],
        wealth_2_upper_curve=endog_grid[idx_before_on_upper_curve],
        value_2_upper_curve=value[idx_before_on_upper_curve],
        policy_2_upper_curve=policy[idx_before_on_upper_curve],
    )
    return (
        intersect_grid,
        intersect_value,
        intersect_policy_left,
        intersect_policy_right,
    )


def _compute_value(
    consumption, value_function, value_function_args, value_function_kwargs
):
    value = value_function(
        consumption,
        *value_function_args,
        **value_function_kwargs,
    )
    return value
