from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp

from upper_envelope.math_funcs import calc_gradient


def determine_cases_and_conduct_necessary_scans(
    point_to_inspect,
    points_j_and_k,
    value,
    policy,
    endog_grid,
    idx_to_scan_from,
    n_points_to_scan,
    last_point_was_intersect,
    second_last_point_was_intersect,
    is_final_point_on_grid,
    jump_thresh,
):
    """Determine cases and if the index to be scanned this iteration should be updated.

    This function is crucial for the optimality of the FUES. We want to have a clear
    documentation of how the cases determined here map into the into the situations
    on how the candidate solutions after solving the euler equation can look like.
    We need to do more work here!

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy and endogenous grid
            of the point to be inspected.
        points_j_and_k (tuple): Tuple containing the value, policy, and endogenous grid
            point of the most recent point that lies on the upper envelope (j) and
            the point before (k).
        last_point_was_intersect (bool): Indicator if the last point was an
            intersection point.
        is_final_point_on_grid (bool): Indicator if this is the final point on the grid.
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - cases (tuple): Tuple containing the indicators for the different cases.
        - update_idx (bool): Indicator if the index should be updated.

    """
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    # Gradient of the last two points.
    grad_before = calc_gradient(
        x1=endog_grid_k_and_j[1],
        y1=value_k_and_j[1],
        x2=endog_grid_k_and_j[0],
        y2=value_k_and_j[0],
    )

    # gradient with leading index to be checked
    grad_next = calc_gradient(
        x1=endog_grid_to_inspect,
        y1=value_to_inspect,
        x2=endog_grid_k_and_j[1],
        y2=value_k_and_j[1],
    )

    does_the_value_func_switch = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_k_and_j[1],
        policy_1=policy_k_and_j[1],
        endog_grid_2=endog_grid_to_inspect,
        policy_2=policy_to_inspect,
        jump_thresh=jump_thresh,
    )

    decreasing_value = value_to_inspect < value_k_and_j[1]

    are_savings_non_monotone = check_for_non_monotone_savings(
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        endog_grid_idx_to_inspect=endog_grid_to_inspect,
        policy_idx_to_inspect=policy_to_inspect,
    )

    # First check if the point is suboptimal by either a decreasing value or the
    # value function is not monotone.
    is_point_suboptimal = decreasing_value | (
        are_savings_non_monotone & (grad_next < grad_before)
    )

    # If the last point was an intersection point or we reached the end of the grid or
    # we already now that the point is suboptimal or we the value function is not
    # switching, we do not need to scan forward.
    # That is case 1, 2 or we know we are in 3 (Subopimality is checked
    # again below). Additionally check if we are in case 4, i.e. the value function
    # does not switch and we the point is not suboptimal.
    is_forward_scan_needed = ~(
        last_point_was_intersect
        | is_final_point_on_grid
        | is_point_suboptimal
        | ~does_the_value_func_switch
    )
    (
        grad_next_forward,
        idx_next_on_lower_curve,
    ) = forward_scan(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        idx_to_scan_from=idx_to_scan_from,
        n_points_to_scan=n_points_to_scan,
        is_scan_needed=is_forward_scan_needed,
        jump_thresh=jump_thresh,
    )

    switch_value_func_and_steep_increase_after = (
        grad_next < grad_next_forward
    ) & does_the_value_func_switch

    # Additionally check if the gradient of the index we inspect and the point j
    # (the most recent point on the same choice-specific policy) is shallower than
    # the gradient joining the i+1 and j. If True, delete the j'th point.
    suboptimal_cond = switch_value_func_and_steep_increase_after | is_point_suboptimal

    # Same conditions as with forward scan.
    is_backward_scan_needed = ~(
        suboptimal_cond
        | last_point_was_intersect
        | is_final_point_on_grid
        | ~does_the_value_func_switch
    )

    (
        grad_next_backward,
        idx_before_on_upper_curve,
    ) = backward_scan(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        value_j=value_k_and_j[1],
        endog_grid_j=endog_grid_k_and_j[1],
        idx_to_scan_from=idx_to_scan_from,
        n_points_to_scan=n_points_to_scan,
        is_scan_needed=is_backward_scan_needed,
        jump_thresh=jump_thresh,
    )

    next_point_past_intersect = (grad_before > grad_next) | (
        grad_next < grad_next_backward
    )
    point_j_past_intersect = grad_next > grad_next_backward

    # Generate cases. They are exclusive in ascending order, i.e. if 1 is true the
    # rest can't be and 2 can only be true if 1 isn't.
    # Start with checking if last iteration was case_5, and we need
    # to add another point to the refined grid.
    case_0 = second_last_point_was_intersect & last_point_was_intersect
    case_1 = last_point_was_intersect & ~case_0
    case_2 = is_final_point_on_grid & ~case_0 & ~case_1
    case_3 = suboptimal_cond & ~case_0 & ~case_1 & ~case_2
    case_4 = ~does_the_value_func_switch * ~case_0 & ~case_1 * ~case_2 * ~case_3
    case_5 = next_point_past_intersect & ~case_0 & ~case_1 & ~case_2 & ~case_3 & ~case_4
    case_6 = (
        point_j_past_intersect
        & ~case_0
        & ~case_1
        & ~case_2
        & ~case_3
        & ~case_4
        & ~case_5
    )

    in_case_134 = case_1 | case_3 | case_4
    update_idx = (in_case_134 | (~in_case_134 & suboptimal_cond)) & (~(case_0 | case_1))

    return (
        (case_0, case_1, case_2, case_3, case_4, case_5, case_6),
        update_idx,
        idx_next_on_lower_curve,
        idx_before_on_upper_curve,
    )


def check_for_non_monotone_savings(
    endog_grid_j, policy_j, endog_grid_idx_to_inspect, policy_idx_to_inspect
):
    """Check if savings are a non-monotone in wealth.

    Check the grid between the most recent point on the upper envelope (j)
    and the current point we inspect.

    Args:
        endog_grid_j (float): The endogenous grid point of the most recent point
            on the upper envelope.
        policy_j (float): The value of the policy function of the most recent point
            on the upper envelope.
        endog_grid_idx_to_inspect (float): The endogenous grid point of the point we
            check.
        policy_idx_to_inspect (float): The policy index of the point we check.

    Returns:
        non_monotone_policy (bool): Indicator if the policy is non-monotone in wealth
            between the most recent point on the upper envelope and the point we check.

    """
    exog_grid_j = endog_grid_j - policy_j
    exog_grid_idx_to_inspect = endog_grid_idx_to_inspect - policy_idx_to_inspect
    are_savings_non_monotone = exog_grid_idx_to_inspect < exog_grid_j

    return are_savings_non_monotone


def create_indicator_if_value_function_is_switched(
    endog_grid_1: float | jnp.ndarray,
    policy_1: float | jnp.ndarray,
    endog_grid_2: float | jnp.ndarray,
    policy_2: float | jnp.ndarray,
    jump_thresh: float | jnp.ndarray,
):
    """Create boolean to indicate whether value function switches between two points.

    Args:
        endog_grid_1 (float): The first endogenous wealth grid point.
        policy_1 (float): The policy function at the first endogenous wealth grid point.
        endog_grid_2 (float): The second endogenous wealth grid point.
        policy_2 (float): The policy function at the second endogenous wealth grid
            point.
        jump_thresh (float): Jump detection threshold.

    Returns:
        bool: Indicator if value function is switched.

    """
    exog_grid_1 = endog_grid_1 - policy_1
    exog_grid_2 = endog_grid_2 - policy_2
    gradient_exog_grid = calc_gradient(
        x1=endog_grid_1,
        y1=exog_grid_1,
        x2=endog_grid_2,
        y2=exog_grid_2,
    )
    gradient_exog_abs = jnp.abs(gradient_exog_grid)
    is_switched = gradient_exog_abs > jump_thresh

    return is_switched


def forward_scan(
    value: jnp.ndarray,
    policy: jnp.array,
    endog_grid: jnp.ndarray,
    endog_grid_j: float,
    policy_j: float,
    idx_to_scan_from: int,
    n_points_to_scan: int,
    is_scan_needed,
    jump_thresh: float,
) -> Tuple[float, int]:
    """Find next point on same value function as most recent point on upper envelope.

    We use the forward scan to find the next point that lies on the same value
    function segment as the most recent point on the upper envelope (j).
    Then we calculate the gradient between the point found on the same value function
    segment and the point we currently inspect at idx_to_scan_from.

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        endog_grid_j (float): Endogenous grid point that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        policy_j (float): Point of the policy function that corresponds to the most
            recent value function point that lies on the upper envelope (j).
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - grad_next_forward (float): The gradient of the next point on the same
            value function.
        - idx_on_same_value (int): Index of next point on the value function.

    """
    (
        grad_next_on_same_value,
        idx_on_same_value,
    ) = back_and_forward_scan_wrapper(
        endog_grid_to_calculate_gradient=endog_grid[idx_to_scan_from],
        value_to_calculate_gradient=value[idx_to_scan_from],
        endog_grid_to_scan_from=endog_grid_j,
        policy_to_scan_from=policy_j,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        idx_to_scan_from=idx_to_scan_from,
        n_points_to_scan=n_points_to_scan,
        is_scan_needed=is_scan_needed,
        jump_thresh=jump_thresh,
        direction="forward",
    )

    return (
        grad_next_on_same_value,
        idx_on_same_value,
    )


def backward_scan(
    value: jnp.ndarray,
    policy: jnp.ndarray,
    endog_grid: jnp.ndarray,
    endog_grid_j,
    value_j,
    idx_to_scan_from: int,
    n_points_to_scan: int,
    is_scan_needed,
    jump_thresh: float,
) -> Tuple[float, int]:
    """Find previous point on same value function as idx_to_scan_from.

    We use the backward scan to find the preceding point that lies on the same value
    function segment as the point we inspect. Then we calculate the gradient between
    the point found and the most recent point on the upper envelope (j).

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        endog_grid_j (float): Endogenous grid point that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        value_j (float): Point of the value function that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - grad_before_on_same_value (float): The gradient of the previous point on
            the same value function.
        - is_before_on_same_value (int): Indicator for whether we have found a
            previous point on the same value function.

    """
    (
        grad_before_on_same_value,
        idx_point_before_on_same_value,
    ) = back_and_forward_scan_wrapper(
        endog_grid_to_calculate_gradient=endog_grid_j,
        value_to_calculate_gradient=value_j,
        endog_grid_to_scan_from=endog_grid[idx_to_scan_from],
        policy_to_scan_from=policy[idx_to_scan_from],
        endog_grid=endog_grid,
        value=value,
        idx_to_scan_from=idx_to_scan_from,
        policy=policy,
        n_points_to_scan=n_points_to_scan,
        is_scan_needed=is_scan_needed,
        jump_thresh=jump_thresh,
        direction="backward",
    )

    return grad_before_on_same_value, idx_point_before_on_same_value


def back_and_forward_scan_wrapper(
    endog_grid_to_calculate_gradient: float | jnp.ndarray,
    value_to_calculate_gradient: float | jnp.ndarray,
    endog_grid_to_scan_from: float | jnp.ndarray,
    policy_to_scan_from: float | jnp.ndarray,
    endog_grid: float | jnp.ndarray,
    value: jnp.ndarray,
    policy: jnp.ndarray,
    idx_to_scan_from: int,
    n_points_to_scan: int,
    is_scan_needed: bool,
    jump_thresh: float,
    direction: str,
):
    """Wrapper function to execute the backwards and forward scan.

    Args:
        endog_grid_to_calculate_gradient (float): The endogenous grid point to calculate
            the gradient from.
        value_to_calculate_gradient (float): The value function point to calculate the
            gradient from.
        endog_grid_to_scan_from (float): The endogenous grid point to scan from. We want
            to find the grid point which is on the same value function segment as the
            point we scan from.
        policy_to_scan_from (float): The policy function point to scan from. We want to
            find the grid point which is on the same value function segment as the point
            we scan from.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        is_scan_needed (bool): Indicator if the scan is needed.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.
        direction (str): The direction of the scan. Either 'forward' or 'backward'.

    Returns:
        tuple:

        - grad_we_search_for (float): The gradient we search for.
        - idx_on_same_value (int): The index of the point on the same value function
            segment as the point we scan from.

    """
    # Prepare body function by partialing in, everything except carry and counter
    partial_body = partial(
        back_and_forward_scan_body,
        endog_grid_to_calculate_gradient=endog_grid_to_calculate_gradient,
        value_to_calculate_gradient=value_to_calculate_gradient,
        endog_grid_to_scan_from=endog_grid_to_scan_from,
        policy_to_scan_from=policy_to_scan_from,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        jump_thresh=jump_thresh,
        direction=direction,
    )

    if direction == "forward":
        max_index = idx_to_scan_from + n_points_to_scan

        def cond_func(carry):
            (
                is_on_same_value,
                idx_on_same_value,
                grad_we_search_for,
                current_index,
            ) = carry
            return (
                is_scan_needed
                & ~is_on_same_value
                & (current_index < max_index)
                & (current_index < len(endog_grid))
            )

        start_index = idx_to_scan_from + 1

    elif direction == "backward":
        min_index = idx_to_scan_from - n_points_to_scan

        def cond_func(carry):
            (
                is_on_same_value,
                idx_on_same_value,
                grad_we_search_for,
                current_index,
            ) = carry
            return (
                is_scan_needed
                & ~is_on_same_value
                & (current_index > min_index)
                & (current_index >= 0)
            )

        start_index = idx_to_scan_from - 1
    else:
        raise ValueError("Direction must be either 'forward' or 'backward'.")

    # Initialize starting values
    is_on_same_value = False
    idx_on_same_value = 0
    grad_we_search_for = 0.0

    # These values will be updated each iteration.
    carry_to_update = (
        is_on_same_value,
        idx_on_same_value,
        grad_we_search_for,
        start_index,
    )

    # Execute scan function. The result is the final carry value.
    final_carry = jax.lax.while_loop(
        cond_fun=cond_func, body_fun=partial_body, init_val=carry_to_update
    )

    # Read out final carry.
    (
        is_on_same_value,
        idx_on_same_value,
        grad_we_search_for,
        start_index,
    ) = final_carry

    return (
        grad_we_search_for,
        idx_on_same_value,
    )


def back_and_forward_scan_body(
    carry,
    endog_grid_to_calculate_gradient,
    value_to_calculate_gradient,
    endog_grid_to_scan_from,
    policy_to_scan_from,
    endog_grid,
    value,
    policy,
    jump_thresh,
    direction,
):
    """The scan body to be executed at each iteration of the backwards and forward scan
    function.

    Args:
        carry (tuple): The carry value passed from the previous iteration. This is a
            tuple containing the variables that are updated in each iteration.
        endog_grid_to_calculate_gradient (float): The endogenous grid point to calculate
            the gradient from.
        value_to_calculate_gradient (float): The value function point to calculate the
            gradient from.
        endog_grid_to_scan_from (float): The endogenous grid point to scan from. We want
            to find the grid point which is on the same value function segment as the
            point we scan from.
        policy_to_scan_from (float): The policy function point to scan from. We want to
            find the grid point which is on the same value function segment as the point
            we scan from.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        jump_thresh (float): Threshold for the jump in the value function.
        direction (str): The direction of the scan. Either 'forward' or 'backward'.

    Returns:
        tuple:

        - carry (tuple): The updated carry value passed to the next iteration.
        - None: Dummy value to be returned.

    """
    (
        _,
        _,
        _,
        current_index_to_scan,
    ) = carry

    is_not_on_same_value = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_to_scan_from,
        policy_1=policy_to_scan_from,
        endog_grid_2=endog_grid[current_index_to_scan],
        policy_2=policy[current_index_to_scan],
        jump_thresh=jump_thresh,
    )
    is_on_same_value = ~is_not_on_same_value

    grad_to_idx_to_scan = calc_gradient(
        x1=endog_grid_to_calculate_gradient,
        y1=value_to_calculate_gradient,
        x2=endog_grid[current_index_to_scan],
        y2=value[current_index_to_scan],
    )

    # Update if we found the point we search for
    idx_on_same_value = current_index_to_scan * is_on_same_value

    # Update the first time a new point is found
    grad_we_search_for = grad_to_idx_to_scan * is_on_same_value
    if direction == "forward":
        current_index_to_scan += 1
    elif direction == "backward":
        current_index_to_scan -= 1

    return (
        is_on_same_value,
        idx_on_same_value,
        grad_we_search_for,
        current_index_to_scan,
    )
