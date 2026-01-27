import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numba import njit

import upper_envelope as upenv

jax.config.update("jax_enable_x64", True)

ROOT_DIR = Path(__file__).resolve().parents[1]

n_runs = 10


def utility_crra_jax(
    consumption: jnp.ndarray, choice: int, params: Dict[str, float]
) -> jnp.ndarray:
    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])
    utility = utility_consumption - (1 - choice) * params["delta"]
    return utility


def utility_crra_np(
    consumption: np.ndarray, choice: int, params: Dict[str, float]
) -> np.ndarray:
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


test_resources = ROOT_DIR / "tests" / "resources" / "upper_envelope_period_tests"

period = 2
value_egm = np.genfromtxt(
    test_resources / f"val{period}.csv", delimiter=",", dtype=float
)
policy_egm = np.genfromtxt(
    test_resources / f"pol{period}.csv", delimiter=",", dtype=float
)

params: Dict[str, float] = {"beta": 0.95, "rho": 1.95, "delta": 0.35}
state_choice = {"lagged_choice": 0, "choice": 0}


def value_func_jax(consumption, choice, params):
    return (
        utility_crra_jax(consumption, choice, params) + params["beta"] * value_egm[1, 0]
    )


def fuex_jax_partial(endog, pol, val, exp_val_zero):
    return upenv.fues_jax(
        endog_grid=jnp.asarray(endog),
        policy=jnp.asarray(pol),
        value=jnp.asarray(val),
        expected_value_zero_savings=exp_val_zero,
        value_function=value_func_jax,
        value_function_args=(state_choice["choice"], params),
    )


# Compile time
start = time.time()
fuex_jax_partial(
    endog=policy_egm[0, 1:],
    pol=policy_egm[1, 1:],
    val=value_egm[1, 1:],
    exp_val_zero=value_egm[1, 0],
)
end = time.time()
print(f"JAX FUES compilation time: {end - start:.4f} seconds")

tot_time = 0.0
for _ in range(n_runs):
    start = time.time()
    jax.block_until_ready(
        fuex_jax_partial(
            endog=policy_egm[0, 1:],
            pol=policy_egm[1, 1:],
            val=value_egm[1, 1:],
            exp_val_zero=value_egm[1, 0],
        )
    )
    end = time.time()
    tot_time += end - start

print(f"JAX FUES average time over {n_runs} runs: {tot_time / n_runs:.6f} seconds")


def drued_jorg_jax_partial(endog, pol, val, m_grid, exp_val_zero):
    return upenv.drued_jorg_jax(
        endog_grid=jnp.asarray(endog),
        policy=jnp.asarray(pol),
        value=jnp.asarray(val),
        m_grid=jnp.asarray(m_grid),
        expected_value_zero_savings=exp_val_zero,
        value_function=value_func_jax,
        value_function_args=(state_choice["choice"], params),
    )


# Compile time
m_min = float(np.min(policy_egm[0, 1:]))
m_max = float(np.max(policy_egm[0, 1:]))
m_grid = np.linspace(m_min, m_max, 500)
start = time.time()
drued_jorg_jax_partial(
    endog=policy_egm[0, 1:],
    pol=policy_egm[1, 1:],
    val=value_egm[1, 1:],
    m_grid=m_grid,
    exp_val_zero=value_egm[1, 0],
)
end = time.time()
print(f"JAX DRUED-JORG compilation time: {end - start:.4f} seconds")
tot_time = 0.0
for _ in range(n_runs):
    start = time.time()
    jax.block_until_ready(
        drued_jorg_jax_partial(
            endog=policy_egm[0, 1:],
            pol=policy_egm[1, 1:],
            val=value_egm[1, 1:],
            m_grid=m_grid,
            exp_val_zero=value_egm[1, 0],
        )
    )
    end = time.time()
    tot_time += end - start
print(
    f"JAX DRUED-JORG average time over {n_runs} runs: {tot_time / n_runs:.6f} seconds"
)


# Numba at last
numba_args = (
    int(state_choice["choice"]),
    float(params["beta"]),
    float(params["rho"]),
    float(params["delta"]),
    float(value_egm[1, 0]),
)

start = time.time()
upenv.fues_numba(
    endog_grid=policy_egm[0, 1:],
    policy=policy_egm[1, 1:],
    value=value_egm[1, 1:],
    expected_value_zero_savings=value_egm[1, 0],
    value_function=value_func_numba,
    value_function_args=numba_args,
)
end = time.time()
print(f"Numba FUES compilation time: {end - start:.4f} seconds")

tot_time = 0.0
for _ in range(n_runs):
    start = time.time()
    jax.block_until_ready(
        upenv.fues_numba(
            endog_grid=policy_egm[0, 1:],
            policy=policy_egm[1, 1:],
            value=value_egm[1, 1:],
            expected_value_zero_savings=value_egm[1, 0],
            value_function=value_func_numba,
            value_function_args=numba_args,
        )
    )
    end = time.time()
    tot_time += end - start

print(f"Numba FUES average time over {n_runs} runs: {tot_time / n_runs:.6f} seconds")
