from typing import Any, Callable

import jax.numpy as jnp
from flax.core import freeze, copy
import optax
import jax
import flax


def identity(x):
    return x


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predictions - targets) ** 2)


# Recursively freezes collections so they are save to use as agent specification
def freeze_static_config(value):
    if isinstance(value, dict):
        return freeze({key: freeze_static_config(item) for key, item in value.items()})

    if isinstance(value, list):
        return tuple(freeze_static_config(item) for item in value)

    if isinstance(value, tuple):
        return tuple(freeze_static_config(item) for item in value)

    return value


def gradient_update(
    variables: flax.core.FrozenDict,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[[flax.core.FrozenDict, Any], jnp.ndarray],
    **loss_arguments,
) -> tuple[flax.core.FrozenDict, optax.OptState, jnp.ndarray]:
    parameters = variables["params"]  # makes sure that only parameters are updated

    def loss_from_parameters(parameters: flax.core.FrozenDict) -> jnp.ndarray:
        complete_variables = copy(
            variables,
            {"params": parameters},
        )
        return loss_function(complete_variables, **loss_arguments)

    loss, gradients = jax.value_and_grad(
        loss_from_parameters,
        has_aux=False,
    )(parameters)

    updates, optimizer_state = optimizer.update(
        gradients,
        optimizer_state,
        parameters,
    )
    parameters = optax.apply_updates(parameters, updates)

    variables = variables.copy({"params": parameters})
    return variables, optimizer_state, loss


def gradient_update_with_auxiliary(
    variables: flax.core.FrozenDict,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    differentiated_loss_function: Callable[[flax.core.FrozenDict, Any], jnp.ndarray],
    **loss_arguments,
) -> tuple[flax.core.FrozenDict, optax.OptState, jnp.ndarray, Any]:
    parameters = variables["params"]

    def loss_from_parameters(parameters: flax.core.FrozenDict) -> jnp.ndarray:
        complete_variables = copy(
            variables,
            {"params": parameters},
        )
        return differentiated_loss_function(complete_variables, **loss_arguments)

    (loss, auxiliary), gradients = jax.value_and_grad(
        loss_from_parameters,
        has_aux=True,
    )(parameters)

    updates, optimizer_state = optimizer.update(
        gradients,
        optimizer_state,
        parameters,
    )
    parameters = optax.apply_updates(parameters, updates)

    variables = variables.copy({"params": parameters})
    return variables, optimizer_state, loss, auxiliary
