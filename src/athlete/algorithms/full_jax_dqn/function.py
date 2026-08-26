import importlib
from typing import Any, Callable

import copy
import jax.numpy as jnp
import flax
import optax
import jax


def identity(x):
    return x


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predictions - targets) ** 2)


# TODO maybe make this part of a fixed jax make function construct instead of doing it manually in every make function
def deepcopy_preserving_callables(value):
    leaves = jax.tree_util.tree_leaves(
        value,
        is_leaf=callable,
    )

    callable_memo = {id(leaf): leaf for leaf in leaves if callable(leaf)}

    return copy.deepcopy(value, memo=callable_memo)


# Recursively freezes collections so they are save to use as agent specification
def freeze_static_config(value):
    if isinstance(value, dict):
        return flax.core.freeze(
            {key: freeze_static_config(item) for key, item in value.items()}
        )

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
        complete_variables = flax.core.copy(
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
        complete_variables = flax.core.copy(
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


def resolve_dotted_reference(path: str):
    parts = path.split(".")

    for split_index in range(len(parts), 0, -1):
        module_path = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_path)
            break
        except ModuleNotFoundError as error:
            missing_module = error.name
            if missing_module is None or not (
                module_path == missing_module
                or module_path.startswith(f"{missing_module}.")
            ):
                raise
    else:
        raise ImportError(f"Could not import any module from path: {path}")

    for attribute_name in parts[split_index:]:
        obj = getattr(obj, attribute_name)

    return obj
