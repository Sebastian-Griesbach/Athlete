import jax.numpy as jnp
from flax.core import freeze


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
