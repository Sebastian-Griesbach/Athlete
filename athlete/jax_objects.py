from typing import Callable, Any

import flax
import flax.struct
import optax

from jax import numpy as jnp


class ModuleState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)

    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.apply_fn(self.params, x, *args, **kwargs)


class OptimizerState(flax.struct.PyTreeNode):
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState = flax.struct.field(pytree_node=True)


class FunctionWrapper(flax.struct.PyTreeNode):
    function: Callable = flax.struct.field(pytree_node=False)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class MutableJaxModule:
    """Mutable wrapper for immutable JAX parameters."""

    def __init__(self, module: ModuleState):
        self._immutable_module = module

    def set(self, module: ModuleState):
        """Set new module."""
        self._immutable_module = module

    def get(self) -> ModuleState:
        """Get the current module."""
        return self._immutable_module


class MutableOptaxOptimizer:
    """Mutable wrapper for immutable optimizer state."""

    def __init__(self, optimizer: OptimizerState):
        self._immutable_optimizer = optimizer

    def set(self, optimizer: OptimizerState):
        """Set new optimizer state."""
        self._immutable_optimizer = optimizer

    def get(self) -> OptimizerState:
        """Get the current optimizer state."""
        return self._immutable_optimizer
