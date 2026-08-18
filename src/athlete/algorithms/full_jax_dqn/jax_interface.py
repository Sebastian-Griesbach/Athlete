from typing import Callable, Dict, Tuple
import inspect
import importlib

import chex
import flax
import jax
import pickle

# TODO think about auto reset modes similar to gymnasium, agent gets two observations and immediately
# calls step for final observation and reset step for initial observation, how exactly?


@chex.dataclass(frozen=True)
class JaxAgent:
    step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    reset_step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    make_evaluation_agent: Callable[
        [flax.struct.PyTreeNode, flax.struct.PyTreeNode],
        Tuple[flax.struct.PyTreeNode, "JaxEvaluationAgent"],
    ]

    @staticmethod
    def save(
        save_path: str, agent_state: flax.struct.PyTreeNode, make_arguments: Dict
    ) -> None:
        checkpoint = {
            "agent_state": flax.serialization.to_bytes(agent_state),
            "make_arguments": encode_references(make_arguments),
        }
        with open(save_path, "wb") as file:
            pickle.dump(checkpoint, file)


@chex.dataclass(frozen=True)
class JaxEvaluationAgent:
    step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    reset_step: Callable[
        [
            flax.struct.PyTreeNode,
            flax.struct.PyTreeNode,
            jax.Array,
        ],
        Tuple[flax.struct.PyTreeNode, jax.Array, Dict[str, jax.Array]],
    ]
    # TODO save and load for evaluation agents, not clear how to handle make arguments, taking detour over train agent seems wasteful


# Helper function for saving and loading that resolves named functions and classes which are part of the make_arguments
def is_importable_reference(value):
    return (
        callable(value)
        and hasattr(value, "__module__")
        and hasattr(value, "__qualname__")
        and "<locals>" not in value.__qualname__
        and "<lambda>" not in value.__qualname__
    )


def encode_references(value):
    if is_importable_reference(value):
        return {
            "__callable_ref__": True,
            "module": value.__module__,
            "qualname": value.__qualname__,
        }

    if isinstance(value, dict):
        return {key: encode_references(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(encode_references(item) for item in value)

    if isinstance(value, list):
        return [encode_references(item) for item in value]

    return value


def decode_references(value):
    if isinstance(value, dict) and value.get("__callable_ref__"):
        module = importlib.import_module(value["module"])
        obj = module
        for part in value["qualname"].split("."):
            obj = getattr(obj, part)
        return obj

    if isinstance(value, dict):
        return {key: decode_references(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(decode_references(item) for item in value)

    if isinstance(value, list):
        return [decode_references(item) for item in value]

    return value
