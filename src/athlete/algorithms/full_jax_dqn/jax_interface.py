from typing import Callable, Dict, Tuple, Any, Optional
import importlib
from dataclasses import dataclass

import chex
import flax
import jax
import pickle

# TODO think about auto reset modes similar to gymnasium, agent gets two observations and immediately
# calls step for final observation and reset step for initial observation, how exactly?


class InfoValue(flax.struct.PyTreeNode):
    value: jax.Array
    valid: jax.Array


@dataclass
class JaxMakeSpecification:
    make_function_path: str
    make_arguments: Dict[str, Any]


@dataclass
class JaxAgentCheckpointPayload:
    agent_state_bytes: bytes
    make_specification: JaxMakeSpecification


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
    def save_to_file(
        save_path: str,
        agent_state: flax.struct.PyTreeNode,
        make_specification: JaxMakeSpecification,
    ) -> None:
        payload = JaxAgent.get_save_payload(
            agent_state=agent_state,
            make_specification=make_specification,
        )

        with open(save_path, "wb") as file:
            pickle.dump(payload, file)

    @staticmethod
    def get_save_payload(
        agent_state: flax.struct.PyTreeNode,
        make_specification: JaxMakeSpecification,
    ) -> JaxAgentCheckpointPayload:

        # create a copy to avoid changing meta data of the running original
        encoded_make_specification = JaxMakeSpecification(
            make_function_path=make_specification.make_function_path,
            make_arguments=encode_references(make_specification.make_arguments),
        )
        payload = JaxAgentCheckpointPayload(
            make_specification=encoded_make_specification,
            agent_state_bytes=flax.serialization.to_bytes(agent_state),
        )
        return payload

    @staticmethod
    def load_from_file(
        load_path: str,
    ):
        with open(load_path, "rb") as file:
            payload: JaxAgentCheckpointPayload = pickle.load(file)
        return JaxAgent.load_from_payload(payload=payload)

    @staticmethod
    def load_from_payload(
        payload: JaxAgentCheckpointPayload,
    ) -> Tuple["JaxAgent", flax.struct.PyTreeNode, JaxMakeSpecification]:

        agent_state_bytes = payload.agent_state_bytes
        make_specification = payload.make_specification
        # During runtime we keep class and object references in the make_arguments

        # Do not change payload in place but make a copy for decoding
        decoded_make_specification = JaxMakeSpecification(
            make_function_path=make_specification.make_function_path,
            make_arguments=decode_references(make_specification.make_arguments),
        )

        make_function_path = make_specification.make_function_path
        module_path, function_name = make_function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        make_function = getattr(module, function_name)

        # avoid having two agent states on GPU at the same time in case they are large
        with jax.default_device(jax.devices("cpu")[0]):
            agent, template_state, _ = make_function(
                **decoded_make_specification.make_arguments
            )
            loaded_state = flax.serialization.from_bytes(
                template_state,
                agent_state_bytes,
            )

        agent_state = jax.device_put(loaded_state)

        return agent, agent_state, decoded_make_specification


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
