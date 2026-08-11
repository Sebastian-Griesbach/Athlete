from typing import Dict, Any, Optional, Tuple
import random

from gymnasium.spaces import Space, Box, Discrete
import optax
import jax.numpy as jnp
import flashbax as fbx
import jax
import flax

import athlete
from athlete.algorithms.full_jax_dqn.agent import DQNAgentSpecification
from athlete.update.update_rule import UpdateRule
from athlete.policy.policy import Policy
from athlete.data_collection.provider import UpdateDataProvider
from athlete import constants
from athlete.data_collection.collector import DataCollector
from athlete.data_collection.transition import GymnasiumTransitionDataCollector
from athlete.global_objects import RNGHandler
from athlete.jax_objects import MutableJaxModule, ModuleState, FunctionWrapper
from athlete.algorithms.jax_dqn.update import JAXDQNUpdate
from athlete.algorithms.jax_dqn.policy import (
    JAXDQNTrainingPolicy,
    JAXDQNEvaluationPolicy,
)
from athlete.module.jax.common import FlaxFCDiscreteQValueFunction
from athlete.function import jax_mse_loss, create_transition_data_info

from athlete.algorithms.full_jax_dqn.agent import DQNAgentState, DQNAgentSpecification


def make_full_jax_dqn(
    observation_space: Space,
    action_space: Space,
    replay_buffer_capacity: int = 100_000,
    replay_buffer_mini_batch_size: int = 128,
    value_network_class: Any = FlaxFCDiscreteQValueFunction,
    value_network_arguments: Dict[str, Any] = {
        "observation_shape": constants.VALUE_PLACEHOLDER,
        "num_actions": constants.VALUE_PLACEHOLDER,
        "hidden_dims": (256, 256),
    },
    optimizer_class: Any = optax.adam,
    optimizer_arguments: Dict[str, Any] = {"learning_rate": 6.3e-4},
    random_key: Optional[jax.Array] = None,
) -> Tuple[DQNAgentSpecification, DQNAgentState]:

    if not isinstance(observation_space, Box):
        raise ValueError(
            f"This DQN implementation only supports {Box.__name__} observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Discrete):
        raise ValueError(
            f"This DQN implementation only supports {Discrete.__name__} action spaces, but got {type(action_space)}"
        )

    # Replay Buffer
    replay_buffer = fbx.make_flat_buffer(
        max_length=replay_buffer_capacity,
        min_length=replay_buffer_mini_batch_size,
        sample_batch_size=replay_buffer_mini_batch_size,
        add_sequence=True,
        add_batch_size=1,
    )

    transition_data_info = create_transition_data_info(
        observation_space=observation_space, action_space=action_space
    )
    dummy_transition = {
        field: jnp.zeros(info["shape"], dtype=info["dtype"])
        for field, info in transition_data_info.items()
    }

    replay_buffer_state = replay_buffer.init(dummy_transition)

    # Value function
    value_network_arguments["observation_shape"] = observation_space.shape
    value_network_arguments["num_actions"] = action_space.n
    q_value_function = value_network_class(**value_network_arguments)

    dummy_input = jnp.zeros((1, *observation_space.shape), dtype=jnp.float32)
    if random_key is None:
        random_key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))

    random_key, sub_key = jax.random.split(random_key)

    q_value_function_variables = target_q_value_function_variables = (
        q_value_function.init(sub_key, dummy_input)
    )

    agent_state = DQNAgentState(
        replay_buffer_func=replay_buffer,
        replay_buffer_state=replay_buffer_state,
        q_value_function_variables=q_value_function_variables,
        target_q_value_function_variables=target_q_value_function_variables,
        random_key=random_key,
    )

    return agent_state
