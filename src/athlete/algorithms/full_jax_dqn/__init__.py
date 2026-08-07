from typing import Dict, Any, Tuple

from gymnasium.spaces import Space, Box, Discrete
import optax
import jax.numpy as jnp
import flashbax as fbx
import jax
import flax

import athlete
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

from athlete.algorithms.full_jax_dqn.objects import DQNAgentState


def make_full_jax_dqn(
    observation_space: Space,
    action_space: Space,
    replay_buffer_capacity: int = 100_000,
    replay_buffer_mini_batch_size: int = 128,
):

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

    agent_state = DQNAgentState(
        replay_buffer_func=replay_buffer, replay_buffer_state=replay_buffer_state
    )

    return agent_state
