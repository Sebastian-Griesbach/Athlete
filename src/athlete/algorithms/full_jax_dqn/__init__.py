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

ARGUMENT_DISCOUNT = "discount"
ARGUMENT_VALUE_NETWORK_CLASS = "value_network_class"
ARGUMENT_VALUE_NETWORK_ARGUMENTS = "value_network_arguments"
ARGUMENT_OPTIMIZER_CLASS = "optimizer_class"
ARGUMENT_OPTIMIZER_ARGUMENTS = "optimizer_arguments"
ARGUMENT_REPLAY_BUFFER_CAPACITY = "replay_buffer_capacity"
ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE = "replay_buffer_mini_batch_size"
ARGUMENT_START_EPSILON = "start_epsilon"
ARGUMENT_END_EPSILON = "end_epsilon"
ARGUMENT_EPSILON_DECAY_STEPS = "epsilon_decay_steps"
ARGUMENT_VALUE_NET_UPDATE_FREQUENCY = "value_net_update_frequency"
ARGUMENT_VALUE_NET_NUMBER_OF_UPDATES = "value_net_number_of_updates"
ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS = (
    "multiply_number_of_updates_by_environment_steps"
)
ARGUMENT_TARGET_NET_UPDATE_FREQUENCY = "target_net_update_frequency"
ARGUMENT_TARGET_NET_TAU = "target_net_tau"
ARGUMENT_ENABLE_DOUBLE_Q_LEARNING = "enable_double_q_learning"
ARGUMENT_CRITERIA = "criteria"
ARGUMENT_GRADIENT_MAX_NORM = "gradient_max_norm"
ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS = "additional_replay_buffer_arguments"
ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING = "post_replay_buffer_data_preprocessing"
ARGUMENT_OBSERVATION_SHAPE = "observation_shape"
ARGUMENT_NUM_ACTIONS = "num_actions"


DEFAULT_CONFIGURATION = {
    ARGUMENT_DISCOUNT: 0.99,
    ARGUMENT_VALUE_NETWORK_CLASS: FlaxFCDiscreteQValueFunction,
    ARGUMENT_VALUE_NETWORK_ARGUMENTS: {
        ARGUMENT_OBSERVATION_SHAPE: constants.VALUE_PLACEHOLDER,
        ARGUMENT_NUM_ACTIONS: constants.VALUE_PLACEHOLDER,
        "hidden_dims": (256, 256),
    },
    ARGUMENT_OPTIMIZER_CLASS: optax.adam,
    ARGUMENT_OPTIMIZER_ARGUMENTS: {"learning_rate": 6.3e-4},
    ARGUMENT_REPLAY_BUFFER_CAPACITY: 100000,
    ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE: 128,
    ARGUMENT_START_EPSILON: 1.0,
    ARGUMENT_END_EPSILON: 0.1,
    constants.GENERAL_ARGUMENT_WARMUP_STEPS: 1000,
    ARGUMENT_EPSILON_DECAY_STEPS: 12000,
    ARGUMENT_VALUE_NET_UPDATE_FREQUENCY: 4,
    ARGUMENT_VALUE_NET_NUMBER_OF_UPDATES: 1,
    ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS: True,
    ARGUMENT_TARGET_NET_UPDATE_FREQUENCY: 250,
    ARGUMENT_TARGET_NET_TAU: 1.0,
    ARGUMENT_ENABLE_DOUBLE_Q_LEARNING: False,
    ARGUMENT_CRITERIA: jax_mse_loss,
    ARGUMENT_GRADIENT_MAX_NORM: 10.0,
    ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS: {},
    ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING: None,
}


def make_full_jax_dqn(
    observation_space: Space, action_space: Space, configuration: Dict[str, Any]
):

    # Replay Buffer
    replay_buffer = fbx.make_flat_buffer(
        max_length=configuration[ARGUMENT_REPLAY_BUFFER_CAPACITY],
        min_length=configuration[ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE],
        sample_batch_size=configuration[ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE],
        add_sequence=True,
        add_batch_size=configuration[ARGUMENT_VALUE_NET_UPDATE_FREQUENCY],
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
