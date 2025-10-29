from typing import Dict, Any, Tuple

from gymnasium.spaces import Space, Box, Discrete
import optax
import jax.numpy as jnp

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
from athlete.function import jax_mse_loss

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


def make_jax_dqn_components(
    observation_space: Space, action_space: Space, configuration: Dict[str, Any]
) -> Tuple[DataCollector, UpdateRule, Policy, Policy]:
    """Creates the components for a DQN agent.

    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        configuration: Algorithm configuration dictionary

    Returns:
        Tuple containing data collector, update rule training policy, and evaluation policy

    Raises:
        ValueError: If observation_space is not Box or action_space is not Discrete
    """

    if not isinstance(observation_space, Box):
        raise ValueError(
            f"This DQN implementation only supports {Box.__name__} observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Discrete):
        raise ValueError(
            f"This DQN implementation only supports {Discrete.__name__} action spaces, but got {type(action_space)}"
        )

    configuration[ARGUMENT_VALUE_NETWORK_ARGUMENTS].update(
        {
            ARGUMENT_OBSERVATION_SHAPE: observation_space.shape,
            ARGUMENT_NUM_ACTIONS: action_space.n.item(),
        }
    )

    value_function = configuration[ARGUMENT_VALUE_NETWORK_CLASS](
        **configuration[ARGUMENT_VALUE_NETWORK_ARGUMENTS]
    )

    # DATA PROVIDER
    update_data_input = UpdateDataProvider()

    # DATA COLLECTOR

    data_collector = GymnasiumTransitionDataCollector(
        update_data_provider=update_data_input,
    )

    # UPDATE RULE
    dummy_input = jnp.zeros((1, *observation_space.shape), dtype=jnp.float32)
    random_key = RNGHandler.get_instance().get_jax_key()
    initial_q_value_function_parameters = value_function.init(random_key, dummy_input)
    immutable_q_value_function = ModuleState(
        apply_fn=value_function.apply,
        params=initial_q_value_function_parameters,
    )
    mutable_q_value_function = MutableJaxModule(
        module=immutable_q_value_function,
    )

    update_rule = JAXDQNUpdate(
        observation_space=observation_space,
        action_space=action_space,
        update_data_input=update_data_input,
        mutable_q_value_function=mutable_q_value_function,
        discount=configuration[ARGUMENT_DISCOUNT],
        optimizer_arguments=configuration[ARGUMENT_OPTIMIZER_ARGUMENTS],
        replay_buffer_capacity=configuration[ARGUMENT_REPLAY_BUFFER_CAPACITY],
        replay_buffer_mini_batch_size=configuration[
            ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE
        ],
        value_net_update_frequency=configuration[ARGUMENT_VALUE_NET_UPDATE_FREQUENCY],
        value_net_number_of_updates=configuration[ARGUMENT_VALUE_NET_NUMBER_OF_UPDATES],
        multiply_number_of_updates_by_environment_steps=configuration[
            ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS
        ],
        target_net_update_frequency=configuration[ARGUMENT_TARGET_NET_UPDATE_FREQUENCY],
        target_net_tau=configuration[ARGUMENT_TARGET_NET_TAU],
        enable_double_q_learning=configuration[ARGUMENT_ENABLE_DOUBLE_Q_LEARNING],
        criteria=configuration[ARGUMENT_CRITERIA],
        optimizer_class=configuration[ARGUMENT_OPTIMIZER_CLASS],
        gradient_max_norm=configuration[ARGUMENT_GRADIENT_MAX_NORM],
        additional_replay_buffer_arguments=configuration[
            ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS
        ],
        post_replay_buffer_data_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    # POLICY

    training_policy = JAXDQNTrainingPolicy(
        mutable_q_value_function=mutable_q_value_function,
        action_space=action_space,
        start_epsilon=configuration[ARGUMENT_START_EPSILON],
        end_epsilon=configuration[ARGUMENT_END_EPSILON],
        epsilon_decay_steps=configuration[ARGUMENT_EPSILON_DECAY_STEPS],
        post_replay_buffer_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    evaluation_policy = JAXDQNEvaluationPolicy(
        mutable_q_value_function=mutable_q_value_function,
        post_replay_buffer_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    return data_collector, update_rule, training_policy, evaluation_policy


athlete.register(
    id="jax_dqn",
    component_factory=make_jax_dqn_components,
    default_configuration=DEFAULT_CONFIGURATION,
)
