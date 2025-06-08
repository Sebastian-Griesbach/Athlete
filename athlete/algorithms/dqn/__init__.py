from typing import Dict, Any, Tuple

from gymnasium.spaces import Space, Box, Discrete
import torch

import athlete
from athlete.update.update_rule import UpdateRule
from athlete.algorithms.dqn.update import DQNUpdate
from athlete.policy.policy import Policy
from athlete.algorithms.dqn.policy import DQNTrainingPolicy, DQNEvaluationPolicy
from athlete.data_collection.provider import UpdateDataProvider
from athlete.module.torch.common import FCDiscreteQValueFunction
from athlete import constants
from athlete.data_collection.collector import DataCollector
from athlete.data_collection.transition import GymnasiumTransitionDataCollector

# Deep Q-Network

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
ARGUMENT_DEVICE = "device"
ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS = "additional_replay_buffer_arguments"
ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING = "post_replay_buffer_data_preprocessing"

DEFAULT_CONFIGURATION = {
    ARGUMENT_DISCOUNT: 0.99,
    ARGUMENT_VALUE_NETWORK_CLASS: FCDiscreteQValueFunction,
    ARGUMENT_VALUE_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [256, 256],
        "init_state_dict_path": None,
    },
    ARGUMENT_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_OPTIMIZER_ARGUMENTS: {"lr": 6.3e-4},
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
    ARGUMENT_TARGET_NET_TAU: None,
    ARGUMENT_ENABLE_DOUBLE_Q_LEARNING: False,
    ARGUMENT_CRITERIA: torch.nn.MSELoss(),
    ARGUMENT_GRADIENT_MAX_NORM: None,
    ARGUMENT_DEVICE: (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS: {},
    ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING: None,
}


def make_dqn_components(
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
            constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: observation_space,
            constants.GENERAL_ARGUMENT_ACTION_SPACE: action_space,
        }
    )

    value_function = configuration[ARGUMENT_VALUE_NETWORK_CLASS](
        **configuration[ARGUMENT_VALUE_NETWORK_ARGUMENTS]
    )

    update_data_input = UpdateDataProvider()

    # DATA COLLECTOR

    data_collector = GymnasiumTransitionDataCollector(
        update_data_provider=update_data_input,
    )

    # UPDATE RULE
    update_rule = DQNUpdate(
        observation_space=observation_space,
        action_space=action_space,
        update_data_input=update_data_input,
        q_value_function=value_function,
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
        device=configuration[ARGUMENT_DEVICE],
        additional_replay_buffer_arguments=configuration[
            ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS
        ],
        post_replay_buffer_data_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    # POLICY

    training_policy = DQNTrainingPolicy(
        q_value_function=update_rule.q_value_function,
        action_space=action_space,
        start_epsilon=configuration[ARGUMENT_START_EPSILON],
        end_epsilon=configuration[ARGUMENT_END_EPSILON],
        epsilon_decay_steps=configuration[ARGUMENT_EPSILON_DECAY_STEPS],
        post_replay_buffer_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    evaluation_policy = DQNEvaluationPolicy(
        q_value_function=update_rule.q_value_function,
        post_replay_buffer_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    return data_collector, update_rule, training_policy, evaluation_policy


athlete.register(
    id="dqn",
    component_factory=make_dqn_components,
    default_configuration=DEFAULT_CONFIGURATION,
)
