from typing import Dict, Any, Tuple

from gymnasium.spaces import Space, Box
import torch

import athlete
from athlete.data_collection.collector import DataCollector
from athlete.data_collection.transition import (
    ActionReplacementGymnasiumTransitionDataCollector,
)
from athlete.update.update_rule import UpdateRule
from athlete.policy.policy_builder import PolicyBuilder
from athlete.algorithms.sac.update import SACUpdate
from athlete.data_collection.provider import UpdateDataProvider
from athlete.algorithms.sac.module import GaussianActor
from athlete.module.torch.common import FCContinuousQValueFunction
from athlete.algorithms.sac.policy import INFO_KEY_UNSCALED_ACTION, SACPolicyBuilder
from athlete import constants

# Soft Actor-Critic

ARGUMENT_DISCOUNT = "discount"
ARGUMENT_CRITIC_1_NETWORK_CLASS = "critic_1_network_class"
ARGUMENT_CRITIC_1_NETWORK_ARGUMENTS = "critic_network_arguments"
ARGUMENT_CRITIC_2_NETWORK_CLASS = "critic_2_network_class"
ARGUMENT_CRITIC_2_NETWORK_ARGUMENTS = "critic_2_network_arguments"
ARGUMENT_ACTOR_NETWORK_CLASS = "actor_network_class"
ARGUMENT_ACTOR_NETWORK_ARGUMENTS = "actor_network_arguments"
ARGUMENT_TEMPERATURE = "temperature"
ARGUMENT_TARGET_ENTROPY = "target_entropy"
ARGUMENT_INITIAL_TEMPERATURE = "initial_temperature"
ARGUMENT_CRITIC_OPTIMIZER_CLASS = "critic_optimizer_class"
ARGUMENT_CRITIC_OPTIMIZER_ARGUMENTS = "critic_optimizer_arguments"
ARGUMENT_ACTOR_OPTIMIZER_CLASS = "actor_optimizer_class"
ARGUMENT_ACTOR_OPTIMIZER_ARGUMENTS = "actor_optimizer_arguments"
ARGUMENT_REPLAY_BUFFER_CAPACITY = "replay_buffer_capacity"
ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE = "replay_buffer_mini_batch_size"

ARGUMENT_CRITIC_UPDATE_FREQUENCY = "critic_update_frequency"
ARGUMENT_CRITIC_NUMBER_OF_UPDATES = "critic_number_of_updates"
ARGUMENT_ACTOR_UPDATE_FREQUENCY = "actor_update_frequency"
ARGUMENT_ACTOR_NUMBER_OF_UPDATES = "actor_number_of_updates"
ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS = (
    "multiply_number_of_updates_by_environment_steps"
)
ARGUMENT_TARGET_CRITIC_UPDATE_FREQUENCY = "target_critic_update_frequency"
ARGUMENT_TARGET_CRITIC_TAU = "target_critic_tau"
ARGUMENT_CRITIC_CRITERIA = "critic_criteria"
ARGUMENT_CRITIC_GRADIENT_MAX_NORM = "critic_gradient_max_norm"
ARGUMENT_ACTOR_GRADIENT_MAX_NORM = "actor_gradient_max_norm"
ARGUMENT_DEVICE = "device"
ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS = "additional_replay_buffer_arguments"
ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING = "post_replay_buffer_data_preprocessing"

DEFAULT_CONFIGURATION = {
    ARGUMENT_DISCOUNT: 0.99,
    ARGUMENT_CRITIC_1_NETWORK_CLASS: FCContinuousQValueFunction,
    ARGUMENT_CRITIC_1_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "init_state_dict_path": None,
    },
    ARGUMENT_CRITIC_2_NETWORK_CLASS: FCContinuousQValueFunction,
    ARGUMENT_CRITIC_2_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "init_state_dict_path": None,
    },
    ARGUMENT_ACTOR_NETWORK_CLASS: GaussianActor,
    ARGUMENT_ACTOR_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "log_std_min": -20.0,
        "log_std_max": 2.0,
        "init_state_dict_path": None,
    },
    ARGUMENT_TEMPERATURE: SACUpdate.SETTING_AUTO,
    ARGUMENT_TARGET_ENTROPY: SACUpdate.SETTING_AUTO,
    ARGUMENT_INITIAL_TEMPERATURE: 1.0,
    ARGUMENT_CRITIC_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_CRITIC_OPTIMIZER_ARGUMENTS: {"lr": 3e-4},
    ARGUMENT_ACTOR_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_ACTOR_OPTIMIZER_ARGUMENTS: {"lr": 3e-4},
    ARGUMENT_REPLAY_BUFFER_CAPACITY: 1000000,
    ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE: 256,
    constants.GENERAL_ARGUMENT_WARMUP_STEPS: 1000,
    ARGUMENT_CRITIC_UPDATE_FREQUENCY: 1,
    ARGUMENT_CRITIC_NUMBER_OF_UPDATES: 1,
    ARGUMENT_ACTOR_UPDATE_FREQUENCY: 1,
    ARGUMENT_ACTOR_NUMBER_OF_UPDATES: 1,
    ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS: False,
    ARGUMENT_TARGET_CRITIC_UPDATE_FREQUENCY: 1,
    ARGUMENT_TARGET_CRITIC_TAU: 0.005,
    ARGUMENT_CRITIC_CRITERIA: torch.nn.MSELoss(),
    ARGUMENT_CRITIC_GRADIENT_MAX_NORM: None,
    ARGUMENT_ACTOR_GRADIENT_MAX_NORM: None,
    ARGUMENT_DEVICE: (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS: {},
    ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING: None,
}


def make_sac_components(
    observation_space: Space, action_space: Space, configuration: Dict[str, Any]
) -> Tuple[DataCollector, UpdateRule, PolicyBuilder]:
    """Creates the components for a SAC agent.

    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        configuration: Algorithm configuration dictionary

    Returns:
        Tuple containing data collector, update rule and policy builder

    Raises:
        ValueError: If observation_space or action_space is not Box
    """

    if not isinstance(observation_space, Box):
        raise ValueError(
            f"This SAC implementation only supports {Box.__name__} observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Box):
        raise ValueError(
            f"This SAC implementation only supports {Box.__name__} action spaces, but got {type(action_space)}"
        )

    environment_info = {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: observation_space,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: action_space,
    }

    configuration[ARGUMENT_CRITIC_1_NETWORK_ARGUMENTS].update(environment_info)
    configuration[ARGUMENT_CRITIC_2_NETWORK_ARGUMENTS].update(environment_info)

    critic_1 = configuration[ARGUMENT_CRITIC_1_NETWORK_CLASS](
        **configuration[ARGUMENT_CRITIC_1_NETWORK_ARGUMENTS]
    )
    critic_2 = configuration[ARGUMENT_CRITIC_2_NETWORK_CLASS](
        **configuration[ARGUMENT_CRITIC_2_NETWORK_ARGUMENTS]
    )

    configuration[ARGUMENT_ACTOR_NETWORK_ARGUMENTS].update(environment_info)

    actor = configuration[ARGUMENT_ACTOR_NETWORK_CLASS](
        **configuration[ARGUMENT_ACTOR_NETWORK_ARGUMENTS]
    )

    update_data_provider = UpdateDataProvider()

    # DATA COLLECTOR
    # This data receiver uses the unscaled action from the policy to replace the action in the transition data.
    data_collector = ActionReplacementGymnasiumTransitionDataCollector(
        policy_info_replacement_key=INFO_KEY_UNSCALED_ACTION,
        update_data_provider=update_data_provider,
    )

    # UPDATE RULE
    update_rule = SACUpdate(
        observation_space=observation_space,
        action_space=action_space,
        update_data_provider=update_data_provider,
        critic_1=critic_1,
        critic_2=critic_2,
        actor=actor,
        discount=configuration[ARGUMENT_DISCOUNT],
        temperature=configuration[ARGUMENT_TEMPERATURE],
        target_entropy=configuration[ARGUMENT_TARGET_ENTROPY],
        initial_temperature=configuration[ARGUMENT_INITIAL_TEMPERATURE],
        critic_optimizer_arguments=configuration[ARGUMENT_CRITIC_OPTIMIZER_ARGUMENTS],
        actor_optimizer_arguments=configuration[ARGUMENT_ACTOR_OPTIMIZER_ARGUMENTS],
        critic_optimizer_class=configuration[ARGUMENT_CRITIC_OPTIMIZER_CLASS],
        actor_optimizer_class=configuration[ARGUMENT_ACTOR_OPTIMIZER_CLASS],
        critic_criteria=configuration[ARGUMENT_CRITIC_CRITERIA],
        replay_buffer_capacity=configuration[ARGUMENT_REPLAY_BUFFER_CAPACITY],
        replay_buffer_mini_batch_size=configuration[
            ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE
        ],
        multiply_number_of_updates_by_environment_steps=configuration[
            ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS
        ],
        critic_update_frequency=configuration[ARGUMENT_CRITIC_UPDATE_FREQUENCY],
        critic_number_of_updates=configuration[ARGUMENT_CRITIC_NUMBER_OF_UPDATES],
        actor_update_frequency=configuration[ARGUMENT_ACTOR_UPDATE_FREQUENCY],
        actor_number_of_updates=configuration[ARGUMENT_ACTOR_NUMBER_OF_UPDATES],
        target_critic_update_frequency=configuration[
            ARGUMENT_TARGET_CRITIC_UPDATE_FREQUENCY
        ],
        target_critic_tau=configuration[ARGUMENT_TARGET_CRITIC_TAU],
        critic_gradient_max_norm=configuration[ARGUMENT_CRITIC_GRADIENT_MAX_NORM],
        actor_gradient_max_norm=configuration[ARGUMENT_ACTOR_GRADIENT_MAX_NORM],
        device=configuration[ARGUMENT_DEVICE],
        additional_replay_buffer_arguments=configuration[
            ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS
        ],
        post_replay_buffer_data_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    # POLICY BUILDER
    policy_builder = SACPolicyBuilder(
        actor=update_rule.actor,
        action_space=action_space,
        post_replay_buffer_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
    )

    return data_collector, update_rule, policy_builder


athlete.register(
    id="sac",
    component_factory=make_sac_components,
    default_configuration=DEFAULT_CONFIGURATION,
)
