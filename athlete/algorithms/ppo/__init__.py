from typing import Dict, Any, Tuple

import torch
from gymnasium.spaces import Space, Box

import athlete
from athlete.data_collection.collector import DataCollector
from athlete.update.update_rule import UpdateRule
from athlete.policy.policy import Policy
from athlete import constants
from athlete.algorithms.ppo.module import GaussianActor, FCPPOValueFunction
from athlete.data_collection.provider import UpdateDataProvider
from athlete.data_collection.on_policy import OnPolicyDataCollector
from athlete.algorithms.ppo.policy import (
    INFO_KEY_LOG_PROB,
    PPOTrainingPolicy,
    PPOEvaluationPolicy,
)
from athlete.algorithms.ppo.update import PPOUpdate

# Proximal Policy Optimization

ARGUMENT_DISCOUNT = "discount"
ARGUMENT_VALUE_NETWORK_CLASS = "value_network_class"
ARGUMENT_VALUE_NETWORK_ARGUMENTS = "value_network_arguments"
ARGUMENT_ACTOR_NETWORK_CLASS = "actor_network_class"
ARGUMENT_ACTOR_NETWORK_ARGUMENTS = "actor_network_arguments"
ARGUMENT_STEPS_PER_UPDATE = "steps_per_update"
ARGUMENT_MINI_BATCH_SIZE = "mini_batch_size"
ARGUMENT_OPTIMIZER_CLASS = "optimizer_class"
ARGUMENT_OPTIMIZER_ARGUMENTS = "optimizer_arguments"
ARGUMENT_GENERALIZED_ADVANTAGE_ESTIMATION_LAMBDA = (
    "generalized_advantage_estimation_lambda"
)
ARGUMENT_EPOCHS_PER_UPDATE = "epochs_per_update"
ARGUMENT_POLICY_RATIO_CLIP_VALUE = "policy_ratio_clip_value"
ARGUMENT_VALUE_LOSS_CLIP_VALUE = "value_loss_clip_value"
ARGUMENT_VALUE_LOSS_COEFFICIENT = "value_loss_coefficient"
ARGUMENT_ENTROPY_LOSS_COEFFICIENT = "entropy_loss_coefficient"
ARGUMENT_BATCH_NORMALIZE_ADVANTAGE = "batch_normalize_advantage"
ARGUMENT_GRADIENT_MAX_NORM = "gradient_max_norm"
ARGUMENT_DEVICE = "device"

DEFAULT_CONFIGURATION = {
    ARGUMENT_DISCOUNT: 0.99,
    ARGUMENT_VALUE_NETWORK_CLASS: FCPPOValueFunction,
    ARGUMENT_VALUE_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [64, 64],
        "init_state_dict_path": None,
    },
    ARGUMENT_ACTOR_NETWORK_CLASS: GaussianActor,
    ARGUMENT_ACTOR_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [64, 64],
        "init_state_dict_path": None,
    },
    ARGUMENT_STEPS_PER_UPDATE: 1024,
    ARGUMENT_MINI_BATCH_SIZE: 256,
    ARGUMENT_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_OPTIMIZER_ARGUMENTS: {
        "lr": 3e-4,
        "eps": 1e-5,
    },
    ARGUMENT_GENERALIZED_ADVANTAGE_ESTIMATION_LAMBDA: 0.95,
    ARGUMENT_EPOCHS_PER_UPDATE: 10,
    ARGUMENT_POLICY_RATIO_CLIP_VALUE: 0.2,
    ARGUMENT_VALUE_LOSS_CLIP_VALUE: None,
    ARGUMENT_VALUE_LOSS_COEFFICIENT: 0.5,
    ARGUMENT_ENTROPY_LOSS_COEFFICIENT: 0.0,
    ARGUMENT_BATCH_NORMALIZE_ADVANTAGE: True,
    ARGUMENT_GRADIENT_MAX_NORM: 0.5,
    ARGUMENT_DEVICE: (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
}


def make_ppo_components(
    observation_space: Space,
    action_space: Space,
    configuration: Dict[str, Any],
) -> Tuple[DataCollector, UpdateRule, Policy, Policy]:
    """Creates the components for a PPO agent.

    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        configuration: Algorithm configuration dictionary

    Returns:
        Tuple containing data collector, update rule, training policy, and evaluation policy

    Raises:
        ValueError: If observation_space or action_space is not Box
    """

    if not isinstance(observation_space, Box):
        raise ValueError(
            f"This PPO implementation only supports {Box.__name__} observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Box):
        raise ValueError(
            f"This PPO implementation only supports {Box.__name__} action spaces, but got {type(action_space)}"
        )

    configuration[ARGUMENT_VALUE_NETWORK_ARGUMENTS].update(
        {constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: observation_space}
    )
    configuration[ARGUMENT_ACTOR_NETWORK_ARGUMENTS].update(
        {
            constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: observation_space,
            constants.GENERAL_ARGUMENT_ACTION_SPACE: action_space,
        }
    )

    value_network = configuration[ARGUMENT_VALUE_NETWORK_CLASS](
        **configuration[ARGUMENT_VALUE_NETWORK_ARGUMENTS]
    )
    actor_network = configuration[ARGUMENT_ACTOR_NETWORK_CLASS](
        **configuration[ARGUMENT_ACTOR_NETWORK_ARGUMENTS]
    )

    update_data_provider = UpdateDataProvider()

    # DATA COLLECTOR
    data_collector = OnPolicyDataCollector(
        observation_space=observation_space,
        action_space=action_space,
        update_data_provider=update_data_provider,
        number_of_collection_steps=configuration[ARGUMENT_STEPS_PER_UPDATE],
        policy_info_log_prob_key=INFO_KEY_LOG_PROB,
    )

    # UPDATE RULE
    update_rule = PPOUpdate(
        update_data_provider=update_data_provider,
        value_function=value_network,
        actor=actor_network,
        optimizer_class=configuration[ARGUMENT_OPTIMIZER_CLASS],
        optimizer_arguments=configuration[ARGUMENT_OPTIMIZER_ARGUMENTS],
        discount=configuration[ARGUMENT_DISCOUNT],
        generalized_advantage_estimation_lambda=configuration[
            ARGUMENT_GENERALIZED_ADVANTAGE_ESTIMATION_LAMBDA
        ],
        epochs_per_update=configuration[ARGUMENT_EPOCHS_PER_UPDATE],
        mini_batch_size=configuration[ARGUMENT_MINI_BATCH_SIZE],
        policy_ratio_clip_value=configuration[ARGUMENT_POLICY_RATIO_CLIP_VALUE],
        value_loss_clip_value=configuration[ARGUMENT_VALUE_LOSS_CLIP_VALUE],
        value_loss_coefficient=configuration[ARGUMENT_VALUE_LOSS_COEFFICIENT],
        entropy_loss_coefficient=configuration[ARGUMENT_ENTROPY_LOSS_COEFFICIENT],
        batch_normalize_advantage=configuration[ARGUMENT_BATCH_NORMALIZE_ADVANTAGE],
        gradient_max_norm=configuration[ARGUMENT_GRADIENT_MAX_NORM],
        device=configuration[ARGUMENT_DEVICE],
    )

    # POLICY

    training_policy = PPOTrainingPolicy(
        actor=actor_network,
        action_space=action_space,
    )

    evaluation_policy = PPOEvaluationPolicy(
        actor=actor_network,
        action_space=action_space,
    )

    return data_collector, update_rule, training_policy, evaluation_policy


athlete.register(
    id="ppo",
    component_factory=make_ppo_components,
    default_configuration=DEFAULT_CONFIGURATION,
)
