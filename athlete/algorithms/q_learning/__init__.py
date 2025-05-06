from typing import Dict, Any, Tuple

from gymnasium.spaces import Space, Discrete

import athlete
from athlete.data_collection.collector import DataCollector
from athlete.data_collection.transition import GymnasiumTransitionDataCollector
from athlete.update.update_rule import UpdateRule
from athlete.algorithms.q_learning.update import QLearningUpdate
from athlete.policy.policy_builder import PolicyBuilder
from athlete.algorithms.q_learning.policy import QLearningPolicyBuilder
from athlete.data_collection.provider import UpdateDataProvider
from athlete import constants

# Q-Learning

ARGUMENT_DISCOUNT = "discount"
ARGUMENT_LEARNING_RATE = "learning_rate"
ARGUMENT_START_EPSILON = "start_epsilon"
ARGUMENT_END_EPSILON = "end_epsilon"
ARGUMENT_EPSILON_DECAY_STEPS = "epsilon_decay_steps"

DEFAULT_CONFIGURATION = {
    ARGUMENT_DISCOUNT: 0.9,
    ARGUMENT_LEARNING_RATE: 0.1,
    ARGUMENT_START_EPSILON: 1.0,
    ARGUMENT_END_EPSILON: 0.1,
    constants.GENERAL_ARGUMENT_WARMUP_STEPS: 0,
    ARGUMENT_EPSILON_DECAY_STEPS: 100,
}


def make_q_learning_components(
    observation_space: Space, action_space: Space, configuration: Dict[str, Any]
) -> Tuple[DataCollector, UpdateRule, PolicyBuilder]:
    """Creates the components for a Q-learning agent.

    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        configuration: Algorithm configuration dictionary

    Returns:
        Tuple containing data collector, update rule and policy builder

    Raises:
        ValueError: If observation_space or action_space is not Discrete
    """

    if not isinstance(observation_space, Discrete):
        raise ValueError(
            f"This Q-Learning implementation only supports {Discrete.__name__} observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Discrete):
        raise ValueError(
            f"This Q-Learning implementation only supports {Discrete.__name__} action spaces, but got {type(action_space)}"
        )

    update_data_provider = UpdateDataProvider()

    # DATA COLLECTOR
    data_collector = GymnasiumTransitionDataCollector(
        update_data_provider=update_data_provider,
    )

    # UPDATE RULE

    update_rule = QLearningUpdate(
        observation_space=observation_space,
        action_space=action_space,
        update_data_provider=update_data_provider,
        discount=configuration[ARGUMENT_DISCOUNT],
        learning_rate=configuration[ARGUMENT_LEARNING_RATE],
    )

    # POLICY BUILDER

    policy_builder = QLearningPolicyBuilder(
        q_table=update_rule.q_table,
        action_space=action_space,
        start_epsilon=configuration[ARGUMENT_START_EPSILON],
        end_epsilon=configuration[ARGUMENT_END_EPSILON],
        epsilon_decay_steps=configuration[ARGUMENT_EPSILON_DECAY_STEPS],
    )

    return data_collector, update_rule, policy_builder


athlete.register(
    id="q_learning",
    component_factory=make_q_learning_components,
    default_configuration=DEFAULT_CONFIGURATION,
)
