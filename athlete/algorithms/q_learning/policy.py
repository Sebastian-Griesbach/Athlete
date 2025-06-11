from typing import Any, Dict, Tuple

import numpy as np
import torch
from gymnasium.spaces import Discrete

from athlete.global_objects import StepTracker, RNGHandler
from athlete.policy.policy import Policy


class QLearningTrainingPolicy(Policy):
    """Training policy implementation for tabular Q-Learning algorithm.

    This policy implements an epsilon-greedy exploration strategy for discrete state and action
    spaces.
    """

    def __init__(
        self,
        q_table: np.ndarray,
        action_space: Discrete,
        start_epsilon: float,
        end_epsilon: float,
        epsilon_decay_steps: int,
    ) -> None:
        """Initializes the Q-Learning training policy.

        Args:
            q_table (np.ndarray): The Q-table to use for action selection.
            action_space (Discrete): The action space of the environment.
            start_epsilon (float): The starting value of epsilon for the decaying epsilon-greedy strategy.
            end_epsilon (float): The minimum value of epsilon for the decaying epsilon-greedy strategy.
            epsilon_decay_steps (int): The number of steps over which epsilon decays linearly from start_epsilon to end_epsilon.
        """
        self.q_table = q_table
        self.num_actions = action_space.n
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.step_tracker = StepTracker.get_instance()
        self.random_number_generator = RNGHandler.get_random_number_generator()

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.epsilon_delta = (start_epsilon - end_epsilon) / epsilon_decay_steps

    def _get_random_action(self) -> int:
        """Returns a random action from the action space in a reproducible way.

        Returns:
            int: Random action sampled from the action space.
        """
        # We don't use action_space.sample() to ensure reproducibility
        return self.random_number_generator.integers(low=0, high=self.num_actions)

    def _select_greedy_action(self, observation: int) -> int:
        """Selects the action with the maximum Q-value for the given observation,
        choosing randomly among actions with the same maximum Q-value.

        Args:
            observation (int): The observation for which to select the action.

        Returns:
            int: The chosen action.
        """
        q_values = self.q_table[observation]
        max_q_value = np.max(q_values)
        # Find all actions with the maximum Q-value
        max_actions = np.where(q_values == max_q_value)[0]
        # Randomly select one of these actions
        if len(max_actions) == 1:
            return max_actions[0]
        else:
            return self.random_number_generator.choice(max_actions)

    def act(self, observation: int) -> Tuple[int, Dict[str, Any]]:
        """Select an action based on the observation using the epsilon-greedy strategy.
        Respects a warmup period where only random actions are taken. The number of warmup steps is defined in the step tracker.

        Args:
            observation (int): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: The chosen action and a dictionary with additional information: the current epsilon value and whether the action was chosen greedily.
        """

        if not self.step_tracker.is_warmup_done:
            return self._get_random_action(), {
                "greedy": False,
                "epsilon": self.start_epsilon,
            }

        epsilon_threshold = max(
            self.end_epsilon,
            self.start_epsilon
            - self.epsilon_delta * self.step_tracker.interactions_after_warmup,
        )
        if self.random_number_generator.random() < epsilon_threshold:
            return self._get_random_action(), {
                "greedy": False,
                "epsilon": epsilon_threshold,
            }

        action = self._select_greedy_action(observation)

        return action, {"greedy": True, "epsilon": epsilon_threshold}


class QLearningEvaluationPolicy(Policy):
    """The Q-Learning evaluation policy.
    This policy selects the action with the maximum Q-value for the given observation,

    """

    def __init__(
        self,
        q_table: torch.nn.Module,
    ) -> None:
        """Initializes the Q-Learning evaluation policy.
        This policy implements the greedy action selection strategy

        Args:
            q_table (torch.nn.Module): The Q-table to use for action selection.
        """
        self.q_table = q_table
        self.random_number_generator = RNGHandler.get_random_number_generator()

    def act(self, observation: int) -> Tuple[int, Dict[str, Any]]:
        """Select an action based on the observation using the greedy strategy.

        Args:
            observation (int): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: The chosen action and an empty dictionary.
        """
        q_values = self.q_table[observation]
        max_q_value = np.max(q_values)
        max_actions = np.where(q_values == max_q_value)[0]
        if len(max_actions) == 1:
            action = max_actions[0]
        else:
            action = self.random_number_generator.choice(max_actions)
        return action, {}
