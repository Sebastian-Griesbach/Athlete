from typing import Any, Dict, Tuple
import numpy as np
import torch
from gymnasium.spaces import Box

from athlete.function import numpy_to_tensor, tensor_to_numpy
from athlete.policy.policy import Policy
from athlete.algorithms.ppo.module import PPOActor

INFO_KEY_LOG_PROB = "log_prob"


class PPOTrainingPolicy(Policy):
    """Training policy implementation for PPO algorithm.

    This policy uses a PPO actor network to sample actions during training and record log probabilities of taken actions.
    """

    def __init__(
        self,
        actor: PPOActor,
        action_space: Box,
    ) -> None:
        """Initializes the PPO training policy.

        Args:
            actor (PPOActor): The PPO actor to use for action selection.
            action_space (Box): The action space of the actor.
        """
        self.actor = actor
        self.action_space = action_space

        self.action_high = action_space.high
        self.action_low = action_space.low

        self.module_device = next(self.actor.parameters()).device

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Samples an action from the actor given the observation.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: A tuple containing the sampled action and a dictionary containing the log probability of the action.
        """
        observation = np.expand_dims(observation, axis=0)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(observation)
        action = tensor_to_numpy(action).squeeze(axis=0)
        log_prob = tensor_to_numpy(log_prob).squeeze(axis=0)

        return action, {INFO_KEY_LOG_PROB: log_prob}


class PPOEvaluationPolicy(Policy):
    """The PPO evaluation policy.
    This policy uses the actor to get the deterministic action mean.
    """

    def __init__(
        self,
        actor: PPOActor,
        action_space: Box,
    ) -> None:
        """Initializes the PPO evaluation policy.

        Args:
            actor (PPOActor): The PPO actor to use for action selection.
            action_space (Box): The action space of the actor.
        """
        self.actor = actor
        self.action_space = action_space

        self.action_high = action_space.high
        self.action_low = action_space.low

        self.module_device = next(self.actor.parameters()).device

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Get the deterministic action mean from the actor given the observation.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: A tuple containing the deterministic action mean and an empty dictionary.
        """
        observation = np.expand_dims(observation, axis=0)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            action = self.actor.get_mean(observation)
        action = tensor_to_numpy(action).squeeze(axis=0)

        return action, {}
