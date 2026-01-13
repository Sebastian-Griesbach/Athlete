import math
from typing import List, Tuple
from abc import ABC, abstractmethod

import torch
from torch.distributions import Distribution, Normal
from gymnasium.spaces import Box
import numpy as np

from athlete.module.torch.fully_connected import NonLinearFullyConnectedNet


# Ruthlessly stolen from https://github.com/vwxyzjn/cleanrl/blob/e648ee2dc8960c59ed3ee6caf9eb0c34b497958f/cleanrl/ppo.py#L94
def layer_init(
    layer: torch.nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> torch.nn.Linear:
    """An initialization function for linear layers commonly used for PPO Actors and Value Functions.

    Args:
        layer (torch.nn.Linear): The layer to be initialized.
        std (float, optional): Standard deviation for the orthogonal initialization. Defaults to np.sqrt(2).
        bias_const (float, optional): Constant value for the bias initialization. Defaults to 0.0.

    Returns:
        torch.nn.Linear: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActor(ABC, torch.nn.Module):
    """An abstract interface defining Actor networks for Proximal Policy Optimization algorithm.

    This class defines the methods required for implementing PPO-compatible actor networks,
    including methods for evaluating action distributions, calculating log probabilities,
    and measuring entropy. Concrete implementations should provide specific network
    architectures and probability distribution handling.
    """

    def __init__(self):
        ABC.__init__(self)
        torch.nn.Module.__init__(self)

    @abstractmethod
    def get_action_and_log_prob(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and log probability of the action given the observations.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled action and its log probability.
        """
        ...

    @abstractmethod
    def get_log_prob_and_entropy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given the observations and actions, return the log probability and entropy of the actions.

        Args:
            observations (torch.Tensor): The revenant observations.
            actions (torch.Tensor): The relevant actions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the log probability of the actions and their entropy.
        """
        ...

    @abstractmethod
    def get_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Get the deterministic action mean given the observations.
            Might be used for evaluation.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            torch.Tensor: The deterministic action mean.
        """
        ...


class GaussianActor(PPOActor):
    """A concrete implementation of the PPOActor interface for a Gaussian actor."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_dims: List[int],
        init_state_dict_path: str = None,
    ) -> None:
        """Initializes the Gaussian PPO actor.

        Args:
            observation_space (Box): Observation space of the actor.
            action_space (Box): Action space of the actor.
            hidden_dims (List[int]): List of hidden dimensions for the neural network not including the input and output layer.
            init_state_dict_path (str, optional): Path to the initial state dictionary for loading pre-trained weights. Defaults to None.
        """
        super(PPOActor, self).__init__()
        self.observation_size = math.prod(observation_space.shape)
        action_size = math.prod(action_space.shape)

        layer_initialization_functions = [lambda layer: layer_init(layer)] * len(
            hidden_dims
        )
        layer_initialization_functions.append(lambda layer: layer_init(layer, std=0.01))

        self.action_mean_net = NonLinearFullyConnectedNet(
            layer_dims=[
                self.observation_size,
                *hidden_dims,
                action_size,
            ],
            activation=torch.nn.Tanh(),
            layer_initialization_functions=layer_initialization_functions,
        )

        self.action_log_std = torch.nn.Parameter(torch.zeros(1, action_size))

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def get_mean(self, observations):
        observations = observations.reshape(-1, self.observation_size)
        mean = self.action_mean_net(observations)
        return mean

    def _get_normal_distribution(self, observations: torch.Tensor) -> Distribution:
        """Get the normal distribution for the given observations.
        This is used to sample actions and calculate log probabilities.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            Distribution: The normal distribution parameterized by the mean and standard deviation.
        """
        observations = observations.reshape(-1, self.observation_size)
        mean = self.action_mean_net(observations)
        action_log_std = self.action_log_std.expand_as(mean)
        action_std = torch.exp(action_log_std)
        return Normal(mean, action_std)

    def get_action_and_log_prob(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normal_distribution = self._get_normal_distribution(observations)
        action = normal_distribution.rsample()
        log_prob = normal_distribution.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def get_log_prob_and_entropy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normal_distribution = self._get_normal_distribution(observations)
        log_prob = normal_distribution.log_prob(actions)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        entropy = normal_distribution.entropy()
        entropy = entropy.sum(dim=1, keepdim=True)
        return log_prob, entropy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample an action given the observations.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            torch.Tensor: The sampled action.
        """
        normal_distribution = self._get_normal_distribution(observations)
        action = normal_distribution.rsample()
        return action


class FCPPOValueFunction(torch.nn.Module):
    """A fully connected value function for with layer initialization as commonly used in PPO."""

    def __init__(
        self,
        observation_space: Box,
        hidden_dims: List[int],
        init_state_dict_path: str = None,
    ) -> None:
        """Initializes the fully connected value function.

        Args:
            observation_space (Box): Observation space of the value function.
            hidden_dims (List[int]): Hidden dimensions for the neural network not including the input and output layer.
            init_state_dict_path (str, optional): Path to the initial state dictionary for loading pre-trained weights. Defaults to None.
        """
        super(FCPPOValueFunction, self).__init__()

        self.observation_size = math.prod(observation_space.shape)

        layer_initialization_functions = [lambda layer: layer_init(layer)] * len(
            hidden_dims
        )
        layer_initialization_functions.append(lambda layer: layer_init(layer, std=1.0))

        self.evaluation_net = NonLinearFullyConnectedNet(
            layer_dims=[
                self.observation_size,
                *hidden_dims,
                1,
            ],
            activation=torch.nn.Tanh(),
            layer_initialization_functions=layer_initialization_functions,
        )

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the value estimate for the given observations.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            torch.Tensor: The value estimate for the given observations.
        """
        observations = observations.reshape(-1, self.observation_size)
        return self.evaluation_net(observations)
