from abc import ABC, abstractmethod
import math
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import Distribution, Normal
from gymnasium.spaces import Box

from athlete.module.torch.fully_connected import NonLinearFullyConnectedNet


class SACActor(ABC, torch.nn.Module):
    """An abstract interface for defining actor networks used in the Soft Actor-Critic algorithm.

    This class defines methods that are necessary for SAC's training process, including
    log-probability calculation and action sampling with reparameterization. Concrete
    implementations must define the specific network architecture and sampling methods.
    """

    def __init__(self):
        ABC.__init__(self)
        torch.nn.Module.__init__(self)

    @abstractmethod
    def get_action_and_log_prob(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns sampled actions and the according log probabilities of the actions
        given the observations.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled actions and their log probabilities
        """
        ...

    @abstractmethod
    def get_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Get the deterministic action mean given the observations.
            Might be useful for evaluation.

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            torch.Tensor: The deterministic action mean.
        """
        ...


class GaussianActor(SACActor):
    """A concrete implementation of the SACActor interface for a Gaussian actor.
    All actions are squeezed to the range [-1, 1] using the tanh function.
    Scaling should be done in the policy class. As inside the agent all actions are
    assumed to be in the range [-1, 1]."""

    ZERO_DIVISION_CONSTANT = 1e-6

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_dims: List[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_state_dict_path: str = None,
    ) -> None:
        """Initialize the gaussian SAC actor.

        Args:
            observation_space (Box): Observation space of the actor.
            action_space (Box): Action space of the actor.
            hidden_dims (List[int]): List of dimensions for the actor network, not including the input and output layers.
            log_std_min (float, optional): Minimum log standard deviation of the gaussian distribution. Defaults to -20.0.
            log_std_max (float, optional): Maximum log standard deviation of the gaussian distribution. Defaults to 2.0.
            init_state_dict_path (str, optional): Path to the initial state dict of the actor. Defaults to None.
        """
        SACActor.__init__(self)
        nn.Module.__init__(self)

        self.state_size = math.prod(observation_space.shape)
        action_size = math.prod(action_space.shape)
        self.log_std_scale = (log_std_max - log_std_min) / 2.0
        self.log_std_offset = (log_std_max + log_std_min) / 2.0

        network_kwargs = {
            "layer_dims": [
                self.state_size,
                *hidden_dims,
                2 * action_size,
            ]
        }
        self.actor_net = NonLinearFullyConnectedNet(**network_kwargs)

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def _get_normal_distribution(self, observations: torch.Tensor) -> Distribution:
        """Get the normal distribution of the actor given the observations.
            The mean and log_std are predicted by the actor network.
            The log_std is scaled to the range [log_std_min, log_std_max].

        Args:
            observations (torch.Tensor): The observations from the environment.

        Returns:
            Distribution: The normal distribution of the actor.
        """
        observations = observations.reshape(-1, self.state_size)
        mean, log_std = torch.chunk(self.actor_net(observations), 2, dim=1)
        # scaling log_std to the range [log_std_min, log_std_max]
        log_std = torch.tanh(log_std) * self.log_std_scale + self.log_std_offset
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Get the deterministic action mean given the observations.

        Args:
            observations (torch.Tensor): The observations from the environment of shape (batch_size, state_size).

        Returns:
            torch.Tensor: The deterministic action mean of shape (batch_size, action_size).
        """
        observations = observations.reshape(-1, self.state_size)
        mean, _ = torch.chunk(self.actor_net(observations), 2, dim=1)

        return torch.tanh(mean)

    def get_action_and_log_prob(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the sampled actions and their log probabilities given the observations.

        Args:
            observations (torch.Tensor): The observations from the environment of shape (batch_size, state_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled actions and their log probabilities
        of shape (batch_size, action_size) and (batch_size, 1) respectively.
        """
        normal_distribution = self._get_normal_distribution(observations)
        gaussian_action = normal_distribution.rsample()
        gaussian_log_prob = normal_distribution.log_prob(gaussian_action)

        action = torch.tanh(gaussian_action)
        log_prob = torch.sum(
            gaussian_log_prob
            - torch.log(1 - action.pow(2) + self.ZERO_DIVISION_CONSTANT),
            dim=1,
            keepdim=True,
        )

        return action, log_prob

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample an action given the observations.

        Args:
            observations (torch.Tensor): The observations from the environment of shape (batch_size, state_size).

        Returns:
            torch.Tensor: The sampled action of shape (batch_size, action_size).
        """
        normal_distribution = self._get_normal_distribution(observations)
        gaussian_action = normal_distribution.rsample()
        return torch.tanh(gaussian_action)
