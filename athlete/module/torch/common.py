import math

from typing import List, Optional
import torch
from gymnasium.spaces import Box, Space, Discrete

from athlete.module.torch.fully_connected import NonLinearFullyConnectedNet


class FCContinuousQValueFunction(torch.nn.Module):
    """A class for a fully connected continuous Q-value function. So taking in actions and observations and returning a Q-value."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_dims: List[int],
        init_state_dict_path: Optional[str] = None,
    ) -> None:
        """Initializes the continuous Q-value function.

        Args:
            observation_space (Box): Observation space of the Q-value function.
            action_space (Box): Action space of the Q-value function.
            hidden_dims (List[int]): List of hidden layer dimensions for the neural network. Does not include the input and output layer.
            init_state_dict_path (Optional[str], optional): Path to a state dict to load the model weights from. Defaults to None.
        """
        super(FCContinuousQValueFunction, self).__init__()

        self.observation_size = math.prod(observation_space.shape)
        self.action_size = math.prod(action_space.shape)

        self.evaluation_net = NonLinearFullyConnectedNet(
            layer_dims=[
                self.observation_size + self.action_size,
                *hidden_dims,
                1,
            ]
        )

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the Q-value function for a given observation and action batch.

        Args:
            observations (torch.Tensor): Observation batch of shape (batch_size, observation_size).
            actions (torch.Tensor): Action batch of shape (batch_size, action_size).

        Returns:
            torch.Tensor: Q-value batch of shape (batch_size, 1).
        """
        observations = observations.reshape(-1, self.observation_size)
        actions = actions.reshape(-1, self.action_size)

        state_actions = torch.hstack([observations, actions])

        return self.evaluation_net(state_actions)


class FCDiscreteQValueFunction(torch.nn.Module):
    """A class for a fully connected discrete Q-value function.
    So taking in observations and returning as many Q-values as there are discrete actions.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        hidden_dims: List[int],
        init_state_dict_path: Optional[str] = None,
    ) -> None:
        """Initializes the discrete Q-value function.

        Args:
            observation_space (Box): Observation space of the Q-value function.
            action_space (Discrete): Action space of the Q-value function.
            hidden_dims (List[int]): List of hidden layer dimensions for the neural network. Does not include the input and output layer.
            init_state_dict_path (Optional[str], optional): Path to a state dict to load the model weights from. Defaults to None.
        """
        super(FCDiscreteQValueFunction, self).__init__()

        self.observation_size = math.prod(observation_space.shape)
        self.num_actions = action_space.n

        self.evaluation_net = NonLinearFullyConnectedNet(
            layer_dims=[
                self.observation_size,
                *hidden_dims,
                self.num_actions,
            ]
        )

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluates the Q-value function for a given observation batch.

        Args:
            observations (torch.Tensor): Observation batch of shape (batch_size, observation_size).

        Returns:
            torch.Tensor: Q-value batch of shape (batch_size, num_actions).
        """
        observations = observations.reshape(-1, self.observation_size)
        return self.evaluation_net(observations)


class FCDeterministicActor(torch.nn.Module):
    """A class for a fully connected deterministic actor.
    So taking in observations and returning deterministic actions.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_dims: List[int],
        squash_action: bool = True,
        init_state_dict_path: Optional[str] = None,
    ) -> None:
        """Initializes the deterministic actor.

        Args:
            observation_space (Box): Observation space of the actor.
            action_space (Box): Action space of the actor.
            hidden_dims (List[int]): List of hidden layer dimensions for the neural network. Does not include the input and output layer.
            squash_action (bool, optional): Whether to squash the action to be between -1 and 1. Defaults to True.
            init_state_dict_path (Optional[str], optional): Path to a state dict to load the model weights from. Defaults to None.
        """
        super().__init__()

        self.state_size = math.prod(observation_space.shape)
        self.action_size = math.prod(action_space.shape)

        network_kwargs = {
            "layer_dims": [
                self.state_size,
                *hidden_dims,
                self.action_size,
            ]
        }
        if squash_action:
            network_kwargs["final_activation"] = torch.nn.Tanh()

        self.actor_net = NonLinearFullyConnectedNet(**network_kwargs)

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns an action based on the given observation.

        Args:
            observations (torch.Tensor): Observation batch of shape (batch_size, state_size).

        Returns:
            torch.Tensor: Action batch of shape (batch_size, action_size).
        """
        observations = observations.reshape(-1, self.state_size)
        return self.actor_net(observations)
