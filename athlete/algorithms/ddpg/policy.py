from typing import Callable, Any, Optional, Dict, Tuple
import numpy as np
import torch
from gymnasium.spaces import Box

from athlete.function import numpy_to_tensor, tensor_to_numpy
from athlete.global_objects import StepTracker, RNGHandler
from athlete.policy.policy import Policy
from athlete.policy.noise import NoiseProcess

INFO_KEY_UNSCALED_ACTION = "unscaled_action"


class DDPGTrainingPolicy(Policy):
    """Training policy implementation for Deep Deterministic Policy Gradient algorithm.

    This policy combines a deterministic actor network with a noise process used for exploration.
    Actions are appropriately scaled to match the environment's action space.
    """

    def __init__(
        self,
        noise_process: NoiseProcess,
        actor: torch.nn.Module,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]],
    ) -> None:
        """Initialize the DDPG training policy.

        Args:
            noise_process (NoiseProcess): The noise process to use for exploration.
            actor (torch.nn.Module): The actor network to use for action selection.
            action_space (Box): The action space of the actor.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]]):
                A function to preprocess the observation before passing it to the actor.
        """
        self.noise_process = noise_process
        self.actor = actor
        self.action_space = action_space
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.action_scales = (action_space.high - action_space.low) / 2
        self.action_offsets = (action_space.high + action_space.low) / 2

        self.unscaled_high = np.ones(self.action_space.shape)
        self.unscaled_low = -np.ones(self.action_space.shape)

        self.module_device = next(self.actor.parameters()).device
        self.step_tracker = StepTracker.get_instance()
        self.random_number_generator = RNGHandler.get_random_number_generator()

    def reset_act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Resets the noise process and returns the action for the given observation.

        Args:
            observation (np.ndarray): Initial observation of the new episode.

        Returns:
            Tuple[int, Dict[str, Any]]: The action chosen by the policy and the policy info containing
                the unscaled action.
        """
        self.noise_process.reset()
        return self.act(observation=observation)

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Returns the action for the given observation.
        This method is called during training and adds noise to the action for exploration.
        It also scales the action to the action space of the environment.
        If the warmup is not done, it returns a random action in the action space.

        Args:
            observation (np.ndarray): The observation to use for action selection.

        Returns:
            Tuple[int, Dict[str, Any]]: The action chosen by the policy and the policy info containing
                the unscaled action.
        """

        if not self.step_tracker.is_warmup_done:
            random_action = self.random_number_generator.random(
                size=self.action_space.shape
            )
            random_scaled_action = (
                random_action * 2 - 1  # This is to scale the action to [-1, 1]
            ) * self.action_scales + self.action_offsets
            return random_scaled_action, {INFO_KEY_UNSCALED_ACTION: random_action}

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            action = self.actor(observation)
        action = tensor_to_numpy(action).squeeze(axis=0)
        noise = self.noise_process.sample()
        action += noise
        action = np.clip(action, self.unscaled_low, self.unscaled_high)
        scaled_action = action * self.action_scales + self.action_offsets

        return scaled_action, {INFO_KEY_UNSCALED_ACTION: action}


class DDPGEvaluationPolicy(Policy):
    """A class to define the evaluation policy for DDPG.
    It takes care of moving data to the correct device and scaling the actions
    to the action space of the environment.
    It is used for evaluation and testing of the agent.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]],
    ) -> None:
        """Initialize the DDPG evaluation policy.
        This policy is used for evaluation and testing of the agent.

        Args:
            actor (torch.nn.Module): The actor network to use for action selection.
            action_space (Box): The action space of the actor.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]]):
                A function to preprocess the observation before passing it to the actor.
        """
        self.actor = actor
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.module_device = next(self.actor.parameters()).device

        self.action_scales = (action_space.high - action_space.low) / 2
        self.action_offsets = (action_space.high + action_space.low) / 2

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Returns the action for the given observation.
        This method is called during evaluation and does not add noise to the action.
        It also scales the action to the action space of the environment.

        Args:
            observation (np.ndarray): The observation to use for action selection.

        Returns:
            Tuple[int, Dict[str, Any]]: The action chosen by the policy and the policy info containing
                the unscaled action.
        """

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            action = self.actor(observation)
        action = tensor_to_numpy(action).squeeze()
        scaled_action = action * self.action_scales + self.action_offsets
        return scaled_action, {INFO_KEY_UNSCALED_ACTION: action}
