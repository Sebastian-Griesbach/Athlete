from typing import Callable, Any, Optional, Dict, Tuple
import numpy as np
import torch
from gymnasium.spaces import Box

from athlete.function import numpy_to_tensor, tensor_to_numpy
from athlete.global_objects import StepTracker, RNGHandler
from athlete.policy.policy_builder import Policy, PolicyBuilder
from athlete.algorithms.sac.module import SACActor

INFO_KEY_UNSCALED_ACTION = "unscaled_action"


class SACTrainingPolicy(Policy):
    """Training policy implementation for the Soft Actor-Critic algorithm.

    This policy samples actions from a stochastic policy, performs purely random actions
    during the warmup period, and scales the actions to the action space of the environment.
    """

    def __init__(
        self,
        actor: SACActor,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initializes the SAC training policy.

        Args:
            actor (SACActor): The SAC actor to use for action selection.
            action_space (Box): The action space of the environment.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]]):
                A function to preprocess the observation before passing it to the actor. Defaults to None.
        """
        self.actor = actor
        self.action_space = action_space
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.action_scales = (action_space.high - action_space.low) / 2
        self.action_offsets = (action_space.high + action_space.low) / 2

        self.module_device = next(self.actor.parameters()).device
        self.step_tracker = StepTracker.get_instance()
        self.random_number_generator = RNGHandler.get_random_number_generator()

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Samples an action from the actor given the observation.
        If the warmup period is not done, a random action is sampled.
        The action is scaled to the action space of the environment.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: A tuple containing the sampled action as a numpy array and a dictionary containing the unscaled action.
        """

        if not self.step_tracker.warmup_is_done:
            random_action = self.random_number_generator.random(
                size=self.action_space.shape
            )
            random_scaled_action = (
                random_action * 2 - 1
            ) * self.action_scales + self.action_offsets  # scales the random action to the action space
            return random_scaled_action, {INFO_KEY_UNSCALED_ACTION: random_action}

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            action = self.actor(observation)
        action = tensor_to_numpy(action).squeeze(axis=0)
        scaled_action = action * self.action_scales + self.action_offsets

        return scaled_action, {INFO_KEY_UNSCALED_ACTION: action}


class SACEvaluationPolicy(Policy):
    """The SAC evaluation policy.
    This policy uses the actor to get the deterministic action mean.
    The actions are scaled to the action space of the environment.
    This policy does not consider the warmup period.
    """

    def __init__(
        self,
        actor: SACActor,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initializes the SAC evaluation policy.

        Args:
            actor (SACActor): The SAC actor to use for action selection.
            action_space (Box): The action space of the environment.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional):
                A function to preprocess the observation before passing it to the actor. Defaults to None.
        """
        self.actor = actor
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.module_device = next(self.actor.parameters()).device

        self.action_scales = (action_space.high - action_space.low) / 2
        self.action_offsets = (action_space.high + action_space.low) / 2

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Returns the deterministic action mean given the observation.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: A tuple containing the deterministic action mean as a numpy array and a dictionary containing the unscaled action.
        """

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            action = self.actor.get_mean(observation)
        action = tensor_to_numpy(action).squeeze()
        scaled_action = action * self.action_scales + self.action_offsets
        return scaled_action, {INFO_KEY_UNSCALED_ACTION: action}


class SACPolicyBuilder(PolicyBuilder):
    """The SAC policy builder."""

    def __init__(
        self,
        actor: SACActor,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initializes the SAC policy builder.

        Args:
            actor (SACActor): The SAC actor to use for action selection.
            action_space (Box): The action space of the environment.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional):
                A function to preprocess the observation before passing it to the actor. Defaults to None.
        """
        super().__init__()
        self.actor = actor
        self.action_space = action_space
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

    def build_training_policy(self) -> Policy:
        """Creates the training policy for SAC.

        Returns:
            Policy: The training policy for SAC.
        """
        return SACTrainingPolicy(
            actor=self.actor,
            action_space=self.action_space,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    def build_evaluation_policy(self) -> Policy:
        """Creates the evaluation policy for SAC.

        Returns:
            Policy: The evaluation policy for SAC.
        """
        return SACEvaluationPolicy(
            actor=self.actor,
            action_space=self.action_space,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    @property
    def requires_rebuild_on_policy_change(self) -> bool:
        """Whether the policy builder requires a rebuild on policy change.

        Returns:
            bool: False, as the update only changes the weights of the actor which are
            referenced by the policy classes.
        """
        return False
