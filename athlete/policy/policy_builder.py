from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from athlete.saving.saveable_component import (
    SaveContext,
)

# TODO remove the PolicyBuilder class and instead let the components factory build training and evaluation policies
# This also includes removing the "changes policy" indicator from the UpdatableComponents


class Policy(ABC):
    """Abstract interface for reinforcement learning policies.

    A policy defines how an agent selects actions based on observations from the environment.
    This abstract class provides the core methods that all policy implementations must support,
    including action selection for both normal steps and environment resets, as well as
    checkpoint management functionality.
    """

    def __init__(self) -> None:
        """Initialize the policy."""
        pass

    def reset_act(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        """Generate an action after environment reset.

        By default, this simply calls the act method. Override this if the policy needs
        special handling for the first action in an episode for example wiping some memory.

        Args:
            observation (Any): The initial observation from the reset environment.

        Returns:
            Tuple[Any, Dict[str, Any]]: The selected action and a dictionary of policy info.
        """
        return self.act(observation=observation)

    @abstractmethod
    def act(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        """Generate an action based on the given observation.

        This is the core method of any policy that determines how to select actions.

        Args:
            observation (Any): The current observation from the environment.

        Returns:
            Tuple[Any, Dict[str, Any]]: The selected action and a dictionary of policy info.
                The policy info can contain additional information like log probabilities
                that might be needed for updates.
        """
        ...

    def save_checkpoint(self, context: SaveContext) -> None:
        """Save the policy state if needed.

        Most policies don't need to save state as they can be reconstructed from
        their component networks, but this method is available if needed.

        Args:
            context (SaveContext): Context for saving the checkpoint.
        """
        pass

    # This should not be registered as a saveable component in another class as the policy can be fully
    # reconstructed from the stateful components and the arguments, this save function can be used
    def load_checkpoint(self, context: SaveContext) -> None:
        """Load the policy state if needed.

        Most policies don't need to load state as they can be reconstructed from
        their component networks, but this method is available if needed.

        Args:
            context (SaveContext): Context for loading the checkpoint.
        """
        pass


class PolicyBuilder(ABC):
    """Abstract factory for creating policies in reinforcement learning algorithms.

    The PolicyBuilder serves as a factory for creating both training and evaluation policies.
    It encapsulates all the dependencies and configuration needed to construct appropriate
    policies for different stages of the learning process. Training policies typically
    incorporate exploration mechanisms, while evaluation policies focus on exploitation
    of learned knowledge.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_training_policy(self) -> Policy:
        """Build a policy optimized for training and exploration.

        The training policy typically includes exploration mechanisms such as epsilon-greedy,
        noise injection, or entropy-based exploration to balance exploration and exploitation
        during the learning process.

        Returns:
            Policy: The configured training policy to be used by the agent during learning.
        """
        ...

    @abstractmethod
    def build_evaluation_policy(self) -> Policy:
        """Build a policy optimized for evaluation and exploitation.

        The evaluation policy is typically deterministic or less stochastic than the training
        policy, focusing on exploiting the current learned knowledge to maximize performance
        rather than exploring the environment.

        Returns:
            Policy: The configured evaluation policy to be used by the agent during evaluation.
        """
        ...

    @property
    @abstractmethod
    def requires_rebuild_on_policy_change(self) -> bool:
        """Determine whether policies need rebuilding when underlying parameters change.

        Some policy implementations maintain direct references to networks and automatically
        reflect parameter updates, while others may require explicit rebuilding to incorporate
        changes in the underlying models or configurations.

        Returns:
            bool: True if policies must be rebuilt when underlying parameters change,
                 False if policies automatically reflect parameter updates.
        """
        ...
