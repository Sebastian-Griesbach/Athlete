from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from athlete.saving.saveable_component import (
    SaveContext,
)


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
