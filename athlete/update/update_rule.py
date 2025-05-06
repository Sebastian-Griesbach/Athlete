from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class UpdatableComponent(ABC):
    """Abstract base class for components that can be updated during reinforcement learning training.

    Each component defines its own update logic and conditions for when updates should occur (e.g., based on step count
    or episode boundaries). The component also tracks whether its updates directly affect the
    agent's policy behavior.
    """

    def __init__(
        self,
        changes_policy: bool,
    ) -> None:
        """Initialise the updatable component.

        Args:
            changes_policy (bool): Whether updating this component immediately changes the policy.
        """

        super().__init__()
        self.changes_policy = changes_policy

    @abstractmethod
    def update(self) -> Dict[str, Any]:
        """Perform the update defined by this component. Implement this method in the derived class.

        Returns:
            Dict[str, Any]: Logging data from the update, return {} if there is nothing to log.
        """
        ...

    @property
    @abstractmethod
    def update_condition(self) -> bool:
        """This indicates whether an update should be performed for this component in the current moment.
        All necessary dependencies to determine this should be provided during initialization. E.g. using the step tracker
        to make the update condition depend on an update frequency.

        Returns:
            bool: Whether the component should be updated.
        """
        ...


class UpdateRule(ABC):
    """Base interface for reinforcement learning algorithm update rules.

    An UpdateRule defines the learning process for RL algorithms by coordinating multiple
    updatable components. It manages when updates occur, ensures proper sequencing of
    component updates, and tracks whether policy changes have occurred. Each concrete
    implementation corresponds to a specific algorithm's learning approach.
    """

    def __init__(self) -> None:
        """In the constructor of the derived class all necessary dependencies should be provided or created to initialize the
        updated components.
        """
        ABC.__init__(self)

    def update(self) -> Tuple[bool, Dict[str, Any]]:
        """Perform an update for each of the updatable components if their update condition is met.

        Returns:
            Tuple[bool, Dict[str, Any]]: A boolean indicating whether the policy has changed and accumulated logging information from all components.
        """
        accumulated_logs = {}
        policy_changed = False

        for component in self.updatable_components:
            # Only update components that meet their update condition
            if component.update_condition:
                component_logs = component.update()
                policy_changed |= component.changes_policy
                if component_logs:
                    accumulated_logs.update(component_logs)

        return policy_changed, accumulated_logs

    @property
    @abstractmethod
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns a tuple of all updatable components in the order they should be updated in. (They are only updated if their update condition is met)

        Returns:
            Tuple[UpdatableComponent]: Tuple of all updatable components in the order they should be updated in.
        """
        ...
