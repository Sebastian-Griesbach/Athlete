from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Tuple, Dict, Any, List

from gymnasium.spaces import Space

from athlete.data_collection.collector import DataCollector
from athlete.update.update_rule import UpdateRule
from athlete.policy.policy_builder import PolicyBuilder


@runtime_checkable
class ComponentFactory(Protocol):
    """An interface for a factory for creating the three main components of an algorithm.
    DataCollector, UpdateRule and PolicyBuilder."""

    def __call__(
        self,
        observation_space: Space,
        action_space: Space,
        configuration: Dict[str, Any],
    ) -> Tuple[DataCollector, UpdateRule, PolicyBuilder]:
        """Create the three main components of an algorithm.
           DataCollector, UpdateRule and PolicyBuilder.

        Args:
            observation_space (Space): Observation space of the environment
            action_space (Space): Action space of the environment
            configuration (Dict[str, Any]): Configuration for the algorithm. This needs to contain
                all necessary information to create the components.

        Returns:
            Tuple[DataCollector, UpdateRule, PolicyBuilder]:
                The three main components of an algorithm.
                DataCollector, UpdateRule and PolicyBuilder.
        """
        ...


@dataclass
class Algorithm:
    """A container for algorithm metadata and factory functions.

    This dataclass stores all the information needed to create an instance of a specific
    reinforcement learning algorithm, including its identifier, factory function for
    creating components, and default configuration parameters.
    """

    id: str
    component_factory: ComponentFactory
    default_configuration: Dict[str, Any]


class AlgorithmRegistry:
    """Registry for managing available reinforcement learning algorithms.

    This registry maintains a collection of supported algorithms, allowing for dynamic
    registration and instantiation of different reinforcement learning implementations.
    It provides methods for registering new algorithms, retrieving available algorithms,
    and accessing their default configurations.
    """

    _algorithms = {}

    @classmethod
    def register(
        cls,
        id: str,
        component_factory: ComponentFactory,
        default_configuration: Dict[str, Any],
    ):
        """Register an algorithm in the singleton registry.

        Args:
            id (str): Identifier of the algorithm.
            component_factory (ComponentFactory): Factory function to create the three components
                of the algorithm.
            default_configuration (Dict[str, Any]): Default configuration for the algorithm.
        """
        algorithm = Algorithm(
            id=id,
            component_factory=component_factory,
            default_configuration=default_configuration,
        )
        cls._algorithms[id] = algorithm

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithms in the registry.

        Returns:
            List[str]: List of all registered algorithms IDs.
        """
        return list(cls._algorithms.keys())

    @classmethod
    def get_algorithm(cls, algorithm_id: str) -> Algorithm:
        """Get an algorithm from the registry by its ID.

        Args:
            algorithm_id (str): Identifier of the algorithm to get.

        Raises:
            ValueError: If the algorithm ID is not found in the registry.

        Returns:
            Algorithm: The algorithm object containing the ID, component factory and default configuration.
        """
        if algorithm_id not in cls._algorithms:
            raise ValueError(f"Algorithm {algorithm_id} not found in registry")
        return cls._algorithms[algorithm_id]

    @classmethod
    def get_default_configuration(cls, algorithm_id: str) -> Dict[str, Any]:
        """Get the default configuration for an algorithm by its ID.

        Args:
            algorithm_id (str): Identifier of the algorithm to get the default configuration for.

        Returns:
            Dict[str, Any]: The default configuration for the algorithm.
        """
        algorithm = cls.get_algorithm(algorithm_id)
        return algorithm.default_configuration
