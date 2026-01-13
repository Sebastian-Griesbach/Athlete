from abc import ABC, abstractmethod


class DataCollector(ABC):
    """An Interface for data collection.
    This is on of the three main components of an algorithm.
    It takes in the data given to the agent and does with it what ever is needed such that the
    update rule can use the data.
    Note that there is no passing of values between the data collector and the update rule.
    All communication is done via injected dependencies.
    """

    @abstractmethod
    def collect(self, *args, **kwargs) -> bool:
        """Collecting data from the environment during a regular step.

        Returns:
            bool: Whether a new datapoint has been created during this collection for not.
        """
        ...

    @abstractmethod
    def collect_reset(self, *args, **kwargs) -> bool:
        """Collecting data from the environment on a reset call of the environment.
        This does not mean that the collector is reset.

        Returns:
            bool: Whether a new datapoint has been created during this collection for not.
        """
        ...
