from typing import Any, Dict, Optional, Tuple


# TODO remove data provider and add this functionality to the data collector,
# pass the get data function to the updatable components that needs to access the data
class UpdateDataProvider:
    """A communication channel between data collectors and update rules.

    This container holds data which can be updated by data collectors and retrieved by update rules.
    It serves as a decoupling mechanism for transferring experience data between the collection
    phase and the learning phase without requiring direct dependencies between these components.
    """

    def __init__(
        self,
        initial_data: Optional[Any] = None,
        initial_metadata: Dict[str, Any] = {},
    ) -> None:
        """Initialize the UpdateDataProvider.

        Args:
            initial_data (Optional[Any], optional): Initial data held in the provider.
                Defaults to None.
            initial_metadata (Dict[str, Any], optional): _Initial metadata held in the provider.
                Defaults to {}.
        """
        self.update_data = initial_data
        self.metadata = initial_metadata

    def set_data(self, update_data: Any, metadata: Dict[str, Any] = {}) -> None:
        """Set new data in the provider.

        Args:
            update_data (Any): New data to be set in the provider.
            metadata (Dict[str, Any], optional): New metadata to be set in the provider.
        """
        self.update_data = update_data
        self.metadata = metadata

    def get_data(self) -> Tuple[Any, Dict[str, Any]]:
        """Get the current data and metadata from the provider.

        Returns:
            Tuple[Any, Dict[str, Any]]: The current data and metadata held in the provider.
        """
        return self.update_data, self.metadata
