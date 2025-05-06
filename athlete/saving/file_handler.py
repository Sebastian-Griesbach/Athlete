from abc import ABC, abstractmethod
import torch
from typing import Any


class FileHandler(ABC):
    """Abstract interface for file operations used in saving and loading checkpoints.

    This class defines the common operations needed to persist and retrieve agent states
    from storage. It abstracts away the details of serialization formats and storage
    mechanisms, allowing for different implementations (e.g., pickle, JSON, torch)
    without changing the checkpoint logic.
    """

    def __init__(self) -> None:
        """Initializes the file handler with an empty cache."""
        super().__init__()
        self.cache = {}

    def save_to_file(self, to_save: Any, save_path: str, enable_cache=False) -> None:
        """Saves a file to the specified path. If the file already exists in the cache, it will not be saved again.

        Args:
            to_save (any): Object to save.
            save_path (str): Path to save the file to.
            enable_cache (bool, optional): Whether to enable caching, if True will not save the same path twice. Defaults to False.
        """
        if save_path in self.cache:
            return

        self._file_save_operation(to_save=to_save, save_path=save_path)
        if enable_cache:
            self.cache[save_path] = to_save

    def load_from_file(self, load_path: str, enable_cache=False) -> Any:
        """Loads a file from the specified path. If the file already exists in the cache, it will be loaded from there.

        Args:
            load_path (str): Path to load the file from.
            enable_cache (bool, optional): Whether to enable caching,
            if True will not load the same path twice from file but return the object loaded the first time on second call.
            Defaults to False.

        Returns:
            Any: Loaded object.
        """

        if enable_cache and load_path in self.cache.keys():
            return self.cache[load_path]

        loaded_file = self._load_file_operation(load_path=load_path)
        if enable_cache:
            self.cache[load_path] = loaded_file

        return loaded_file

    def reset_cache(self) -> None:
        """Resets the cache."""

        self.cache = {}

    @abstractmethod
    def _file_save_operation(self, to_save: Any, save_path: str) -> None:
        """Saves a file to the specified path. This method should be implemented in the derived class.

        Args:
            to_save (Any): Object to save.
            save_path (str): Path to save the file to.
        """
        ...

    @abstractmethod
    def _load_file_operation(self, load_path: str) -> Any:
        """Loads a file from the specified path. This method should be implemented in the derived class.

        Args:
            load_path (str): Path to load the file from.

        Returns:
            Any: Loaded object.
        """
        ...


class TorchFileHandler(FileHandler):
    """Simple implementation of a FileHandler that uses torch to save and load files."""

    def __init__(self) -> None:
        super().__init__()

    def _file_save_operation(self, to_save: Any, save_path: str) -> None:
        """Saves file using torch and pickle protocol 4. It adds ".pt" to the file name.

        Args:
            to_save (Any): Object to save.
            save_path (str): Path to save the file.
        """
        torch.save(to_save, save_path + ".pt", pickle_protocol=4)

    def _load_file_operation(self, load_path: str) -> Any:
        """Loads file using torch. It adds ".pt" to the file name."

        Args:
            load_path (str): Path to load the file from.

        Returns:
            Any: Loaded object.
        """
        return torch.load(load_path + ".pt", weights_only=False)
