from typing import Protocol, Dict, Any, runtime_checkable, Type, Callable
import os
from dataclasses import dataclass

import torch
import numpy as np

from athlete.saving.file_handler import FileHandler


@dataclass
class SaveContext:
    """Dataclass that provides context information for saving and loading operations.

    This class encapsulates all necessary information required by components during
    checkpoint operations, including file handling, paths, naming prefixes, and metadata.
    Prefixes can be used if the same object type is used in different places to avoid name clashes.
    """

    file_handler: FileHandler
    save_path: str
    prefix: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def with_prefix(self, additional_prefix: str) -> "SaveContext":
        """Create a new context with an additional prefix.

        Args:
            additional_prefix (str): Additional prefix to append to the existing prefix.

        Returns:
            SaveContext: New SaveContext with the updated prefix.
        """
        return SaveContext(
            file_handler=self.file_handler,
            save_path=self.save_path,
            prefix=f"{self.prefix}{additional_prefix}",
            metadata=self.metadata.copy(),
        )

    def replace_metadata(self, **kwargs) -> "SaveContext":
        """Create a new context with replaced metadata.

        Returns:
            SaveContext: New SaveContext with the replaced metadata.
        """
        return SaveContext(
            file_handler=self.file_handler,
            save_path=self.save_path,
            prefix=self.prefix,
            metadata=kwargs,
        )


@runtime_checkable
class SaveableComponent(Protocol):
    """Protocol for saveable components. Any object that implements this protocol can be saved and loaded."""

    def save_checkpoint(self, context: SaveContext) -> None:
        """Save this object's state using the provided context.

        Args:
            context (SaveContext): The context to use for saving the object's state.
        """
        pass

    def load_checkpoint(self, context: SaveContext) -> None:
        """Load this object's state using the provided context.

        Args:
            context (SaveContext): The context to use for loading the object's state.
        """
        pass


class CompositeSaveableComponent:
    """An object that contains multiple saveable components. It provides functionality to register
    saveable components such that they will be saved and loaded upon calling the respective methods.
    The components can be either an object following the SaveableComponent Protocol or a saver function must be registered
    in the SaverRegistry for this type.
    """

    def __init__(self) -> None:
        """Initialize the CompositeSaveableComponent.
        This initializes two dictionaries to keep track of the components:
        - _saveable_components: Components that implement the SaveableComponent Protocol
        - _external_components: Components that can not save themselves but need a saver function registered in the SaverRegistry
        """
        self._saveable_components = {}  # Components that implement Saveable
        self._external_components = {}  # Components that need SaverRegistry

    def register_saveable_component(
        self,
        name: str,
        component: Any,
    ) -> None:
        """Register a component under the given name.

        The component is automatically categorized as either saveable or external.
        For externals, this checks if a matching saver exists in the registry.

        Args:
            name: Name to identify this component, also used as the filename for external components
            component: The component to save

        Raises:
            TypeError: If component is not saveable and no saver is registered for its type.
        """

        # Check if component implements the Saveable protocol by looking for required methods
        is_saveable = isinstance(component, SaveableComponent)

        if is_saveable:
            self._saveable_components[name] = component
        else:
            # Check if there's a registered saver for this component type
            has_saver = False
            for data_type in SaverRegistry._savers.keys():
                # This also considers subclassing
                if isinstance(component, data_type):
                    has_saver = True
                    break

            if not has_saver:
                raise TypeError(
                    f"Cannot register component '{name}' of type {type(component).__name__}: "
                    f"it does not implement the saveable component protocol and no saver is registered for this type."
                )

            self._external_components[name] = component

    def save_checkpoint(self, context: SaveContext) -> None:
        """Save all registered components.
        And save any local state according to the _save_local_state function.

        Args:
            context (SaveContext): The context to use for saving the object's state.
        """
        # First save any local state
        self._save_local_state(context)

        # Save all Saveable components
        for name, component in self._saveable_components.items():
            component.save_checkpoint(context=context)

        # Save all external components using SaverRegistry
        for name, component in self._external_components.items():
            SaverRegistry.save(obj=component, context=context, filename=name)

    def load_checkpoint(self, context: SaveContext) -> None:
        """Load all registered components.
        And load any local state according to the _load_local_state function.

        Args:
            context (SaveContext): The context to use for loading the object's state.

        Raises:
            RuntimeError: If loading a external component fails.
        """
        # First load any local state
        self._load_local_state(context)

        # Load all Saveable components
        for name, component in self._saveable_components.items():
            component.load_checkpoint(context)

        # Load all external components using SaverRegistry
        for name, component in self._external_components.items():
            try:
                SaverRegistry.load(component, context, name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load external component '{name}': {e}"
                ) from e

    def _save_local_state(self, context: SaveContext) -> None:
        """Override to save component-specific state.

        Args:
            context (SaveContext): The context to use for saving the object's state.
        """
        pass

    def _load_local_state(self, context: SaveContext) -> None:
        """Override to load component-specific state.

        Args:
            context (SaveContext): The context to use for loading the object's state.
        """
        pass


class SaverRegistry:
    """A registry of specialized saving and loading function for different data types.
    This is useful of the object you want to save can not implement the SaveableComponent protocol.
    """

    _savers = {}
    _loaders = {}

    @classmethod
    def register(
        cls,
        data_type: Type,
        saver_fn: Callable[[Any, SaveContext, str], None],
        loader_fn: Callable[[Any, SaveContext, str], None],
    ):
        """Register a saver and loader function for a specific data type.

        Args:
            data_type (Type): Data type to register the saver and loader for
            saver_fn (Callable[[Any, SaveContext, str], None]): Saver function to register
            loader_fn (Callable[[Any, SaveContext, str], None]): Loader function to register,
            note that this returns none, it has to overwrite the object in place.
        """
        cls._savers[data_type] = saver_fn
        cls._loaders[data_type] = loader_fn

    @classmethod
    def save(cls, obj: Any, context: SaveContext, filename: str) -> None:
        """Save an object using the appropriate registered saver.

        Args:
            obj (_type_): Object to save
            context (SaveContext): The save context
            filename (str): Name of the file to save to

        Raises:
            TypeError: If no registered saver is found for the object's type
        """
        for data_type, saver_fn in cls._savers.items():
            if isinstance(obj, data_type):
                saver_fn(obj, context, filename)
                return
        raise TypeError(f"No registered saver for object of type: {type(obj)}")

    @classmethod
    def load(cls, obj: Any, context: SaveContext, filename: str) -> None:
        """Load an object using the appropriate registered loader.
        This function modifies the object in place and returns None.

        Args:
            obj (Any): Object to load
            context (SaveContext): The save context
            filename (str): Name of the file to load from

        Raises:
            TypeError: If no registered loader is found for the object's type
        """
        for data_type, loader_fn in cls._loaders.items():
            if isinstance(obj, data_type):
                loader_fn(obj, context, filename)
                return

        raise TypeError(f"No registered loader for type: {obj}")


# Register specialized savers, note that the load functions need to modify the to_load object in place


def save_torch_module(
    to_save: torch.nn.Module, context: SaveContext, filename: str
) -> None:
    """Save a PyTorch module."""
    save_path = os.path.join(context.save_path, context.prefix + filename)
    context.file_handler.save_to_file(to_save.state_dict(), save_path)


def load_torch_module(
    to_load: torch.nn.Module, context: SaveContext, filename: str
) -> None:
    """Load a PyTorch module."""
    load_path = os.path.join(context.save_path, context.prefix + filename)
    state_dict = context.file_handler.load_from_file(load_path)
    to_load.load_state_dict(state_dict)
    return to_load


def save_torch_optimizer(
    to_save: torch.optim.Optimizer, context: SaveContext, filename: str
) -> None:
    """Save a PyTorch optimizer."""
    save_path = os.path.join(context.save_path, context.prefix + filename)
    context.file_handler.save_to_file(to_save.state_dict(), save_path)


def load_torch_optimizer(
    to_load: torch.optim.Optimizer, context: SaveContext, filename: str
) -> None:
    """Load a PyTorch optimizer."""
    load_path = os.path.join(context.save_path, context.prefix + filename)
    state_dict = context.file_handler.load_from_file(load_path)
    to_load.load_state_dict(state_dict)


def save_default(to_save: Any, context: SaveContext, filename: str) -> None:
    """Saves anything that doesn't require special handling."""
    save_path = os.path.join(context.save_path, context.prefix + filename)
    context.file_handler.save_to_file(to_save, save_path)


def load_numpy_array(to_load: np.ndarray, context: SaveContext, filename: str) -> None:
    """Load a NumPy array."""
    load_path = os.path.join(context.save_path, context.prefix + filename)
    loaded = context.file_handler.load_from_file(load_path)
    np.copyto(to_load, loaded)


def save_torch_parameter(
    to_save: torch.nn.Parameter, context: SaveContext, filename: str
) -> None:
    """Save a PyTorch parameter."""
    save_path = os.path.join(context.save_path, context.prefix + filename)
    context.file_handler.save_to_file(to_save.data, save_path)


def load_torch_parameter(
    to_load: torch.nn.Parameter, context: SaveContext, filename: str
) -> None:
    """Load a PyTorch parameter."""
    load_path = os.path.join(context.save_path, context.prefix + filename)
    loaded = context.file_handler.load_from_file(load_path)
    to_load.data.copy_(loaded)


# Register the savers
SaverRegistry.register(torch.nn.Module, save_torch_module, load_torch_module)
SaverRegistry.register(
    torch.optim.Optimizer, save_torch_optimizer, load_torch_optimizer
)
SaverRegistry.register(np.ndarray, save_default, load_numpy_array)
SaverRegistry.register(torch.nn.Parameter, save_torch_parameter, load_torch_parameter)
