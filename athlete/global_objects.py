import os
from typing import Optional, Any

import numpy as np
import torch

from athlete.saving.saveable_component import SaveContext
from athlete import constants

# TODO test algorithm with updated StepTracker implementation


class StepTracker:
    """The StepTracker tracks information that is relevant for the update conditions of the updatable components."""

    FILE_SAVE_NAME = "step_tracker"
    _instance = None

    @classmethod
    def get_instance(cls) -> "StepTracker":
        """Returns the global instance of the StepTracker.

        Returns:
            StepTracker: The singleton instance
        """
        if cls._instance is None:
            raise Exception(
                "StepTracker has not been initialized. Please call set_global_instance() first."
            )
        return cls._instance

    @classmethod
    def set_global_instance(cls, instance: "StepTracker") -> None:
        """Sets the global instance of the StepTracker.

        Args:
            step_tracker (StepTracker): The instance to set as the global instance.
        """
        cls._instance = instance

    def __init__(self, warmup_steps: int = 0) -> None:
        """Initializes a StepTracker instance.

        Args:
            warmup_steps (int): Number of warmup steps that an algorithm might perform,
                this affects the interactions_after_warmup property.
        """
        self.registered_trackers: dict[str, int] = {}
        self.meta_data: dict[str, Any] = {}

        self.register_tracker(
            id=constants.TRACKER_ENVIRONMENT_INTERACTIONS, inital_value=0
        )
        self.register_tracker(id=constants.TRACKER_ENVIRONMENT_EPISODES, inital_value=0)
        self.register_tracker(id=constants.TRACKER_DATA_POINTS, inital_value=0)

        self.meta_data[constants.GENERAL_ARGUMENT_WARMUP_STEPS] = warmup_steps

    def register_tracker(self, id: str, inital_value: int = 0) -> str:
        """Registers a tracker with a unique ID. If the ID is already registered a number will be appended to the ID to make it unique.

        Args:
            id (str): Identifier for the tracker.
            inital_value (int, optional): Initial value for the tracker. Defaults to 0.

        Returns:
            str: The ID that was finally used for the tracker, which might be different from the input ID if it was already registered.
        """
        if id not in self.registered_trackers:
            self.registered_trackers[id] = inital_value
            return id

        # If the ID is already registered, append a number to make it unique
        i = 2
        id_candidate = f"{id}_{i}"
        while id_candidate in self.registered_trackers:
            i += 1
            id_candidate = f"{id}_{i}"

        self.registered_trackers[id_candidate] = inital_value
        return id_candidate

    def increment_tracker(self, id: str, increment: int = 1) -> None:
        """Increments the value of a registered tracker.

        Args:
            id (str): Identifier for the tracker.
            increment (int, optional): Value to increment the tracker by. Defaults to 1.

        Raises:
            KeyError: If the tracker with the given ID is not registered.
        """
        if id not in self.registered_trackers:
            raise KeyError(f"Tracker with ID '{id}' is not registered.")
        self.registered_trackers[id] += increment

    def set_tracker_value(self, id: str, value: int) -> None:
        """Sets the value of a registered tracker.

        Args:
            id (str): Identifier for the tracker.
            value (int): Value to set for the tracker.

        Raises:
            KeyError: If the tracker with the given ID is not registered.
        """
        if id not in self.registered_trackers:
            raise KeyError(f"Tracker with ID '{id}' is not registered.")
        self.registered_trackers[id] = value

    def get_tracker_value(self, id: str) -> int:
        """Gets the value of a registered tracker.

        Args:
            id (str): Identifier for the tracker.

        Raises:
            KeyError: If the tracker with the given ID is not registered.

        Returns:
            int: Value of the tracker.
        """
        if id not in self.registered_trackers:
            raise KeyError(f"Tracker with ID '{id}' is not registered.")
        return self.registered_trackers[id]

    @property
    def interactions_after_warmup(self) -> int:
        """Returns the number of interactions after the warmup period.

        Returns:
            int: Number of interactions after the warmup period.
        """
        return max(
            0,
            self.get_tracker_value(id=constants.TRACKER_ENVIRONMENT_INTERACTIONS)
            - self.meta_data[constants.GENERAL_ARGUMENT_WARMUP_STEPS],
        )

    @property
    def is_warmup_done(self) -> bool:
        """Checks if the warmup period is done.

        Returns:
            bool: True if the warmup period is done, False otherwise.
        """
        return (
            self.get_tracker_value(id=constants.TRACKER_ENVIRONMENT_INTERACTIONS)
            >= self.meta_data[constants.GENERAL_ARGUMENT_WARMUP_STEPS]
        )

    def save_checkpoint(self, context: SaveContext) -> None:
        """Saves the state of the step tracker to a checkpoint.

        Args:
            context (SaveContext): The context for saving the checkpoint.
        """
        to_save = (
            self.registered_trackers,
            self.meta_data,
        )
        save_path = os.path.join(
            context.save_path, context.prefix + self.FILE_SAVE_NAME
        )

        context.file_handler.save_to_file(to_save=to_save, save_path=save_path)

    def load_checkpoint(self, context: SaveContext) -> None:
        """Loads the state of the step tracker from a checkpoint.

        Args:
            context (SaveContext): The context for loading the checkpoint.
        """
        load_path = os.path.join(
            context.save_path, context.prefix + self.FILE_SAVE_NAME
        )

        (
            self.registered_trackers,
            self.meta_data,
        ) = context.file_handler.load_from_file(load_path=load_path)


class RNGHandler:
    """This class handles random number generation to ensure reproducibility across training runs.
    It should be used as a singleton. It sets the global seed for numpy and torch as well as provides
    a consistent random number generator instance from numpy that can be accessed throughout the codebase.
    """

    FILE_SAVE_NAME = "rng_handler"

    _instance = None

    @classmethod
    def get_random_number_generator(cls) -> np.random.Generator:
        """Returns the global random number generator instance.

        Raises:
            Exception: If the instance has not been initialized.

        Returns:
            np.random.Generator: The global random number generator instance.
        """
        if cls._instance is None:
            raise Exception("RNGHandler has not been initialized.")
        return cls._instance._random_number_generator

    @classmethod
    def get_seed(cls) -> int:
        """Returns the global seed.

        Raises:
            Exception: If the instance has not been initialized.

        Returns:
            int: The global seed.
        """
        if cls._instance is None:
            raise Exception("RNGHandler has not been initialized.")
        return cls._instance.seed

    @classmethod
    def get_instance(cls) -> "RNGHandler":
        """Returns the global instance of the RNGHandler.

        Raises:
            Exception: If the instance has not been initialized.

        Returns:
            RNGHandler: The global instance of the RNGHandler.
        """
        if cls._instance is None:
            raise Exception("RNGHandler has not been initialized.")
        return cls._instance

    @classmethod
    def set_global_instance(cls, rng_handler: "RNGHandler") -> None:
        """Sets the global instance of the RNGHandler.

        Args:
            rng_handler (RNGHandler): The instance to set as the global instance.
        """
        cls._instance = rng_handler

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initializes a RNGHandler instance. This can be different from the global instance.

        Args:
            seed (Optional[int], optional): Seed for the random number generator. If None, a random seed will be generated. Defaults to None.
        """

        if not seed:
            seed = int(np.random.randint(low=0, high=np.iinfo(np.uint32).max))
        self.seed = seed

        # For convenience this ensures reproducibility for functions that do not use the global rng but numpy or torch
        # as long as they are used in the same order
        # Set seed for all numpy functions
        np.random.seed(seed)
        # Set seed for all torch functions
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Create a random number generator instance
        # Best practice always use this one if possible
        self._random_number_generator = np.random.default_rng(seed)

    def save_checkpoint(self, context: SaveContext) -> None:

        # Global numpy random state
        np_state = np.random.get_state()

        # Global torch random state (CPU)
        torch_state = torch.get_rng_state()

        # Save CUDA states for all devices
        cuda_states = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cuda_states[i] = torch.cuda.get_rng_state(i)

        # Random number generator state
        rng_state = self._random_number_generator.bit_generator.state

        to_save = (self.seed, np_state, torch_state, cuda_states, rng_state)

        context.file_handler.save_to_file(
            to_save=to_save,
            save_path=os.path.join(
                context.save_path, context.prefix + self.FILE_SAVE_NAME
            ),
        )

    def load_checkpoint(self, context: SaveContext) -> None:

        loaded = context.file_handler.load_from_file(
            load_path=os.path.join(
                context.save_path, context.prefix + self.FILE_SAVE_NAME
            )
        )

        self.seed, np_state, torch_state, cuda_states, rng_state = loaded
        # Set global numpy random state
        np.random.set_state(np_state)

        # Set global torch random state (CPU)
        torch.set_rng_state(torch_state)

        # Set CUDA states for all devices
        if cuda_states and torch.cuda.is_available():
            for device_id, state in cuda_states.items():
                if device_id < torch.cuda.device_count():
                    torch.cuda.set_rng_state(state, device_id)

        # Set random number generator state
        self._random_number_generator.bit_generator.state = rng_state
