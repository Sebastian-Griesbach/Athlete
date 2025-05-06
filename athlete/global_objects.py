import os
from typing import Optional

import numpy as np
import torch

from athlete.saving.saveable_component import SaveContext


class StepTracker:
    """The StepTracker holds the number of interactions, episodes and datapoints added to the replay buffer.
    This information might be useful for updating conditions. This class can be used as a singleton.
    """

    FILE_SAVE_NAME = "step_tracker"

    _instance = None

    @classmethod
    def get_instance(cls) -> "StepTracker":
        """Returns the global instance of the StepTracker.

        Raises:
            Exception: If the instance has not been initialized.

        Returns:
            StepTracker: The global instance of the StepTracker.
        """
        if cls._instance is None:
            raise Exception("TrainingTracker has not been initialized.")
        return cls._instance

    @classmethod
    def set_global_instance(cls, instance: "StepTracker") -> None:
        """Sets the global instance of the StepTracker.

        Args:
            instance (StepTracker): The instance to set as the global instance.
        """
        cls._instance = instance

    def __init__(self, warmup_steps: int) -> None:
        """Initializes a StepTracker instance. Can be different from the global instance.

        Args:
            warmup_steps (int): Number of warmup steps that an algorithm might perform, this affects the interactions_after_warmup property.
        """
        self.total_interactions = 0
        self.total_number_of_datapoints_added = 0
        self.total_number_of_episodes = 0
        self.warmup_steps = warmup_steps

    def increment_environment_interaction(self, num_interactions: int = 1) -> None:
        """Increments the number of interactions with the environment.

        Args:
            num_interactions (int, optional): Number of interactions to add. Defaults to 1.
        """
        self.total_interactions += num_interactions

    def increment_datapoint(self, num_datapoints: int = 1) -> None:
        """Increments the number of collected datapoints, what exactly this means depends on the algorithm.

        Args:
            num_datapoints (int, optional): Number of datapoints to add. Defaults to 1.
        """
        self.total_number_of_datapoints_added += num_datapoints

    def increment_episode(self, num_episodes: int = 1) -> None:
        """Increments the number of episodes.

        Args:
            num_episodes (int, optional): Number of episodes to add. Defaults to 1.
        """
        self.total_number_of_episodes += num_episodes

    @property
    def interactions_after_warmup(self) -> int:
        """Returns the number of interactions after the warmup period.

        Returns:
            int: Number of interactions after the warmup period.
        """
        return max(0, self.total_interactions - self.warmup_steps)

    @property
    def warmup_is_done(self) -> bool:
        """Checks if the warmup period is done.

        Returns:
            bool: True if the warmup period is done, False otherwise.
        """
        return self.total_interactions >= self.warmup_steps

    def save_checkpoint(self, context: SaveContext) -> None:

        to_save = (
            self.total_interactions,
            self.total_number_of_episodes,
            self.total_number_of_datapoints_added,
        )
        save_path = os.path.join(
            context.save_path, context.prefix + self.FILE_SAVE_NAME
        )

        context.file_handler.save_to_file(to_save=to_save, save_path=save_path)

    def load_checkpoint(self, context: SaveContext) -> None:

        load_path = os.path.join(
            context.save_path, context.prefix + self.FILE_SAVE_NAME
        )

        (
            self.total_interactions,
            self.total_number_of_episodes,
            self.total_number_of_datapoints_added,
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
