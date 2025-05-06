from typing import Tuple, Any
from abc import ABC, abstractmethod
import os

import numpy as np

from athlete.global_objects import RNGHandler
from athlete.saving.saveable_component import SaveContext


class NoiseProcess(ABC):
    """Interface for noise processes."""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """Overwrite this if the noise process has any state that needs to be reset upon resetting the environment."""
        pass

    @abstractmethod
    def sample(self) -> Any:
        """Sample noise from the noise process.

        Returns:
            Any: The sampled noise. The type of the noise depends on the implementation of the noise process.
        """
        ...

    def save_checkpoint(self, context: SaveContext) -> None:
        """Overwrite this if the noise process has any state that needs to be saved.

        Args:
            context (SaveContext): The context object that contains the save path and file handler.
        """
        pass

    def load_checkpoint(self, context: SaveContext) -> None:
        """Overwrite this if the noise process has any state that needs to be saved.

        Args:
            context (SaveContext): The context object that contains the save path and file handler.
        """
        pass


class GaussianNoise(NoiseProcess):
    """A Gaussian noise process."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        noise_std: float = 0.2,
    ) -> None:
        """Initialize the Gaussian noise process.

        Args:
            shape (Tuple[int, ...]): The shape of the noise to be sampled.
            noise_std (float, optional): The standard deviation of the Gaussian noise. Defaults to 0.2.
        """
        super().__init__()
        self.shape = shape
        self.noise_std = noise_std
        self.random_numbers_generator = RNGHandler.get_random_number_generator()

    def sample(self) -> np.ndarray:
        """Sample noise from the Gaussian noise process.

        Returns:
            np.ndarray: The sampled Gaussian noise.
        """
        return self.random_numbers_generator.normal(
            loc=0.0, scale=self.noise_std, size=self.shape
        )


class OhrsteinUhlenbeckNoise(NoiseProcess):
    """A class that implements the Ornstein-Uhlenbeck process for generating noise."""

    SAVE_FILE_NAME = "ohrnstein_uhlenbeck_noise"

    def __init__(
        self,
        shape: Tuple[int, ...],
        noise_std: float = 0.2,
        theta: float = 0.15,
        dt: float = 0.01,
        mu: float = 0.0,
    ) -> None:
        """Initialize the Ornstein-Uhlenbeck noise process.

        Args:
            shape (Tuple[int, ...]): The shape of the noise to be sampled.
            noise_std (float, optional): The standard deviation of the Gaussian noise. Defaults to 0.2.
            theta (float, optional): The rate of mean reversion. Defaults to 0.15.
            dt (float, optional): The time increment per sample, the lower this is the smoother the process becomes. Defaults to 0.01.
            mu (float, optional): The long-term mean of the process. Defaults to 0.0.
        """
        super().__init__()
        self.noise_std = noise_std
        self.theta = theta
        self.dt = dt
        self.mu = mu
        self.shape = shape
        self.last_noise = np.zeros(shape)
        self.random_numbers_generator = RNGHandler.get_random_number_generator()

    def reset(self) -> None:
        """Reset the noise process to its initial state which means it samples near the mean of the process."""
        self.last_noise = np.zeros(self.shape)

    def sample(self) -> np.ndarray:
        """Sample noise from the Ornstein-Uhlenbeck process.

        Returns:
            np.ndarray: The sampled noise from the Ornstein-Uhlenbeck process.
        """
        noise = (
            self.last_noise
            + self.theta * (self.mu - self.last_noise) * self.dt
            + self.noise_std
            * np.sqrt(self.dt)
            * self.random_numbers_generator.normal(size=self.shape)
        )
        self.last_noise = noise
        return noise

    def save_checkpoint(self, context: SaveContext) -> None:
        """Saving the last noise state of the Ornstein-Uhlenbeck process."""
        full_save_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        handling_stats = (self.last_noise,)

        context.file_handler.save_to_file(
            to_save=handling_stats, save_path=full_save_path
        )

    def load_checkpoint(self, context: SaveContext) -> None:
        """Load the last noise state of the Ornstein-Uhlenbeck process."""
        full_load_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        handling_stats = context.file_handler.load_from_file(load_path=full_load_path)

        self.last_noise = handling_stats[0]


class ColoredNoise(NoiseProcess):
    """A class that implements a colored noise process. A noise type that filters high frequencies out such that the noise is smoother than white noise."""

    SAVE_FILE_NAME = "colored_noise"

    def __init__(
        self,
        shape: Tuple[int, ...],
        horizon: int,
        beta: float = 1.0,
        noise_std: float = 1.0,
    ) -> None:
        """Initialize the colored noise process.

        Args:
            shape (Tuple[int, ...]): The shape of the noise to be sampled.
            horizon (int): The length of the generated noise sequence. If the sequence is depleted, a new one is generated.
            beta (float, optional): The exponent for the frequency scaling. A value of 0 corresponds to white noise, while a value of 2 corresponds to Brownian noise. Defaults to 1.0.
            noise_std (float, optional): The standard deviation of the Gaussian noise. Defaults to 1.0.
        """
        super().__init__()
        self.shape = shape
        self.horizon = horizon
        self.beta = beta
        self.sequence_buffer = np.zeros((horizon, *shape))
        self.pointer = 0
        self.noise_std = noise_std
        self.random_numbers_generator = RNGHandler.get_random_number_generator()

    def reset(self) -> None:
        """This will cause a new sequence to be generated on the next sample call"""
        self.pointer = 0

    def sample(self) -> np.ndarray:
        """Sample noise from the colored noise process.
        The noise is sampled from a pre-generated sequence. If the sequence is depleted, a new one is generated.

        Returns:
            np.ndarray: The sampled noise from the colored noise process.
        """
        if self.pointer == 0:
            self.sequence_buffer = self._generate_colored_noise_sequence(
                beta=self.beta,
                sequence_length=self.horizon,
                dimensions=np.prod(self.shape),
            ).T  # transposing so that first dimension is the time dimension second are the dimensions of the action space

        noise = self.noise_std * self.sequence_buffer[self.pointer]
        self.pointer = (self.pointer + 1) % self.horizon

        return noise

    def _generate_colored_noise_sequence(
        self, beta, sequence_length, dimensions=1
    ) -> np.ndarray:
        """Generate a sequence of colored noise using the Fourier method.
        The noise is generated in the frequency domain and then transformed to the time domain using the inverse FFT.
        The generated noise is normalized to have a standard deviation of 1.

        Args:
            beta (_type_): The exponent for the frequency scaling. A value of 0 corresponds to white noise, while a value of 2 corresponds to Brownian noise.
            sequence_length (_type_): The length of the generated noise sequence.
            dimensions (int, optional): The number of dimensions of the generated noise. Defaults to 1.

        Returns:
            np.ndarray: The generated colored noise sequence.
        """
        # Create frequency bins
        frequency_amplitudes = np.fft.rfftfreq(sequence_length)

        num_positive_frequencies = frequency_amplitudes.shape[0]

        # Create the scaling factors for each frequency
        frequencies_scaling = np.zeros(num_positive_frequencies)
        frequencies_scaling[1:] = np.power(
            np.abs(frequency_amplitudes[1:]), -beta / 2
        )  # Avoid division by zero

        # Calculate the theoretical standard deviation for scaling
        sigma = np.sqrt(np.sum(frequencies_scaling**2) / num_positive_frequencies)

        # Generate random phases. Introduce a random phase shift for each dimension.
        phases = self.random_numbers_generator.uniform(
            low=0, high=2 * np.pi, size=(dimensions, num_positive_frequencies)
        )

        # Create the complex frequency components. We generate them across all dimensions
        signal_frequencies = frequencies_scaling * np.exp(1j * phases)

        # Inverse FFT to get the time-domain signal, for each dimension
        signal = np.fft.irfft(signal_frequencies, n=sequence_length)

        # Normalize the signal
        signal /= sigma

        return signal

    def save_checkpoint(self, context: SaveContext) -> None:
        """Saves the current sequence buffer and pointer of the colored noise process.

        Args:
            context (SaveContext): The context object that contains the save path and file handler.
        """
        full_save_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        handling_stats = (
            self.sequence_buffer,
            self.pointer,
        )

        context.file_handler.save_to_file(
            to_save=handling_stats, save_path=full_save_path
        )

    def load_checkpoint(self, context: SaveContext) -> None:
        """Loads the current sequence buffer and pointer of the colored noise process.

        Args:
            context (SaveContext): The context object that contains the save path and file handler.
        """
        full_load_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        handling_stats = context.file_handler.load_from_file(load_path=full_load_path)

        (
            self.sequence_buffer,
            self.pointer,
        ) = handling_stats
