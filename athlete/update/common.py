from typing import Dict, Any, Optional
from abc import abstractmethod
import os

import torch
from torch.optim import Optimizer
import numpy as np

from athlete.update.update_rule import UpdatableComponent
from athlete.data_collection.provider import UpdateDataProvider
from athlete.update.buffer import Buffer
from athlete.global_objects import StepTracker
from athlete.saving.saveable_component import SaveContext


class TorchFrequentGradientUpdate(UpdatableComponent):
    """Base class for components that require gradient-based updates with configurable frequency.

    This class provides a foundation for updatable components in reinforcement learning algorithms
    that use torch optimizers and require updates at specific frequencies. It handles update timing,
    gradient manipulation, update counting, and logging. Derived classes need only implement the
    calculate_loss() method to specify the specific loss computation for their algorithm.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        log_tag: str,
        save_file_name: str,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        gradient_max_norm: float = None,
    ) -> None:
        """Initializes the TorchFrequentGradientUpdate class with the given parameters.

        Args:
            optimizer (Optimizer): The optimizer to be used for the update.
            log_tag (str): The tag used for logging the resulting loss
            save_file_name (str): The name of the file to save the state of the component to.
            update_frequency (int, optional): The frequency of the update according to the number of environment interactions.
            If the update frequency is -1, updates will only be performed at the end of an episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to be performed when the update condition is met. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            gradient_max_norm (float, optional): The maximum norm for the gradients. If None, no gradient clipping is performed. Defaults to None.
        """
        UpdatableComponent.__init__(self)

        self.optimizer = optimizer
        self.update_frequency = update_frequency
        self.log_tag = log_tag
        self.save_file_name = save_file_name
        self.number_of_updates = number_of_updates
        self.multiply_number_of_updates_by_environment_steps = (
            multiply_number_of_updates_by_environment_steps
        )

        if gradient_max_norm:
            self.gradient_manipulation_function = (
                lambda: torch.nn.utils.clip_grad_norm_(
                    parameters=[
                        parameter
                        for group in self.optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    max_norm=gradient_max_norm,
                )
            )
        else:
            self.gradient_manipulation_function = lambda: None

        self.step_tracker = StepTracker.get_instance()

        self._last_update_step = 0
        self._last_update_episode = 0

    def update(self) -> Dict[str, Any]:
        """Performing a loss driven update with a torch optimizer following the defined loss function.

        Returns:
            Dict[str, Any]: Logging data from the update containing the average loss
            and potential additional information added in the post_update_routine.
        """
        losses = []
        number_of_updates = (
            self.number_of_updates
            if not self.multiply_number_of_updates_by_environment_steps
            else self.number_of_updates
            * (self.step_tracker.interactions_after_warmup - self._last_update_step)
        )
        self._last_update_step = self.step_tracker.interactions_after_warmup
        self._last_update_episode = self.step_tracker.total_number_of_episodes
        for _ in range(number_of_updates):

            loss = self.calculate_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.gradient_manipulation_function()
            self.optimizer.step()

            losses.append(loss.item())

        log_data = {self.log_tag: np.mean(losses).item()}
        log_data.update(self.post_update_routine())
        return log_data

    def post_update_routine(self) -> Dict[str, Any]:
        """
        This method is called after the update step.
        It can be used to perform any additional operations needed after the update and to add
        to the logging information.

        Returns:
            Dict[str, Any]: A dictionary containing additional logging information.
        """
        return {}

    @abstractmethod
    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Abstract method to be implemented in the derived class. This function should
        calculate the loss to be used for the update. All necessary dependencies should be provided
        in the constructor of the derived class such that no arguments are needed.

        Returns:
            torch.Tensor: The calculated loss tensor as a scalar.
        """
        ...

    @property
    def update_condition(self) -> bool:
        if self.update_frequency > 0:
            # Update if training frequency is met
            return (
                self.step_tracker.is_warmup_done
                and (
                    self.step_tracker.interactions_after_warmup % self.update_frequency
                    == 0
                )  # But only if the effective number of updates is > 0
                and (
                    not self.multiply_number_of_updates_by_environment_steps
                    or (
                        self.step_tracker.interactions_after_warmup
                        > self._last_update_step
                    )
                )
            )
        # If update_frequency is <= 0, we update when an episode ends
        return (
            self.step_tracker.is_warmup_done
            and (self.step_tracker.total_number_of_episodes > self._last_update_episode)
            and self.step_tracker.interactions_after_warmup > self._last_update_step
        )

    def save_checkpoint(self, context: SaveContext):
        save_path = os.path.join(
            context.save_path, context.prefix + self.save_file_name
        )
        handling_stats = (self._last_update_step, self._last_update_episode)

        context.file_handler.save_to_file(to_save=handling_stats, save_path=save_path)

    def load_checkpoint(self, context: SaveContext):
        save_path = os.path.join(
            context.save_path, context.prefix + self.save_file_name
        )
        handling_stats = context.file_handler.load_from_file(load_path=save_path)

        self._last_update_step, self._last_update_episode = handling_stats


class TargetNetUpdate(UpdatableComponent):
    """Component responsible for updating target networks in deep RL algorithms.

    This class handles target network updates, which are crucial for stabilizing training
    in many deep reinforcement learning algorithms such as DQN, DDPG, and SAC. It supports
    both hard updates (direct copying of weights) and soft updates (exponential moving average)
    depending on the tau parameter.
    """

    def __init__(
        self,
        source_net: torch.nn.Module,
        target_net: torch.nn.Module,
        tau: Optional[float] = None,
        update_frequency: int = 1,
    ) -> None:
        """Initializes the TargetNetUpdate class with the given parameters.

        Args:
            source_net (torch.nn.Module): The source network to be used for the update.
            target_net (torch.nn.Module): The target network to be updated.
            tau (Optional[float], optional): Tau value for soft update. If None, a hard update is performed. Defaults to None.
            update_frequency (int, optional): Frequency of the update according to the number of environment interactions. Defaults to 1.
        """
        super().__init__()

        self.source_net = source_net
        self.target_net = target_net
        self.step_tracker = StepTracker.get_instance()
        self.tau = tau
        self.update_frequency = update_frequency
        if tau == None or tau == 1.0:
            self.update = self._hard_update
        else:
            self.update = self._soft_update

    def update(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "This method should be monkey patched in the initialization of the class."
        )

    def _soft_update(self) -> Dict[str, Any]:
        for target_param, param in zip(
            self.target_net.parameters(), self.source_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        return {}

    def _hard_update(self) -> Dict[str, Any]:
        self.target_net.load_state_dict(self.source_net.state_dict())
        return {}

    @property
    def update_condition(self) -> bool:
        # True if warmup is done and the update frequency is met
        return (
            self.step_tracker.is_warmup_done
            and self.step_tracker.interactions_after_warmup % self.update_frequency == 0
        )


class ReplayBufferUpdate(UpdatableComponent):
    """Component that transfers collected experience data into replay buffers.

    This class serves as the bridge between data collection and the replay buffer,
    checking for new data from the UpdateDataProvider and adding it to the buffer.
    It automatically tracks which datapoints have been processed to avoid duplicates.
    """

    def __init__(
        self,
        update_data_provider: UpdateDataProvider,
        replay_buffer: Buffer,
    ) -> None:
        """Initializes a replay buffer updatable component.


        Args:
            update_data_provider (UpdateDataProvider): UpdateDataProvider instance to provide the data for the replay buffer.
            replay_buffer (Buffer): The replay buffer itself to add the data to.
        """
        super().__init__()

        self.update_data_provider = update_data_provider
        self.replay_buffer = replay_buffer
        self.step_tracker = StepTracker.get_instance()
        self.datapoints_added = 0

    def update(self) -> Dict[str, Any]:
        """Adds data from the update data provider to the replay buffer.

        Returns:
            Dict[str, Any]: {}
        """

        replay_data, metadata = self.update_data_provider.get_data()
        self.replay_buffer.add(data_dictionary=replay_data, metadata=metadata)
        return {}

    @property
    def update_condition(self) -> bool:
        """True if there is a new datapoint since the last update."""
        return (
            self.step_tracker.total_number_of_datapoints_added > self.datapoints_added
        )
