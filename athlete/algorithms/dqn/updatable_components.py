import torch
from typing import Dict, Callable

from athlete import constants
from athlete.update.common import TorchFrequentGradientUpdate


class DQNValueUpdate(TorchFrequentGradientUpdate):
    """The updatable component to update the Q-value function in DQN."""

    LOG_TAG_LOSS = "loss"

    SAVE_HANDLING_STATS = "dqn_value_update_handling_stats"

    def __init__(
        self,
        q_value_function: torch.nn.Module,
        target_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        cross_validation: bool = False,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        discount: float = 0.99,
        criteria: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        log_tag: str = LOG_TAG_LOSS,
        save_file_name: str = SAVE_HANDLING_STATS,
        gradient_max_norm: float = None,
    ) -> None:
        """Initializes the DQN value update component.

        Args:
            q_value_function (torch.nn.Module): Q-value function to be updated.
            target_net (torch.nn.Module): Target network to be used for calculating the target Q-values.
            optimizer (torch.optim.Optimizer): Optimizer to be used for updating the Q-value function.
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): A callable that returns a mini-batch of data
                in the form of a dictionary upon calling it.
            cross_validation (bool, optional): Weather to use cross-validation for the target calculation. Also
                known as double DQN. Defaults to False.
            update_frequency (int, optional): Update frequency of the Q-value function according to the number of
                environment steps. If -1, the update is performed at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to perform at each update step. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            discount (float, optional): Discount factor for the Q-value update. Defaults to 0.99.
            criteria (torch.nn.modules.loss._Loss, optional): Loss function to be used for the Q-value update. Defaults to torch.nn.MSELoss().
            log_tag (str, optional): Tag used for logging the loss. Defaults to 'loss'.
            save_file_name (str, optional): File name for saving the handling stats. Defaults to 'dqn_value_update_handling_stats'.
            gradient_max_norm (float, optional): Maximum norm for gradient clipping. Defaults to None.
        """
        super().__init__(
            optimizer=optimizer,
            log_tag=log_tag,
            save_file_name=save_file_name,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )
        self.value_function = q_value_function
        self.target_net = target_net
        self.discount = discount
        self.criteria = criteria
        self.data_sampler = data_sampler
        self.cross_validation = cross_validation

        if self.cross_validation:
            self._calculate_target = self._calculate_cross_validation_target

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the Q-value function according to DQN.

        Returns:
            torch.Tensor: The calculated loss.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]
        actions = batch_dictionary[constants.DATA_ACTIONS]
        rewards = batch_dictionary[constants.DATA_REWARDS]
        next_observations = batch_dictionary[constants.DATA_NEXT_OBSERVATIONS]
        terminateds = batch_dictionary[constants.DATA_TERMINATEDS]

        # prediction
        raw_prediction = self.value_function(observations)
        predicted_values = torch.gather(raw_prediction, dim=1, index=actions)

        # target
        with torch.no_grad():
            target = self._calculate_target(
                rewards=rewards,
                next_observations=next_observations,
                terminateds=terminateds,
            )

        loss = self.criteria(predicted_values, target)

        return loss

    def _calculate_target(
        self,
        rewards: torch.tensor,
        next_observations: torch.tensor,
        terminateds: torch.tensor,
    ) -> torch.Tensor:
        """Calculates the target for the Q-value function according to regular DQN.

        Args:
            rewards (torch.tensor): Batch of rewards.
            next_observations (torch.tensor): Batch of next observations.
            terminateds (torch.tensor): Batch of whether the next observation is terminal.

        Returns:
            torch.Tensor: Batch of calculated target Q-values.
        """
        target = torch.clone(rewards)
        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)
        next_q_values = torch.max(
            self.target_net(next_observations[not_terminateds]), dim=1, keepdim=True
        )[0]
        target[not_terminateds] += self.discount * next_q_values
        return target

    def _calculate_cross_validation_target(
        self,
        rewards: torch.tensor,
        next_observations: torch.tensor,
        terminateds: torch.tensor,
    ) -> torch.Tensor:
        """Calculates the target for the Q-value function according to double DQN.

        Args:
            rewards (torch.tensor): Batch of rewards.
            next_observations (torch.tensor): Batch of next observations.
            terminateds (torch.tensor): Batch of whether the next observation is terminal.

        Returns:
            torch.Tensor: Batch of calculated target Q-values.
        """
        target = torch.clone(rewards)
        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)
        next_actions = torch.argmax(
            self.value_function(next_observations[not_terminateds]), dim=1, keepdim=True
        )
        next_q_values = torch.gather(
            self.target_net(next_observations[not_terminateds]),
            dim=1,
            index=next_actions,
        )
        target[not_terminateds] += self.discount * next_q_values
        return target
