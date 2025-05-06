from typing import Dict, Callable, Optional

import torch
from torch.optim import Optimizer

from athlete import constants
from athlete.update.common import TorchFrequentGradientUpdate


class TD3CriticUpdate(TorchFrequentGradientUpdate):
    """The updatable component for the TD3 critic update."""

    CRITIC_LOSS_LOG_TAG = "critic_loss"

    SAVE_HANDLING_STATS = "td3_value_update_handling_stats"

    def __init__(
        self,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        critic_1: torch.nn.Module,
        critic_2: torch.nn.Module,
        target_critic_1: torch.nn.Module,
        target_critic_2: torch.nn.Module,
        critic_optimizer: Optimizer,
        target_actor: torch.nn.Module,
        discount: float = 0.99,
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        changes_policy: bool = False,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        criteria: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        log_tag: str = CRITIC_LOSS_LOG_TAG,
        save_file_name: str = SAVE_HANDLING_STATS,
        gradient_max_norm: Optional[float] = None,
    ) -> None:
        """Initializes the TD3 critic update.

        Args:
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): A function that upon calling it returns a batch of data to use for updates in the form of a dictionary.
            critic_1 (torch.nn.Module): The first critic network to update.
            critic_2 (torch.nn.Module): The second critic network to update.
            target_critic_1 (torch.nn.Module): The target network for the first critic.
            target_critic_2 (torch.nn.Module): The target network for the second critic.
            critic_optimizer (Optimizer): The optimizer to use for both critic networks.
            target_actor (torch.nn.Module): The target actor network to use for the target value calculation.
            discount (float, optional): The discount factor for the value update. Defaults to 0.99.
            target_noise_std (float, optional): The standard deviation of the noise to add to the target actions. Defaults to 0.2.
            target_noise_clip (float, optional): The maximum absolute value of the noise to add to the target actions. Defaults to 0.5.
            changes_policy (bool, optional): Whether this update immediately changes the policy. For regular TD3 this is False. Defaults to False.
            update_frequency (int, optional): The update frequency according to the number of environment steps. If -1 an update is performed at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to perform at each update step. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            criteria (torch.nn.modules.loss._Loss, optional): The loss function used for calculating the loss between the critic predictions and the target values. Defaults to torch.nn.MSELoss().
            log_tag (str, optional): The log tag to use for logging the loss. Defaults to "critic_loss".
            save_file_name (str, optional): The file name to use for saving the handling stats. Defaults to "td3_value_update_handling_stats".
            gradient_max_norm (Optional[float], optional): The maximum norm for gradient clipping. If None, no clipping is performed. Defaults to None.
        """
        super().__init__(
            optimizer=critic_optimizer,
            changes_policy=changes_policy,
            log_tag=log_tag,
            save_file_name=save_file_name,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.critic_1 = critic_1
        self.target_critic_1 = target_critic_1
        self.critic_2 = critic_2
        self.target_critic_2 = target_critic_2
        self.target_actor = target_actor
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.discount = discount
        self.criteria = criteria
        self.data_sampler = data_sampler

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the critics loss according to the TD3 algorithm.

        Returns:
            torch.Tensor: The loss of the critics.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]
        actions = batch_dictionary[constants.DATA_ACTIONS]
        rewards = batch_dictionary[constants.DATA_REWARDS]
        next_observations = batch_dictionary[constants.DATA_NEXT_OBSERVATIONS]
        terminateds = batch_dictionary[constants.DATA_TERMINATEDS]

        # prediction
        prediction_1 = self.critic_1(observations, actions)
        prediction_2 = self.critic_2(observations, actions)

        # target
        with torch.no_grad():
            target = self._calculate_target(
                rewards=rewards,
                next_observations=next_observations,
                terminateds=terminateds,
            )

        loss_1 = self.criteria(prediction_1, target)
        loss_2 = self.criteria(prediction_2, target)

        return loss_1 + loss_2

    def _calculate_target(
        self,
        rewards: torch.tensor,
        next_observations: torch.tensor,
        terminateds: torch.tensor,
    ) -> torch.Tensor:
        """Calculates the target value for the critics according to the TD3 algorithm.

        Args:
            rewards (torch.tensor): Rewards batch of shape (batch_size, 1).
            next_observations (torch.tensor): Next observations batch of shape (batch_size, observation_size).
            terminateds (torch.tensor): Terminateds batch of shape (batch_size, 1).

        Returns:
            torch.Tensor: Target values of shape (batch_size, 1).
        """
        target = torch.clone(rewards)
        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)
        next_actions = self.target_actor(next_observations[not_terminateds])
        # according to TD3 paper, we add clipped noise to the target actions
        clipped_noise = torch.clip(
            torch.randn_like(next_actions) * self.target_noise_std,
            min=-self.target_noise_clip,
            max=self.target_noise_clip,
        )
        # We assume unscaled actions here, scaling is done in policy and unscaled actions are used for update
        noisy_next_actions = torch.clip(next_actions + clipped_noise, min=-1.0, max=1.0)

        next_q_values_1 = self.target_critic_1(
            next_observations[not_terminateds],
            noisy_next_actions,
        )
        next_q_values_2 = self.target_critic_2(
            next_observations[not_terminateds],
            noisy_next_actions,
        )
        min_next_q_values = torch.min(next_q_values_1, next_q_values_2)
        target[not_terminateds] += self.discount * min_next_q_values
        return target
