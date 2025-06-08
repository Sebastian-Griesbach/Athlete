from typing import Dict, Callable

import torch
from torch.optim import Optimizer

from athlete import constants
from athlete.update.common import TorchFrequentGradientUpdate


class DDPGCriticUpdate(TorchFrequentGradientUpdate):
    """Updatable component for the DDPG critic."""

    CRITIC_LOSS_LOG_TAG = "critic_loss"

    SAVE_HANDLING_STATS = "ddpg_critic_update_handling_stats"

    def __init__(
        self,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        critic_optimizer: Optimizer,
        target_actor: torch.nn.Module,
        discount: float = 0.99,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        criteria: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        log_tag: str = CRITIC_LOSS_LOG_TAG,
        save_file_name: str = SAVE_HANDLING_STATS,
        gradient_max_norm: float = None,
    ) -> None:
        """Initializes the DDPG critic update component.

        Args:
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): A callable that returns a batch of data in the form of a dictionary
                upon calling it.
            critic (torch.nn.Module): The critic network which is being updated.
            target_critic (torch.nn.Module): The target critic network which is used to calculate the target value.
            critic_optimizer (Optimizer): The Torch optimizer used to update the critic network.
            target_actor (torch.nn.Module): The target actor network which is used to calculate the target value.
            discount (float, optional): The discount factor for the target value. Defaults to 0.99.
            update_frequency (int, optional): The frequency of applying this update according to the environment steps. If -1, the update is applied at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to be applied at each update frequency. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether the number of updates is multiplied by the environment steps since the last update. Defaults to False.
            criteria (torch.nn.modules.loss._Loss, optional): The criteria to calculate the loss between the prediction and the target value. Defaults to torch.nn.MSELoss().
            log_tag (str, optional): The tag used for logging the loss. Defaults to critic_loss.
            save_file_name (str, optional): The file name used for saving the handling stats. Defaults to ddpg_critic_update_handling_stats.
            gradient_max_norm (float, optional): The maximum norm for the gradients. If None, no gradient clipping is applied. Defaults to None.
        """
        super().__init__(
            optimizer=critic_optimizer,
            log_tag=log_tag,
            save_file_name=save_file_name,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )
        self.critic = critic
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.discount = discount
        self.criteria = criteria
        self.data_sampler = data_sampler

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the DDPG critic according to the DDPG algorithm.

        Returns:
            torch.Tensor: The loss value calculated by the criteria between the prediction and the target value.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]
        actions = batch_dictionary[constants.DATA_ACTIONS]
        rewards = batch_dictionary[constants.DATA_REWARDS]
        next_observations = batch_dictionary[constants.DATA_NEXT_OBSERVATIONS]
        terminateds = batch_dictionary[constants.DATA_TERMINATEDS]

        # prediction
        prediction = self.critic(observations, actions)

        # target
        with torch.no_grad():
            target = self._calculate_target(
                rewards=rewards,
                next_observations=next_observations,
                terminateds=terminateds,
            )

        loss = self.criteria(prediction, target)

        return loss

    def _calculate_target(
        self,
        rewards: torch.tensor,
        next_observations: torch.tensor,
        terminateds: torch.tensor,
    ) -> torch.Tensor:
        """Calculates the target value for the DDPG critic according to the DDPG algorithm.
            Overwrite this function if you want to change the target value calculation.

        Args:
            rewards (torch.tensor): Rewards batch.
            next_observations (torch.tensor): Next observations batch.
            terminateds (torch.tensor): Terminateds batch.

        Returns:
            torch.Tensor: The target value calculated by the DDPG algorithm.
        """
        target = torch.clone(rewards)
        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)
        next_q_values = self.target_critic(
            next_observations[not_terminateds],
            self.target_actor(next_observations[not_terminateds]),
        )
        target[not_terminateds] += self.discount * next_q_values
        return target


class DDPGActorUpdate(TorchFrequentGradientUpdate):
    """Updatable component for the DDPG actor."""

    ACTOR_LOSS_LOG_TAG = "actor_loss"

    SAVE_HANDLING_STATS = "ddpg_actor_update_handling_stats"

    def __init__(
        self,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        actor: torch.nn.Module,
        actor_optimizer: torch.nn.Module,
        critic: torch.nn.Module,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        log_tag: str = ACTOR_LOSS_LOG_TAG,
        save_file_name: str = SAVE_HANDLING_STATS,
        gradient_max_norm: float = None,
    ) -> None:
        """Initializes the DDPG actor update component.

        Args:
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): A callable that returns a batch of data in the form of a dictionary
                upon calling it. For this update, it may only return the observations.
            actor (torch.nn.Module): The actor network which is being updated.
            actor_optimizer (torch.nn.Module): The Torch optimizer used to update the actor network.
            critic (torch.nn.Module): The critic network which is used to calculate the actor loss.
            update_frequency (int, optional): The frequency of applying this update according to the environment steps. If -1, the update is applied at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to be applied at each update frequency. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether the number of updates is multiplied by the environment steps since the last update. Defaults to False.
            log_tag (str, optional): The tag used for logging the loss. Defaults to actor_loss.
            save_file_name (str, optional): The file name used for saving the handling stats. Defaults to ddpg_actor_update_handling_stats.
            gradient_max_norm (float, optional): The maximum norm for the gradients. If None, no gradient clipping is applied. Defaults to None.
        """
        super().__init__(
            optimizer=actor_optimizer,
            log_tag=log_tag,
            save_file_name=save_file_name,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.critic = critic
        self.actor = actor
        self.data_sampler = data_sampler

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the DDPG actor according to the DDPG algorithm.

        Returns:
            torch.Tensor: The loss value as the negative value of the critic output.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]

        actions = self.actor(observations)
        values = self.critic(observations, actions)
        loss = -values.mean()

        return loss
