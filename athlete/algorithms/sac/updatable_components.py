import torch
from typing import Dict, Any, Callable

from athlete import constants
from athlete.update.common import TorchFrequentGradientUpdate
from athlete.algorithms.sac.module import SACActor


class SACCriticUpdate(TorchFrequentGradientUpdate):
    """The updatable component for the critic of the SAC algorithm. It updates the two critics"""

    CRITIC_LOSS_LOG_TAG = "critic_loss"

    def __init__(
        self,
        temperature: torch.nn.Parameter,
        actor: SACActor,
        critic_1: torch.nn.Module,
        critic_2: torch.nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        target_critic_1: torch.nn.Module,
        target_critic_2: torch.nn.Module,
        data_sampler: Callable[[None], Dict[str, torch.Tensor]],
        discount: float = 0.99,
        criteria: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        gradient_max_norm: float = None,
        log_tag: str = CRITIC_LOSS_LOG_TAG,
    ) -> None:
        """Initializes the SACCriticUpdate class.

        Args:
            temperature (torch.nn.Parameter): The temperature parameter for the SAC algorithm.
            actor (SACActor): The actor of the SAC algorithm.
            critic_1 (torch.nn.Module): The first critic to update.
            critic_2 (torch.nn.Module): The second critic to update.
            critic_optimizer (torch.optim.Optimizer): The optimizer for both critics.
            target_critic_1 (torch.nn.Module): Target network for the first critic.
            target_critic_2 (torch.nn.Module): Target network for the second critic.
            data_sampler (Callable[[None], Dict[str, torch.Tensor]]): A function that returns a batch of data to update on upon call.
            discount (float, optional): The discount factor for the SAC algorithm. Defaults to 0.99.
            criteria (torch.nn.modules.loss._Loss, optional): Function used to calculate the loss between the target and the predicted Q-values. Defaults to torch.nn.MSELoss().
            update_frequency (int, optional): The frequency in which this updated is applied according to the environment steps. If -1, it is applied at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to perform per call. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the environment steps since the last update. Defaults to False.
            gradient_max_norm (float, optional): The maximum norm for the gradients. If None, no gradient clipping is applied. Defaults to None.
            log_tag (str, optional): Log tag for the loss. Defaults to "critic_loss".
        """
        super().__init__(
            optimizer=critic_optimizer,
            log_tag=log_tag,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.temperature = temperature
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2
        self.data_sampler = data_sampler
        self.discount = discount
        self.criteria = criteria

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the critic loss according to the SAC algorithm.

        Returns:
            torch.Tensor: The loss of the critics.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]
        actions = batch_dictionary[constants.DATA_ACTIONS]
        rewards = batch_dictionary[constants.DATA_REWARDS]
        next_observations = batch_dictionary[constants.DATA_NEXT_OBSERVATIONS]
        terminateds = batch_dictionary[constants.DATA_TERMINATEDS]

        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)

        with torch.no_grad():
            next_actions, next_log_probabilities = self.actor.get_action_and_log_prob(
                next_observations[not_terminateds]
            )
            next_q_values_1 = self.target_critic_1(
                next_observations[not_terminateds], next_actions
            )
            next_q_values_2 = self.target_critic_2(
                next_observations[not_terminateds], next_actions
            )
            min_next_q_values = torch.min(next_q_values_1, next_q_values_2)
            min_next_q_values_with_entropy = (
                min_next_q_values - self.temperature * next_log_probabilities
            )
            target_q_values = torch.clone(rewards)
            target_q_values[not_terminateds] += (
                self.discount * min_next_q_values_with_entropy
            )

        q_value_predictions_1 = self.critic_1(observations, actions)
        q_value_predictions_2 = self.critic_2(observations, actions)
        loss_1 = self.criteria(q_value_predictions_1, target_q_values)
        loss_2 = self.criteria(q_value_predictions_2, target_q_values)
        return loss_1 + loss_2


class SACActorUpdate(TorchFrequentGradientUpdate):
    """The updatable component for the actor of the SAC algorithm. It updates the actor network."""

    ACTOR_LOSS_LOG_TAG = "actor_loss"

    def __init__(
        self,
        actor: SACActor,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: torch.nn.Module,
        critic_2: torch.nn.Module,
        temperature: torch.nn.Parameter,
        data_sampler: Callable[[None], Dict[str, torch.Tensor]],
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        gradient_max_norm: float = None,
        log_tag: str = ACTOR_LOSS_LOG_TAG,
    ) -> None:
        """Initializes the SAC actor update class.

        Args:
            actor (SACActor): The actor to update.
            actor_optimizer (torch.optim.Optimizer): The optimizer of the actor.
            critic_1 (torch.nn.Module): The first critic to use for the update.
            critic_2 (torch.nn.Module): The second critic to use for the update.
            temperature (torch.nn.Parameter): The temperature parameter for the SAC algorithm.
            data_sampler (Callable[[None], Dict[str, torch.Tensor]]): A function that returns a batch of data to update on upon call.
                Here, the batch only needs to contain the observations.
            update_frequency (int, optional): The frequency in which this updated is applied according to the environment steps. If -1, it is applied at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to perform per call. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the environment steps since the last update. Defaults to False.
            gradient_max_norm (float, optional): The maximum norm for the gradients. If None, no gradient clipping is applied. Defaults to None.
            log_tag (str, optional): Log tag for the loss. Defaults to "actor_loss".
        """
        super().__init__(
            optimizer=actor_optimizer,
            log_tag=log_tag,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.temperature = temperature
        self.data_sampler = data_sampler

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the actor loss according to the SAC algorithm.

        Returns:
            torch.Tensor: The loss of the actor.
        """
        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]

        actions, log_probabilities = self.actor.get_action_and_log_prob(observations)
        q_values_1 = self.critic_1(observations, actions)
        q_values_2 = self.critic_2(observations, actions)
        min_q_values = torch.min(q_values_1, q_values_2)
        loss = torch.mean((self.temperature * log_probabilities) - min_q_values)
        return loss


class SACTemperatureUpdate(TorchFrequentGradientUpdate):
    """The updatable component for the temperature of the SAC algorithm. It updates the temperature parameter
    and can be used if the temperature should be adjusted automatically.
    """

    TEMPERATURE_LOSS_LOG_TAG = "temperature_loss"
    TEMPERATURE_LOG_TAG = "temperature"

    def __init__(
        self,
        target_entropy: float,
        log_temperature: torch.nn.Parameter,
        temperature: torch.nn.Parameter,
        temperature_optimizer: torch.optim.Optimizer,
        actor: SACActor,
        data_sampler: Callable[[None], Dict[str, torch.Tensor]],
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        gradient_max_norm: float = None,
        temperature_log_tag: str = TEMPERATURE_LOG_TAG,
        loss_log_tag: str = TEMPERATURE_LOSS_LOG_TAG,
    ) -> None:
        """Initializes the SAC temperature update class.

        Args:
            target_entropy (float): The target entropy for the actor.
            log_temperature (torch.nn.Parameter): The log temperature parameter, this is updated.
            temperature (torch.nn.Parameter): The actual temperature parameter, this is updated by taking the exponential of the log temperature.
            temperature_optimizer (torch.optim.Optimizer): The optimizer for the temperature parameter.
            actor (SACActor): The stochastic actor of the SAC algorithm.
            data_sampler (Callable[[None], Dict[str, torch.Tensor]]): A function that returns a batch of data to update on upon call, here it only needs to return the observations.
            update_frequency (int, optional): The frequency in which this updated is applied according to the environment steps. If -1, it is applied at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to perform per call. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the environment steps since the last update. Defaults to False.
            gradient_max_norm (float, optional): The maximum norm for the gradients. If None, no gradient clipping is applied. Defaults to None.
            temperature_log_tag (str, optional): Log tag used for the log temperature loss. Defaults to "temperature_loss".
            loss_log_tag (str, optional): Log tag used to log the current temperature. Defaults to "temperature".
        """
        super().__init__(
            optimizer=temperature_optimizer,
            log_tag=loss_log_tag,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.target_entropy = target_entropy
        self.log_temperature = log_temperature
        self.temperature = temperature
        self.actor = actor
        self.data_sampler = data_sampler

        self.number_of_updates = number_of_updates
        self.update_per_environment_interaction = (
            multiply_number_of_updates_by_environment_steps
        )

        self.temperature_log_tag = temperature_log_tag
        self.loss_log_tag = loss_log_tag

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the temperature loss according to the target entropy.

        Returns:
            torch.Tensor: The loss of the temperature.
        """
        batch_dictionary = self.data_sampler()
        observations = batch_dictionary[constants.DATA_OBSERVATIONS]

        with torch.no_grad():
            _, log_probabilities = self.actor.get_action_and_log_prob(observations)
        loss = torch.mean(
            -torch.exp(self.log_temperature) * (log_probabilities + self.target_entropy)
        )
        return loss

    def post_update_routine(self) -> Dict[str, Any]:
        """Updates the actual temperature as the exponential of the log temperature
        and add the current temperature to the logging dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing the current temperature.
        """
        # Update actual temperature value used by other components
        self.temperature.data = torch.exp(self.log_temperature)
        return {self.temperature_log_tag: self.temperature.item()}
