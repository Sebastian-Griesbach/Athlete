from typing import Optional
import os
from collections import defaultdict

import torch

from athlete.update.update_rule import UpdatableComponent
from athlete.data_collection.provider import UpdateDataProvider
from athlete import constants
from athlete.update.on_policy_buffer import OnPolicyBuffer
from athlete.data_collection.provider import UpdateDataProvider
from athlete.update.update_rule import UpdatableComponent
from athlete.function import numpy_to_tensor
from athlete.algorithms.ppo.module import PPOActor
from athlete.global_objects import StepTracker


class PPOBufferUpdate(UpdatableComponent):
    """Component that prepares collected trajectory data for PPO algorithm training.

    This component transforms raw experience data received from the data collector into
    the format needed for PPO updates. It computes values for observations, calculates
    advantages using Generalized Advantage Estimation (GAE), and computes returns.
    The processed data is then stored in an on-policy buffer for use in subsequent
    gradient updates.
    """

    # Prepares Data and puts it into the buffer
    def __init__(
        self,
        update_data_provider: UpdateDataProvider,
        value_function: torch.nn.Module,
        discount: float,
        generalized_advantage_estimation_lambda: float,
        on_policy_buffer: OnPolicyBuffer,
        # TODO continue updating step tracker implementations (updateable components, and update rules, no more saving required)
    ) -> None:
        """Initializes the PPOBufferUpdate class.

        Args:
            update_data_provider (UpdateDataProvider): The data provider used to communicate with the data collector.
            value_function (torch.nn.Module): The value function used to calculate the value of the observations.
            discount (float): The discount factor used to calculate the return.
            generalized_advantage_estimation_lambda (float): The lambda value used for generalized advantage estimation.
            on_policy_buffer (OnPolicyBuffer): The on-policy buffer used to store the data.
        """
        super().__init__()
        self.update_data_provider = update_data_provider
        self.value_function = value_function
        self.discount = discount
        self.generalized_advantage_estimation_lambda = (
            generalized_advantage_estimation_lambda
        )

        self.on_policy_buffer = on_policy_buffer

        self.device = value_function.parameters().__next__().device
        self.step_tracker = StepTracker.get_instance()
        self._last_datapoint_updated_on_tracker_id = self.step_tracker.register_tracker(
            id="ppo_buffer_last_datapoint_updated_on"
        )

    def update(self):
        """Takes the data from the data collector and prepares it to be used by the PPO update.
            The prepared data is then put into the on-policy buffer.
        The data is prepared by calculating the values, advantages and returns which are needed for the PPO update.
        """
        update_data = self.update_data_provider.get_data()[0]

        observations = numpy_to_tensor(
            update_data[constants.DATA_OBSERVATIONS], device=self.device
        )
        actions = numpy_to_tensor(
            update_data[constants.DATA_ACTIONS], device=self.device
        )
        log_probs = numpy_to_tensor(
            update_data[constants.DATA_LOG_PROBS], device=self.device
        )
        rewards = numpy_to_tensor(
            update_data[constants.DATA_REWARDS], device=self.device
        )
        next_dones = numpy_to_tensor(
            update_data[constants.DATA_NEXT_DONES], device=self.device
        )

        all_values = self.value_function(observations)
        values = all_values[:-1]
        next_values = all_values[1:]

        next_not_dones = ~next_dones
        advantages = torch.zeros_like(all_values)

        # Calculate the return using Generalized Advantage Estimation (GAE)
        with torch.no_grad():

            # Calculate advantages
            temporal_differences = (
                rewards + self.discount * next_values * next_not_dones - values
            )

            for step in reversed(range(len(values))):
                advantages[step] = temporal_differences[
                    step
                ] + self.discount * self.generalized_advantage_estimation_lambda * (
                    advantages[step + 1] * next_not_dones[step]
                )
            # advantages is one step longer for the initial (the last entry) 0 value
            advantages = advantages[:-1]
            returns = advantages + values

        data = {
            constants.DATA_OBSERVATIONS: observations[
                :-1
            ],  # the last observation is not used
            constants.DATA_ACTIONS: actions,
            constants.DATA_LOG_PROBS: log_probs,
            constants.DATA_VALUES: values,
            constants.DATA_ADVANTAGES: advantages,
            constants.DATA_RETURNS: returns,
        }

        self.on_policy_buffer.set_data(data_dict=data)

        # remember when the last update was done, used for update condition
        self.step_tracker.set_tracker_value(
            id=self._last_datapoint_updated_on_tracker_id,
            value=self.step_tracker.get_tracker_value(id=constants.TRACKER_DATA_POINTS),
        )

    @property
    def update_condition(self) -> bool:
        """Whether to perform the update or not.

        Returns:
            bool: True if the the data collector has finished collecting a new batch of data (a datapoint).
        """
        return self.step_tracker.get_tracker_value(
            id=constants.TRACKER_DATA_POINTS
        ) > self.step_tracker.get_tracker_value(
            id=self._last_datapoint_updated_on_tracker_id
        )


# TODO continue updating the implementation of step tracker


class PPOGradientUpdate(UpdatableComponent):
    """Component responsible for performing PPO's policy gradient and value function updates.

    This component implements the core Proximal Policy Optimization algorithm update mechanism,
    which includes computing clipped surrogate objectives for policy updates, value function
    losses, and optional entropy regularization. It handles multiple epochs of updates on
    batched data from the on-policy buffer.
    """

    POLICY_LOSS_LOG_TAG = "policy_loss"
    VALUE_LOSS_LOG_TAG = "value_loss"
    ENTROPY_LOSS_LOG_TAG = "entropy_loss"
    TOTAL_LOSS_LOG_TAG = "loss"

    ZERO_DIVISION_CONSTANT = 1e-8

    def __init__(
        self,
        value_function: torch.nn.Module,
        actor: PPOActor,
        optimizer: torch.optim.Optimizer,
        on_policy_buffer: OnPolicyBuffer,
        epochs_per_update: int = 10,
        policy_ratio_clip_value: float = 0.2,
        value_loss_clip_value: Optional[float] = 0.2,  # None means no clipping
        value_loss_coefficient: float = 0.5,
        entropy_loss_coefficient: float = 0.00,
        batch_normalize_advantage: bool = False,
        gradient_max_norm: float = None,
        # Logging tags
        policy_loss_log_tag: str = POLICY_LOSS_LOG_TAG,
        value_loss_log_tag: str = VALUE_LOSS_LOG_TAG,
        entropy_loss_log_tag: str = ENTROPY_LOSS_LOG_TAG,
        total_loss_log_tag: str = TOTAL_LOSS_LOG_TAG,
    ) -> None:
        """Initializes the PPO gradient updatable component.

        Args:
            value_function (torch.nn.Module): Value function to be updated.
            actor (PPOActor): Actor to be updated.
            optimizer (torch.optim.Optimizer): Optimizer to be used for updating the actor and value function.
            on_policy_buffer (OnPolicyBuffer): The on-policy buffer used to sample the shuffled mini-batches from.
            epochs_per_update (int, optional): How many epochs to perform the update. Defaults to 10.
            policy_ratio_clip_value (float, optional): The value used to clip the policy ratio. Defaults to 0.2.
            value_loss_clip_value (Optional[float], optional): The value used to clip the value loss. Defaults to None.
            entropy_loss_coefficient (float, optional): The coefficient used to weight the entropy loss. Defaults to 0.0.
            batch_normalize_advantage (bool, optional): Whether to batch normalize the advantages. Defaults to False.
            gradient_max_norm (float, optional): The maximum norm for the gradients. Defaults to None.
            policy_loss_log_tag (str, optional): The tag used for logging the policy loss. Defaults to 'policy_loss'.
            value_loss_log_tag (str, optional): The tag used for logging the value loss. Defaults to 'value_loss'.
            entropy_loss_log_tag (str, optional): The tag used for logging the entropy loss. Defaults to 'entropy_loss'.
            total_loss_log_tag (str, optional): The tag used for logging the total loss. Defaults to 'loss'.
        """
        super().__init__()
        self.value_function = value_function
        self.on_policy_buffer = on_policy_buffer
        self.actor = actor
        self.optimizer = optimizer

        self.epochs_per_update = epochs_per_update
        self.loss_clip_epsilon = policy_ratio_clip_value
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_loss_clip_value = value_loss_clip_value
        self.clip_value_loss = self.value_loss_clip_value is not None
        self.batch_normalize_advantage = batch_normalize_advantage

        self.step_tracker = StepTracker.get_instance()
        self._last_datapoint_updated_on_tracker_id = self.step_tracker.register_tracker(
            id="ppo_update_last_datapoint_updated_on"
        )

        self.policy_loss_log_tag = policy_loss_log_tag
        self.value_loss_log_tag = value_loss_log_tag
        self.entropy_loss_log_tag = entropy_loss_log_tag
        self.total_loss_log_tag = total_loss_log_tag

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

    def update(self) -> None:
        """Performing the PPO update."""

        total_logging_info = defaultdict(list)

        for epoch in range(self.epochs_per_update):
            for batch in self.on_policy_buffer.generate_shuffled_batched_epoch():
                observations = batch[constants.DATA_OBSERVATIONS]
                actions = batch[constants.DATA_ACTIONS]
                old_log_probs = batch[constants.DATA_LOG_PROBS]
                old_values = batch[constants.DATA_VALUES]
                advantages = batch[constants.DATA_ADVANTAGES]
                returns = batch[constants.DATA_RETURNS]

                loss, batch_logging_info = self._calculate_loss(
                    observations=observations,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    old_values=old_values,
                    advantages=advantages,
                    returns=returns,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.gradient_manipulation_function()
                self.optimizer.step()

                for key, value in batch_logging_info.items():
                    total_logging_info[key].append(value)

        for key, value in total_logging_info.items():
            total_logging_info[key] = sum(value) / len(value)

        self.step_tracker.set_tracker_value(
            id=self._last_datapoint_updated_on_tracker_id,
            value=self.step_tracker.get_tracker_value(id=constants.TRACKER_DATA_POINTS),
        )

        return total_logging_info

    def _calculate_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the loss for the PPO update.

        Args:
            observations (torch.Tensor): Mini-batch of observations.
            actions (torch.Tensor): Mini-batch of actions.
            old_log_probs (torch.Tensor): Mini-batch of log probabilities of the actions taken.
            old_values (torch.Tensor): Mini-batch of values of the observations.
            advantages (torch.Tensor): Mini-batch of advantages.
            returns (torch.Tensor): Mini-batch of returns.

        Returns:
            torch.Tensor: The calculated loss.
        """

        new_log_prob, entropy = self.actor.get_log_prob_and_entropy(
            observations=observations, actions=actions
        )

        new_values = self.value_function(observations)

        # Clipped Advantage loss
        log_ratio = new_log_prob - old_log_probs
        # Ratio between old policy before any updates and new policy to take these actions
        likelihood_ratio = log_ratio.exp()

        if self.batch_normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + self.ZERO_DIVISION_CONSTANT
            )

        # We negate the loss as in the original paper the loss is maximized, but
        # pytorch minimizes the loss by default
        policy_loss = -torch.min(
            likelihood_ratio * advantages,
            torch.clamp(
                likelihood_ratio,
                1 - self.loss_clip_epsilon,
                1 + self.loss_clip_epsilon,
            )
            * advantages,
        ).mean()

        # Value loss
        value_loss = (new_values - returns).pow(2)
        if self.clip_value_loss:
            clipped_values = old_values + torch.clamp(
                new_values - old_values,
                -self.value_loss_clip_value,
                self.value_loss_clip_value,
            )
            clipped_value_loss = (clipped_values - returns).pow(2)
            # Assuming through out the update phase, the unclamped loss decreases,
            # the clipped value loss will be greater as it uses mainly the old values,
            # It does not actually clip the loss, but it clips the influence of the
            # new value estimate and as such the resulting gradients to limit the size of the
            # update step, similar to how we limit the policy update step with the min operator
            value_loss = torch.max(value_loss, clipped_value_loss)
        # 0.5 due to the derivation of the MSE loss, probably does not have any affect but is often included
        # in vanilla PPO implementations
        value_loss = 0.5 * value_loss.mean()

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_loss_coefficient * value_loss
            + self.entropy_loss_coefficient * entropy_loss
        )

        logging_info = {
            self.policy_loss_log_tag: policy_loss.item(),
            self.value_loss_log_tag: value_loss.item(),
            self.entropy_loss_log_tag: entropy_loss.item(),
            self.total_loss_log_tag: total_loss.item(),
        }

        return total_loss, logging_info

    @property
    def update_condition(self) -> bool:
        """When to perform the update.

        Returns:
            bool: True when new data is available and the warmup is done.
        """
        return self.step_tracker.is_warmup_done and self.step_tracker.get_tracker_value(
            id=constants.TRACKER_DATA_POINTS
        ) > self.step_tracker.get_tracker_value(
            id=self._last_datapoint_updated_on_tracker_id
        )
