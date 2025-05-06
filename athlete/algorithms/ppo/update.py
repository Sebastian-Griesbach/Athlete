from typing import Type, Dict, Any, Optional, Tuple

import torch

from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.data_collection.provider import UpdateDataProvider
from athlete.algorithms.ppo.module import PPOActor
from athlete.saving.saveable_component import CompositeSaveableComponent
from athlete.algorithms.ppo.updatable_components import (
    PPOBufferUpdate,
    PPOGradientUpdate,
)
from athlete.update.on_policy_buffer import OnPolicyBuffer


class PPOUpdate(UpdateRule, CompositeSaveableComponent):
    """The Update rule for PPO.
    This class manages all updatable components and the saving and loading of stateful objects.
    """

    def __init__(
        self,
        update_data_provider: UpdateDataProvider,
        value_function: torch.nn.Module,
        actor: PPOActor,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_arguments: Dict[str, Any],
        discount: float,
        generalized_advantage_estimation_lambda: float,
        epochs_per_update: int,
        mini_batch_size: int,
        policy_ratio_clip_value: float,
        value_loss_clip_value: Optional[float],
        value_loss_coefficient: float,
        entropy_loss_coefficient: float,
        batch_normalize_advantage: bool,
        gradient_max_norm: Optional[float],
        device: str,
    ):
        """Initializes the PPO update rule.

        Args:
            update_data_provider (UpdateDataProvider): The data provider used to communicate with the data collector.
            value_function (torch.nn.Module): The value function network.
            actor (PPOActor): The actor network.
            optimizer_class (Type[torch.optim.Optimizer]): The optimizer class to use for the value function and actor networks.
            optimizer_arguments (Dict[str, Any]): The initialization arguments for the optimizer of actor and value function. Network parameters are added automatically.
            discount (float): The discount factor used for return calculation.
            generalized_advantage_estimation_lambda (float): The lambda parameter for generalized advantage estimation.
            epochs_per_update (int): The number of epochs to perform for each update.
            mini_batch_size (int): The mini-batch size for sampling from the buffer.
            policy_ratio_clip_value (float): The clip value for the policy ratio.
            value_loss_clip_value (Optional[float]): The clip value for the value loss. None means no clipping.
            value_loss_coefficient (float): The coefficient for the value loss in the total loss.
            entropy_loss_coefficient (float): The coefficient for the entropy loss in the total loss.
            batch_normalize_advantage (bool): Whether to batch normalize the advantage before calculating the loss.
            gradient_max_norm (Optional[float]): The maximum norm for gradient clipping. None means no clipping.
            device (str): The device to use for the networks (e.g., "cpu" or "cuda").
        """
        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        self.value_function = value_function.to(device)
        self.actor = actor.to(device)

        self.optimizer = optimizer_class(
            [
                {
                    "params": self.actor.parameters(),
                    **optimizer_arguments,
                },
                {
                    "params": self.value_function.parameters(),
                    **optimizer_arguments,
                },
            ]
        )

        self.register_saveable_component("value_function", self.value_function)
        self.register_saveable_component("actor", self.actor)
        self.register_saveable_component("optimizer", self.optimizer)

        # PPO Buffer Update
        self.buffer = OnPolicyBuffer(mini_batch_size=mini_batch_size)

        self.buffer_update = PPOBufferUpdate(
            update_data_provider=update_data_provider,
            value_function=self.value_function,
            discount=discount,
            generalized_advantage_estimation_lambda=generalized_advantage_estimation_lambda,
            on_policy_buffer=self.buffer,
            changes_policy=False,
            save_file_name=PPOBufferUpdate.SAVE_FILE_NAME,
        )

        self.register_saveable_component("ppo_buffer_update", self.buffer_update)

        # PPO Gradient Update

        self.gradient_update = PPOGradientUpdate(
            value_function=self.value_function,
            actor=self.actor,
            optimizer=self.optimizer,
            on_policy_buffer=self.buffer,
            epochs_per_update=epochs_per_update,
            policy_ratio_clip_value=policy_ratio_clip_value,
            value_loss_clip_value=value_loss_clip_value,
            value_loss_coefficient=value_loss_coefficient,
            entropy_loss_coefficient=entropy_loss_coefficient,
            batch_normalize_advantage=batch_normalize_advantage,
            gradient_max_norm=gradient_max_norm,
            policy_loss_log_tag=PPOGradientUpdate.POLICY_LOSS_LOG_TAG,
            value_loss_log_tag=PPOGradientUpdate.VALUE_LOSS_LOG_TAG,
            entropy_loss_log_tag=PPOGradientUpdate.ENTROPY_LOSS_LOG_TAG,
            total_loss_log_tag=PPOGradientUpdate.TOTAL_LOSS_LOG_TAG,
            changes_policy=True,
            save_file_name=PPOGradientUpdate.SAVE_FILE_NAME,
        )

        self.register_saveable_component("ppo_gradient_update", self.gradient_update)

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns a tuple of all updatable components of the PPO update rule in the order they should be updated in.

        Returns:
            Tuple[UpdatableComponent]: The updatable components of the PPO update rule:
                1. PPO buffer update
                2. PPO gradient update
        """
        return self.buffer_update, self.gradient_update
