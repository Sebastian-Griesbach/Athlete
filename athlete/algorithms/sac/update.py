from typing import Callable, Dict, Any, Type, Optional, Union, Tuple
import copy

import torch
from torch import nn
from torch.optim import Optimizer
from gymnasium.spaces import Space
import numpy as np

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.sac.updatable_components import (
    SACCriticUpdate,
    SACActorUpdate,
    SACTemperatureUpdate,
)
from athlete.update.common import TargetNetUpdate
from athlete.saving.saveable_component import CompositeSaveableComponent
from athlete.data_collection.provider import UpdateDataProvider
from athlete.update.buffer import EpisodicCPPReplayBuffer
from athlete.update.common import TargetNetUpdate, ReplayBufferUpdate
from athlete.update.buffer_wrapper import (
    PostBufferPreprocessingWrapper,
)
from athlete.function import extract_data_from_batch, create_transition_data_info
from athlete.algorithms.sac.module import SACActor


class SACUpdate(UpdateRule, CompositeSaveableComponent):
    """The Update Rule for the Soft Actor-Critic algorithm.
    This class is managing all updatable components and handles the saving of
    stateful components.
    """

    SETTING_AUTO = "auto"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        update_data_provider: UpdateDataProvider,
        critic_1: nn.Module,
        critic_2: nn.Module,
        actor: SACActor,
        discount: float,
        temperature: Union[float, str],
        target_entropy: Union[float, str],
        initial_temperature: float,
        critic_optimizer_class: Type[Optimizer],
        actor_optimizer_class: Type[Optimizer],
        critic_optimizer_arguments: Dict[str, Any],
        actor_optimizer_arguments: Dict[str, Any],
        critic_criteria: torch.nn.modules.loss._Loss,
        replay_buffer_capacity: int,
        replay_buffer_mini_batch_size: int,
        critic_update_frequency: int,
        critic_number_of_updates: int,
        actor_update_frequency: int,
        actor_number_of_updates: int,
        multiply_number_of_updates_by_environment_steps: bool,
        target_critic_update_frequency: int,
        target_critic_tau: Optional[float],
        critic_gradient_max_norm: Optional[float],
        actor_gradient_max_norm: Optional[float],
        device: str,
        additional_replay_buffer_arguments: Dict[str, Any],
        post_replay_buffer_data_preprocessing: Optional[Dict[str, Callable]],
    ) -> None:
        """Initializes the SAC update rule.

        Args:
            observation_space (Space): Observation space of the environment.
            action_space (Space): Action space of the environment.
            update_data_provider (UpdateDataProvider): The update data provider used to communicate with the data collector.
            critic_1 (nn.Module): The first critic network.
            critic_2 (nn.Module): The second critic network.
            actor (SACActor): The actor network.
            discount (float): Discount factor for the critic update.
            temperature (Union[float, str]): The temperature for the SAC algorithm. This can be a float value or "auto" for automatic temperature tuning.
            target_entropy (Union[float, str]): The target entropy for the SAC algorithm. This can be a float value or "auto" for automatic target entropy calculation.
            initial_temperature (float): The initial temperature for the SAC algorithm.
            critic_optimizer_class (Type[Optimizer]): The optimizer class for the critic networks.
            actor_optimizer_class (Type[Optimizer]): The optimizer class for the actor network.
            critic_optimizer_arguments (Dict[str, Any]): The initialization arguments for the critic optimizer without the critic network parameters.
            actor_optimizer_arguments (Dict[str, Any]): The initialization arguments for the actor optimizer without the actor network parameters.
            critic_criteria (torch.nn.modules.loss._Loss): The loss function used to calculate the loss between the prediction of the critic and the target value.
            replay_buffer_capacity (int): The maximum capacity of the replay buffer.
            replay_buffer_mini_batch_size (int): The mini-batch size for sampling from the replay buffer.
            critic_update_frequency (int): The frequency of the critic update according to the environment steps. If -1, updates will happen at the end of each episode.
            critic_number_of_updates (int): The number of critic updates per update step.
            actor_update_frequency (int): The frequency of the actor update according to the environment steps. If -1, updates will happen at the end of each episode.
            actor_number_of_updates (int): The number of actor updates per update step.
            multiply_number_of_updates_by_environment_steps (bool): Whether the number of updates should be multiplied by the environment steps since the last update.
            target_critic_update_frequency (int): The frequency of the target critic update according to the environment steps.
            target_critic_tau (Optional[float]): The soft update factor for the target critic networks, if None a hard update is performed.
            critic_gradient_max_norm (Optional[float]): The maximum norm for the critic gradients. If None, no gradient clipping is performed.
            actor_gradient_max_norm (Optional[float]): The maximum norm for the actor gradients. If None, no gradient clipping is performed.
            device (str): The device to use for the training. E.g. this can be "cpu" or "cuda".
            additional_replay_buffer_arguments (Dict[str, Any]): Additional initialization arguments for the replay buffer.
                This might be used for specific memory compression techniques.
            post_replay_buffer_data_preprocessing (Optional[Dict[str, Callable]]):
                A dictionary of functions to preprocess the data before passing it to the actor and critic networks after data has been sampled from the replay buffer.
                The keys of the dictionary should be the names of the data field to be preprocessed and the values should be the functions to be applied.
                This is used for example if the replay buffer holds int8 values to save memory but the networks expect float32 values.
        """

        super().__init__()

        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        # Updatable components

        self.critic_1 = critic_1.to(device)
        self.critic_2 = critic_2.to(device)
        self.actor = actor.to(device)

        self.critic_optimizer = critic_optimizer_class(
            [
                {
                    "params": critic_1.parameters(),
                    **critic_optimizer_arguments,
                },
                {
                    "params": critic_2.parameters(),
                    **critic_optimizer_arguments,
                },
            ]
        )

        self.actor_optimizer = actor_optimizer_class(
            params=self.actor.parameters(), **actor_optimizer_arguments
        )

        self.register_saveable_component("critic_1", self.critic_1)
        self.register_saveable_component("critic_2", self.critic_2)
        self.register_saveable_component("actor", self.actor)
        self.register_saveable_component("critic_optimizer", self.critic_optimizer)
        self.register_saveable_component("actor_optimizer", self.actor_optimizer)

        self.automatic_temperature_update = temperature == self.SETTING_AUTO

        if self.automatic_temperature_update:
            # requires grad false because this is just a mirror of log_temperature which is being updated
            self.temperature = torch.nn.Parameter(
                torch.tensor(initial_temperature), requires_grad=False
            )
            self.log_temperature = torch.nn.Parameter(
                torch.log(torch.tensor(initial_temperature, requires_grad=True))
            )
            # for simplicity we use the same optimizer setting for the temperature as for the critic when needed
            self.temperature_optimizer = critic_optimizer_class(
                params=[self.log_temperature], **critic_optimizer_arguments
            )

            # We only need to save these if they are dynamically updated
            self.register_saveable_component("temperature", self.temperature)
            self.register_saveable_component("log_temperature", self.log_temperature)
            self.register_saveable_component(
                "temperature_optimizer", self.temperature_optimizer
            )
        else:
            self.temperature = torch.nn.Parameter(
                torch.log(torch.tensor(self.temperature, requires_grad=False))
            )

        if target_entropy == self.SETTING_AUTO:
            self.target_entropy = float(-np.prod(action_space.shape).astype(np.float32))
        else:
            self.target_entropy = target_entropy

        self.target_critic_1 = copy.deepcopy(self.critic_1).eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).eval()
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)

        self.register_saveable_component("target_critic_1", self.target_critic_1)
        self.register_saveable_component("target_critic_2", self.target_critic_2)

        # Replay Buffer Update

        additional_arguments = {
            "next_of": constants.DATA_OBSERVATIONS,
        }
        additional_arguments.update(additional_replay_buffer_arguments)

        update_data_info = create_transition_data_info(
            observation_space=observation_space,
            action_space=action_space,
        )

        self.replay_buffer = EpisodicCPPReplayBuffer(
            capacity=replay_buffer_capacity,
            replay_buffer_info=update_data_info,
            additional_arguments=additional_arguments,
        )

        self.register_saveable_component("replay_buffer", self.replay_buffer)

        self.replay_buffer_update = ReplayBufferUpdate(
            update_data_provider=update_data_provider,
            replay_buffer=self.replay_buffer,
        )

        if post_replay_buffer_data_preprocessing is not None:
            sample_replay_buffer = PostBufferPreprocessingWrapper(
                replay_buffer=self.replay_buffer,
                post_replay_buffer_data_preprocessing=post_replay_buffer_data_preprocessing,
            )
        else:
            sample_replay_buffer = self.replay_buffer

        # Critic Update

        critic_data_keys = list(update_data_info.keys())
        critic_data_sampler_function = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=critic_data_keys,
            device=device,
        )

        self.critic_update = SACCriticUpdate(
            temperature=self.temperature,
            actor=self.actor,
            critic_1=self.critic_1,
            critic_2=self.critic_2,
            critic_optimizer=self.critic_optimizer,
            target_critic_1=self.target_critic_1,
            target_critic_2=self.target_critic_2,
            data_sampler=critic_data_sampler_function,
            discount=discount,
            criteria=critic_criteria,
            update_frequency=critic_update_frequency,
            number_of_updates=critic_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            changes_policy=False,
            gradient_max_norm=critic_gradient_max_norm,
            log_tag=SACCriticUpdate.CRITIC_LOSS_LOG_TAG,
            save_file_name=SACCriticUpdate.SAVE_HANDLING_STATS,
        )

        # Actor Update
        actor_data_keys = [constants.DATA_OBSERVATIONS]
        actor_data_sampler_function = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=actor_data_keys,
            device=device,
        )

        self.actor_update = SACActorUpdate(
            actor=self.actor,
            actor_optimizer=self.actor_optimizer,
            critic_1=self.critic_1,
            critic_2=self.critic_2,
            temperature=self.temperature,
            data_sampler=actor_data_sampler_function,
            update_frequency=actor_update_frequency,
            number_of_updates=actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=actor_gradient_max_norm,
            changes_policy=True,
            log_tag=SACActorUpdate.ACTOR_LOSS_LOG_TAG,
            save_file_name=SACActorUpdate.SAVE_HANDLING_STATS,
        )

        if self.automatic_temperature_update:
            # Temperature Update
            self.temperature_update = SACTemperatureUpdate(
                target_entropy=self.target_entropy,
                log_temperature=self.log_temperature,
                temperature=self.temperature,
                temperature_optimizer=self.temperature_optimizer,
                actor=self.actor,
                data_sampler=actor_data_sampler_function,
                update_frequency=critic_update_frequency,
                number_of_updates=critic_number_of_updates,
                multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
                changes_policy=False,
                gradient_max_norm=None,
                temperature_log_tag=SACTemperatureUpdate.TEMPERATURE_LOG_TAG,
                loss_log_tag=SACTemperatureUpdate.TEMPERATURE_LOSS_LOG_TAG,
                save_file_name=SACTemperatureUpdate.SAVE_HANDLING_STATS,
            )

        # Target Critic Update
        self.target_critic_1_update = TargetNetUpdate(
            source_net=self.critic_1,
            target_net=self.target_critic_1,
            tau=target_critic_tau,
            changes_policy=False,
            update_frequency=target_critic_update_frequency,
        )

        self.target_critic_2_update = TargetNetUpdate(
            source_net=self.critic_2,
            target_net=self.target_critic_2,
            tau=target_critic_tau,
            changes_policy=False,
            update_frequency=target_critic_update_frequency,
        )

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns all updatable components of the update rule in the order they should be updated in.

        Returns:
            Tuple[UpdatableComponent]: A tuple of all updatable components:
                1. Replay buffer update
                2. SAC critic Update
                3. SAC actor update
                4. SAC Temperature update, #(only if automatic temperature update is used)
                5. Target critic update 1
                6. Target critic update 2
        """
        if self.automatic_temperature_update:
            return (
                self.replay_buffer_update,
                self.critic_update,
                self.actor_update,
                self.temperature_update,
                self.target_critic_1_update,
                self.target_critic_2_update,
            )
        return (
            self.replay_buffer_update,
            self.critic_update,
            self.actor_update,
            self.target_critic_1_update,
            self.target_critic_2_update,
        )
