from typing import Callable, Dict, Any, Type, Optional, Tuple
import copy

import torch
from torch import nn
from torch.optim import Optimizer
from gymnasium.spaces import Box

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.ddpg.updatable_components import DDPGActorUpdate
from athlete.algorithms.td3.updatable_components import TD3CriticUpdate
from athlete.update.common import TargetNetUpdate
from athlete.saving.saveable_component import CompositeSaveableComponent
from athlete.data_collection.provider import UpdateDataProvider
from athlete.update.buffer import EpisodicCPPReplayBuffer
from athlete.update.common import TargetNetUpdate, ReplayBufferUpdate
from athlete.update.buffer_wrapper import (
    PostBufferPreprocessingWrapper,
)
from athlete.function import extract_data_from_batch, create_transition_data_info


class TD3Update(UpdateRule, CompositeSaveableComponent):
    """The Update rule for TD3.
    This class manages all updatable components and the saving and loading of stateful objects.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        update_data_provider: UpdateDataProvider,
        critic_1: nn.Module,
        critic_2: nn.Module,
        actor: nn.Module,
        discount: float,
        target_noise_std: float,
        target_noise_clip: float,
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
        target_actor_update_frequency: int,
        target_critic_tau: Optional[float],
        target_actor_tau: Optional[float],
        critic_gradient_max_norm: Optional[float],
        actor_gradient_max_norm: Optional[float],
        device: str,
        additional_replay_buffer_arguments: Dict[str, Any],
        post_replay_buffer_data_preprocessing: Optional[Dict[str, Callable]],
    ) -> None:
        """Initializes the TD3 update rule. Provide all necessary dependencies to cerate updatable components.

        Args:
            observation_space (Box): Observation space of the environment.
            action_space (Box): Action space of the environment.
            update_data_provider (UpdateDataProvider): Data provider used to communicate with the data collector.
            critic_1 (nn.Module): First critic network.
            critic_2 (nn.Module): Second critic network.
            actor (nn.Module): Actor network.
            discount (float): Discount factor for the Q-value function.
            target_noise_std (float): Standard deviation of the noise to add to the target actions.
            target_noise_clip (float): Maximum absolute value of the noise to add to the target actions.
            critic_optimizer_class (Type[Optimizer]): The optimizer class to use for the critic networks.
            actor_optimizer_class (Type[Optimizer]): The optimizer class to use for the actor network.
            critic_optimizer_arguments (Dict[str, Any]): The initialization arguments for the critic optimizer without the critic network parameters.
            actor_optimizer_arguments (Dict[str, Any]): The initialization arguments for the actor optimizer without the actor network parameters.
            critic_criteria (torch.nn.modules.loss._Loss): The loss function used for calculating the loss between the critic predictions and the target values.
            replay_buffer_capacity (int): The maximum capacity of the replay buffer.
            replay_buffer_mini_batch_size (int): The mini-batch size for sampling from the replay buffer.
            critic_update_frequency (int): The update frequency according to the number of environment steps. If -1 an update is performed at the end of each episode.
            critic_number_of_updates (int): The number of updates to perform at each update step.
            actor_update_frequency (int): The update frequency according to the number of environment steps. If -1 an update is performed at the end of each episode.
            actor_number_of_updates (int): The number of updates to perform at each update step.
            multiply_number_of_updates_by_environment_steps (bool): Whether to multiply the number of updates by the number of environment steps since the last update.
            target_critic_update_frequency (int): The update frequency for the target critic networks according to the number of environment steps.
            target_actor_update_frequency (int): The update frequency for the target actor network according to the number of environment steps.
            target_critic_tau (Optional[float]): The soft update parameter for the target critic networks, if None, a hard update is performed.
            target_actor_tau (Optional[float]): The soft update parameter for the target actor network, if None, a hard update is performed.
            critic_gradient_max_norm (Optional[float]): The maximum norm for gradient clipping for the critic networks. If None, no clipping is performed.
            actor_gradient_max_norm (Optional[float]): The maximum norm for gradient clipping for the actor network. If None, no clipping is performed.
            device (str): The device to use for the networks (e.g., "cpu" or "cuda").
            additional_replay_buffer_arguments (Dict[str, Any]): Additional arguments for initializing the replay buffer.
                This might be used for specific memory compression techniques.
            post_replay_buffer_data_preprocessing (Optional[Dict[str, Callable]]):
                A dictionary of functions to preprocess the data after sampling before passing it to the updates.
                The keys are the names of the data fields, and the values are the functions themselves.
                This might be used for example if the data in the replay buffer is an int8 to save memory but needs to be converted to float32 for the update.
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

        self.target_critic_1 = copy.deepcopy(self.critic_1).eval()
        self.target_critic_2 = copy.deepcopy(self.critic_2).eval()
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_actor.requires_grad_(False)

        self.register_saveable_component("target_critic_1", self.target_critic_1)
        self.register_saveable_component("target_critic_2", self.target_critic_2)
        self.register_saveable_component("target_actor", self.target_actor)

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

        self.critic_update = TD3CriticUpdate(
            data_sampler=critic_data_sampler_function,
            critic_1=self.critic_1,
            target_critic_1=self.target_critic_1,
            critic_2=self.critic_2,
            target_critic_2=self.target_critic_2,
            critic_optimizer=self.critic_optimizer,
            target_actor=self.target_actor,
            discount=discount,
            target_noise_std=target_noise_std,
            target_noise_clip=target_noise_clip,
            changes_policy=False,
            update_frequency=critic_update_frequency,
            number_of_updates=critic_number_of_updates,
            multiply_number_of_updates_by_environment_steps=(
                multiply_number_of_updates_by_environment_steps
            ),
            criteria=critic_criteria,
            log_tag=TD3CriticUpdate.CRITIC_LOSS_LOG_TAG,
            gradient_max_norm=critic_gradient_max_norm,
        )

        self.register_saveable_component("critic_update", self.critic_update)

        # Actor Update
        actor_data_keys = [constants.DATA_OBSERVATIONS]
        actor_data_sampler_function = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=actor_data_keys,
            device=device,
        )

        # TD3 uses the same actor update as DDPG
        self.actor_update = DDPGActorUpdate(
            data_sampler=actor_data_sampler_function,
            actor=self.actor,
            actor_optimizer=self.actor_optimizer,
            critic=self.critic_1,
            changes_policy=True,
            update_frequency=actor_update_frequency,
            number_of_updates=actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=(
                multiply_number_of_updates_by_environment_steps
            ),
            log_tag=DDPGActorUpdate.ACTOR_LOSS_LOG_TAG,
            gradient_max_norm=actor_gradient_max_norm,
        )

        self.register_saveable_component("actor_update", self.actor_update)

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

        # Target Actor Update
        self.target_actor_update = TargetNetUpdate(
            source_net=self.actor,
            target_net=self.target_actor,
            tau=target_actor_tau,
            changes_policy=False,
            update_frequency=target_actor_update_frequency,
        )

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns all updatable components of the update rule in the order they should be updated in.

        Returns:
            Tuple[UpdatableComponent]: returns a tuple of all updatable components:
                1. Replay buffer update
                2. Critic update
                3. Actor update
                4. Target critic update 1
                5. Target critic update 2
                5. Target actor update
        """
        return (
            self.replay_buffer_update,
            self.critic_update,
            self.actor_update,
            self.target_critic_1_update,
            self.target_critic_2_update,
            self.target_actor_update,
        )
