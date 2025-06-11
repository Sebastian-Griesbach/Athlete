from typing import Callable, Dict, Any, Type, Optional, Tuple
import copy

import torch
from torch import nn
from torch.optim import Optimizer
from gymnasium.spaces import Space

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.ddpg.updatable_components import (
    DDPGCriticUpdate,
    DDPGActorUpdate,
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


class DDPGUpdate(UpdateRule, CompositeSaveableComponent):
    """Update rule for the DDPG algorithm.
    This class manages all updatable components of the DDPG algorithm.
    It also handles the saving off all stateful objects of the DDPG algorithm.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        update_data_input: UpdateDataProvider,
        critic: nn.Module,
        actor: nn.Module,
        discount: float,
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
        """
        Initializes the DDPG update rule.

        Args:
            observation_space (Space): The observation space of the environment.
            action_space (Space): The action space of the environment.
            update_data_input (UpdateDataProvider): The data provider for the update rule. This is used
                to communicate with the data collector.
            critic (nn.Module): The critic network.
            actor (nn.Module): The actor network.
            discount (float): The discount factor for the critic.
            critic_optimizer_class (Type[Optimizer]): The class of the critic optimizer, a subclass of torch.optim.Optimizer.
            actor_optimizer_class (Type[Optimizer]): The class of the actor optimizer, a subclass of torch.optim.Optimizer.
            critic_optimizer_arguments (Dict[str, Any]): The arguments for initializing the critic optimizer, not including the parameters critic parameters.
            actor_optimizer_arguments (Dict[str, Any]): The arguments for initializing the actor optimizer, not including the parameters actor parameters.
            critic_criteria (torch.nn.modules.loss._Loss): The loss function used for calculating the loss between the predicted and target values for the critic.
            replay_buffer_capacity (int): The maximum capacity of the replay buffer.
            replay_buffer_mini_batch_size (int): The mini-batch size for sampling from the replay buffer.
            critic_update_frequency (int): The frequency of the critic update according to the number of environment steps. If -1 the critic is updated at the end of each episode.
            critic_number_of_updates (int): The number of updates for the critic on each update step.
            actor_update_frequency (int): The frequency of the actor update according to the number of environment steps. If -1 the actor is updated at the end of each episode.
            actor_number_of_updates (int): The number of updates for the actor on each update step.
            multiply_number_of_updates_by_environment_steps (bool): Whether the number of updates for actor and critic is multiplied by the environment steps since the last update.
            target_critic_update_frequency (int): The frequency of the target critic update according to the number of environment steps.
            target_actor_update_frequency (int): The frequency of the target actor update according to the number of environment steps.
            target_critic_tau (Optional[float]) : The tau parameter for the target critic update. If none, a hard update is performed.
            target_actor_tau (Optional[float]): The tau parameter for the target actor update. If none, a hard update is performed.
            critic_gradient_max_norm (Optional[float]): The maximum norm for the critic gradient clipping. If None, no gradient clipping is performed.
            actor_gradient_max_norm (Optional[float]): The maximum norm for the actor gradient clipping. If None, no gradient clipping is performed.
            device (str): The device to use for the update rule. This is used to move the networks and data to the correct device.
            additional_replay_buffer_arguments (Dict[str, Any]): Additional arguments for the replay buffer initialization. Might be used for specific memory compression.
            post_replay_buffer_data_preprocessing (Optional[Dict[str, Callable]]): A dictionary of functions to preprocess the data sampled from the replay buffer. The keys are the field names of the data and the values are the functions to apply.
        """

        super().__init__()

        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        # Updatable components
        # General stateful components

        self.critic = critic.to(device)
        self.actor = actor.to(device)

        self.critic_optimizer = critic_optimizer_class(
            params=self.critic.parameters(), **critic_optimizer_arguments
        )
        self.actor_optimizer = actor_optimizer_class(
            params=self.actor.parameters(), **actor_optimizer_arguments
        )

        self.register_saveable_component("critic", self.critic)
        self.register_saveable_component("actor", self.actor)
        self.register_saveable_component("critic_optimizer", self.critic_optimizer)
        self.register_saveable_component("actor_optimizer", self.actor_optimizer)

        self.target_critic = copy.deepcopy(self.critic).eval()
        self.target_actor = copy.deepcopy(self.actor).eval()

        self.register_saveable_component("target_critic", self.target_critic)
        self.register_saveable_component("target_actor", self.target_actor)

        # Replay Buffer Update

        update_data_info = create_transition_data_info(
            observation_space=observation_space,
            action_space=action_space,
        )

        additional_arguments = {
            "next_of": constants.DATA_OBSERVATIONS,
        }
        additional_arguments.update(additional_replay_buffer_arguments)

        self.replay_buffer = EpisodicCPPReplayBuffer(
            capacity=replay_buffer_capacity,
            replay_buffer_info=update_data_info,
            additional_arguments=additional_arguments,
        )

        self.register_saveable_component("replay_buffer", self.replay_buffer)

        self.replay_buffer_update = ReplayBufferUpdate(
            update_data_provider=update_data_input,
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

        self.critic_update = DDPGCriticUpdate(
            data_sampler=critic_data_sampler_function,
            critic=self.critic,
            target_critic=self.target_critic,
            critic_optimizer=self.critic_optimizer,
            target_actor=self.target_actor,
            discount=discount,
            update_frequency=critic_update_frequency,
            number_of_updates=critic_number_of_updates,
            multiply_number_of_updates_by_environment_steps=(
                multiply_number_of_updates_by_environment_steps
            ),
            criteria=critic_criteria,
            log_tag=DDPGCriticUpdate.CRITIC_LOSS_LOG_TAG,
            gradient_max_norm=critic_gradient_max_norm,
        )

        # Actor Update
        actor_data_keys = [constants.DATA_OBSERVATIONS]
        actor_data_sampler_function = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=actor_data_keys,
            device=device,
        )

        self.actor_update = DDPGActorUpdate(
            data_sampler=actor_data_sampler_function,
            actor=self.actor,
            actor_optimizer=self.actor_optimizer,
            critic=self.critic,
            update_frequency=actor_update_frequency,
            number_of_updates=actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=(
                multiply_number_of_updates_by_environment_steps
            ),
            log_tag=DDPGActorUpdate.ACTOR_LOSS_LOG_TAG,
            gradient_max_norm=actor_gradient_max_norm,
        )

        # Target Critic Update
        self.target_critic_update = TargetNetUpdate(
            source_net=self.critic,
            target_net=self.target_critic,
            tau=target_critic_tau,
            update_frequency=target_critic_update_frequency,
        )

        # Target Actor Update
        self.target_actor_update = TargetNetUpdate(
            source_net=self.actor,
            target_net=self.target_actor,
            tau=target_actor_tau,
            update_frequency=target_actor_update_frequency,
        )

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns all updatable components of the DDPG algorithm in the order they are
        updated.

        Returns:
            Tuple[UpdatableComponent]: A tuple of updatable components.
                1. Replay Buffer Update
                2. Critic Update
                3. Actor Update
                4. Target Critic Update
                5. Target Actor Update
        """
        return (
            self.replay_buffer_update,
            self.critic_update,
            self.actor_update,
            self.target_critic_update,
            self.target_actor_update,
        )
