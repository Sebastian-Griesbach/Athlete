import torch
from torch import nn
from typing import Callable, Dict, Any, Tuple, Optional
import copy

from gymnasium.spaces import Box, Discrete

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.dqn.updatable_components import DQNValueUpdate
from athlete.update.common import TargetNetUpdate, ReplayBufferUpdate
from athlete.data_collection.provider import UpdateDataProvider
from athlete.update.buffer import EpisodicCPPReplayBuffer
from athlete.update.buffer_wrapper import (
    PostBufferPreprocessingWrapper,
)
from athlete.function import extract_data_from_batch, create_transition_data_info
from athlete.saving.saveable_component import CompositeSaveableComponent


class DQNUpdate(UpdateRule, CompositeSaveableComponent):
    """The Update rule for DQN.
    Manages all updatable components and the saving and loading of stateful objects.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        update_data_input: UpdateDataProvider,
        q_value_function: nn.Module,
        discount: float,
        optimizer_class: torch.optim.Optimizer,
        optimizer_arguments: Dict[str, Any],
        replay_buffer_capacity: int,
        replay_buffer_mini_batch_size: int,
        value_net_update_frequency: int,
        value_net_number_of_updates: int,
        multiply_number_of_updates_by_environment_steps: bool,
        target_net_update_frequency: int,
        target_net_tau: Optional[float],
        enable_double_q_learning: bool,
        criteria: torch.nn.modules.loss._Loss,
        gradient_max_norm: Optional[float],
        device: str,
        additional_replay_buffer_arguments: Dict[str, Any],
        post_replay_buffer_data_preprocessing: Optional[Dict[str, Callable]],
    ) -> None:
        """Initializes the DQN update rule.

        Args:
            observation_space (Box): Observation space of the environment.
            action_space (Discrete): Action space of the environment.
            update_data_input (UpdateDataProvider): Data provider used to communicate with the data collector.
            q_value_function (nn.Module): The Q-value function network.
            discount (float): The discount factor for the Q-value function.
            optimizer_class (torch.optim.Optimizer): The optimizer class to use for the Q-value function.
            optimizer_arguments (Dict[str, Any]): The initialization arguments for the optimizer with the Q-value function parameters.
            replay_buffer_capacity (int): The maximum capacity of the replay buffer.
            replay_buffer_mini_batch_size (int): The mini-batch size for sampling from the replay buffer.
            value_net_update_frequency (int): The frequency of updates for the Q-value function according to the number of environment steps.
                If -1, the update is performed at the end of each episode.
            value_net_number_of_updates (int): The number of updates to perform at each update step.
            multiply_number_of_updates_by_environment_steps (bool): Whether to multiply the number of updates by the number of environment steps since the last update.
            target_net_update_frequency (int): The frequency of updates for the target network.
            target_net_tau (Optional[float]): The soft update factor for the target network if none, uses hard updates instead.
            enable_double_q_learning (bool): Whether to use double Q-learning for the target calculation.
            criteria (torch.nn.modules.loss._Loss): The loss function used to calculate the loss based on prediction and target.
            gradient_max_norm (Optional[float]): The maximum norm for gradient clipping. If None, no clipping is performed.
            device (str): The device on which the updates are performed (e.g., "cpu" or "cuda").
            additional_replay_buffer_arguments (Dict[str, Any]): Additional arguments for initializing the replay buffer.
                This might be used for specific memory compression techniques.
            post_replay_buffer_data_preprocessing (Optional[Dict[str, Callable]]):
                A dictionary of functions to preprocess the data before passing it to the updates.
                The keys are the names of the data fields, and the values are the functions themselves.
        """

        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        # General stateful components

        self.q_value_function = q_value_function.to(device)

        self.optimizer = optimizer_class(
            params=q_value_function.parameters(), **optimizer_arguments
        )
        self.register_saveable_component("value_function", self.q_value_function)
        self.register_saveable_component("optimizer", self.optimizer)

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

        # Value function Update
        self.target_net = copy.deepcopy(q_value_function).eval()
        self.register_saveable_component("target_net", self.target_net)

        extract_keys = list(update_data_info.keys())
        data_sampler_function = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=extract_keys,
            device=device,
        )

        self.value_function_update = DQNValueUpdate(
            q_value_function=q_value_function,
            target_net=self.target_net,
            optimizer=self.optimizer,
            data_sampler=data_sampler_function,
            cross_validation=enable_double_q_learning,
            discount=discount,
            criteria=criteria,
            log_tag=DQNValueUpdate.LOG_TAG_LOSS,
            number_of_updates=value_net_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
            update_frequency=value_net_update_frequency,
        )

        # Target Net Update

        self.target_net_update = TargetNetUpdate(
            source_net=q_value_function,
            target_net=self.target_net,
            tau=target_net_tau,
            update_frequency=target_net_update_frequency,
        )

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns all updatable components of the update rule in the order they should be updated.

        Returns:
            Tuple[UpdatableComponent]: A tuple of all updatable components:
                1. Replay buffer update
                2. Value function update
                3. Target network update
        """
        return (
            self.replay_buffer_update,
            self.value_function_update,
            self.target_net_update,
        )
