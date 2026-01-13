from typing import Callable, Dict, Any, Tuple, Optional

from gymnasium.spaces import Box, Discrete
import optax

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.update.common import ReplayBufferUpdate
from athlete.data_collection.provider import UpdateDataProvider
from athlete.update.buffer import EpisodicCPPReplayBuffer
from athlete.update.buffer_wrapper import (
    PostBufferPreprocessingWrapper,
)
from athlete.function import jax_extract_data_from_batch, create_transition_data_info
from athlete.saving.saveable_component import CompositeSaveableComponent
from athlete.jax_objects import (
    MutableJaxModule,
    MutableOptaxOptimizer,
    FunctionWrapper,
    OptimizerState,
)
from athlete.algorithms.jax_dqn.updatable_components import JAXDQNValueUpdate
from athlete.update.common import JAXTargetNetUpdate


class JAXDQNUpdate(UpdateRule, CompositeSaveableComponent):
    """The Update rule for DQN.
    Manages all updatable components and the saving and loading of stateful objects.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        update_data_input: UpdateDataProvider,
        mutable_q_value_function: MutableJaxModule,
        discount: float,
        optimizer_class: optax.GradientTransformation,
        optimizer_arguments: Dict[str, Any],
        replay_buffer_capacity: int,
        replay_buffer_mini_batch_size: int,
        value_net_update_frequency: int,
        value_net_number_of_updates: int,
        multiply_number_of_updates_by_environment_steps: bool,
        target_net_update_frequency: int,
        target_net_tau: Optional[float],
        enable_double_q_learning: bool,
        criteria: FunctionWrapper,
        gradient_max_norm: Optional[float],
        additional_replay_buffer_arguments: Dict[str, Any],
        post_replay_buffer_data_preprocessing: Optional[Dict[str, Callable]],
    ) -> None:

        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        # General stateful components

        self.mutable_q_value_function = mutable_q_value_function

        if gradient_max_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(gradient_max_norm),
                optimizer_class(**optimizer_arguments),
            )
        else:
            optimizer = optimizer_class(**optimizer_arguments)

        initial_optimizer_state = optimizer.init(
            self.mutable_q_value_function.get().params
        )
        immutable_optimizer_state = OptimizerState(
            tx=optimizer,
            opt_state=initial_optimizer_state,
        )
        self.mutable_optimizer = MutableOptaxOptimizer(
            optimizer=immutable_optimizer_state
        )

        self.register_saveable_component("optimizer", self.mutable_optimizer)
        self.register_saveable_component(
            "q_value_function", self.mutable_q_value_function
        )

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
        self.mutable_target_net = MutableJaxModule(
            module=mutable_q_value_function.get()  # same immutable part as for the online q_value function
        )

        self.register_saveable_component("target_net", self.mutable_target_net)

        extract_keys = list(update_data_info.keys())
        data_sampler_function = lambda: jax_extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=extract_keys,
        )

        self.value_function_update = JAXDQNValueUpdate(
            mutable_q_value_function=self.mutable_q_value_function,
            mutable_target_net=self.mutable_target_net,
            mutable_optimizer=self.mutable_optimizer,
            data_sampler=data_sampler_function,
            cross_validation=enable_double_q_learning,
            discount=discount,
            criteria=criteria,
            update_frequency=value_net_update_frequency,
            number_of_updates=value_net_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            log_tag=JAXDQNValueUpdate.LOG_TAG_LOSS,
        )

        # Target Net Update

        self.target_net_update = JAXTargetNetUpdate(
            mutable_target_net=self.mutable_target_net,
            mutable_q_value_function=self.mutable_q_value_function,
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
