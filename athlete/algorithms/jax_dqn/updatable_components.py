from typing import Callable, Dict, Any, Tuple
from functools import partial

import flax
import optax
import numpy as np
from jax import numpy as jnp
import jax

from athlete.update.update_rule import UpdatableComponent
from athlete.global_objects import StepTracker
from athlete import constants
from athlete.jax_objects import (
    MutableJaxModule,
    MutableOptaxOptimizer,
    ModuleState,
    OptimizerState,
    FunctionWrapper,
)

from athlete.function import jax_mse_loss


class JAXDQNValueUpdate(UpdatableComponent):
    LOG_TAG_LOSS = "loss"

    def __init__(
        self,
        mutable_q_value_function: MutableJaxModule,
        mutable_target_net: MutableJaxModule,
        mutable_optimizer: MutableOptaxOptimizer,
        data_sampler: Callable[[None], Dict[str, np.ndarray]],
        cross_validation: bool = False,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        discount: float = 0.99,
        criteria: FunctionWrapper = FunctionWrapper(function=jax_mse_loss),
        log_tag: str = LOG_TAG_LOSS,
    ) -> None:

        self.mutable_q_value_function = mutable_q_value_function
        self.mutable_target_net = mutable_target_net
        self.mutable_optimizer = mutable_optimizer
        self.data_sampler = data_sampler
        self.cross_validation = cross_validation
        self.update_frequency = update_frequency
        self.number_of_updates = number_of_updates
        self.multiply_number_of_updates_by_environment_steps = (
            multiply_number_of_updates_by_environment_steps
        )
        self.discount = discount
        self.criteria = criteria
        self.log_tag = log_tag
        self.target_calculation = FunctionWrapper(
            function=(
                JAXDQNValueUpdate._calculate_cross_validation_target
                if cross_validation
                else JAXDQNValueUpdate._calculate_target
            )
        )

        self.step_tracker = StepTracker.get_instance()
        self._last_interaction_updated_on_tracker_id = (
            self.step_tracker.register_tracker(
                id="jax_frequent_update_last_interaction_updated_on_tracker_id"
            )
        )
        self._last_episode_updated_on_tracker_id = self.step_tracker.register_tracker(
            id="jax_frequent_update_last_episode_updated_on_tracker_id"
        )

    @partial(jax.jit, static_argnames=["criteria", "target_calculation"])
    def _jitable_update(
        q_value_function: ModuleState,
        target_net: ModuleState,
        optimizer: OptimizerState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        terminateds: jnp.ndarray,
        discount: float,
        criteria: FunctionWrapper,
        target_calculation: FunctionWrapper,
    ) -> Tuple[jnp.ndarray, optax.OptState, flax.core.FrozenDict]:

        # calculate target
        target = target_calculation(
            rewards=rewards,
            next_observations=next_observations,
            terminateds=terminateds,
            target_net=target_net,
            discount=discount,
            q_value_function=q_value_function,
        )

        # calculate loss

        loss, gradients = jax.value_and_grad(JAXDQNValueUpdate._calculate_loss)(
            q_value_function.params,
            q_value_function=q_value_function,
            observations=observations,
            actions=actions,
            target=target,
            criteria=criteria,
        )

        # apply gradients
        updates, new_optimizer_state = optimizer.tx.update(
            gradients, optimizer.opt_state, q_value_function.params
        )

        new_q_value_function_parameters = optax.apply_updates(
            q_value_function.params, updates
        )

        # create new q_value_function and optimizer
        new_q_value_function = q_value_function.replace(
            params=new_q_value_function_parameters
        )
        new_optimizer = optimizer.replace(opt_state=new_optimizer_state)

        return loss, new_q_value_function, new_optimizer

    @jax.jit
    def _calculate_loss(
        q_value_function_parameters: flax.core.FrozenDict,
        q_value_function: ModuleState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        target: jnp.ndarray,
        criteria: FunctionWrapper,
    ) -> jnp.ndarray:
        # calculate predictions
        raw_q_values = q_value_function.apply_fn(
            q_value_function_parameters, observations
        )
        q_values = jnp.take_along_axis(raw_q_values, actions, axis=1)

        # calculate loss
        loss = criteria(q_values, target)
        return loss

    @jax.jit
    def _calculate_target(
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        terminateds: jnp.ndarray,
        target_net: ModuleState,
        discount: float,
        q_value_function: ModuleState,  # Not needed only here to have same signature as cross validation
    ) -> jnp.ndarray:
        raw_next_q_values = target_net(next_observations)
        next_q_values = jnp.max(raw_next_q_values, axis=1, keepdims=True)
        not_terminateds = jnp.logical_not(terminateds)
        target = rewards + not_terminateds * discount * next_q_values
        return target

    @jax.jit
    def _calculate_cross_validation_target(
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        terminateds: jnp.ndarray,
        target_net: ModuleState,
        discount: float,
        q_value_function: ModuleState,
    ) -> jnp.ndarray:
        next_actions = jnp.argmax(
            q_value_function(next_observations), axis=1, keepdims=True
        )
        next_q_values = jnp.take_along_axis(
            target_net(next_observations), next_actions, axis=1
        )
        not_terminateds = jnp.logical_not(terminateds)
        target = rewards + not_terminateds * discount * next_q_values
        return target

    def update(self) -> Dict[str, Any]:
        losses = []
        number_of_updates = (
            self.number_of_updates
            if not self.multiply_number_of_updates_by_environment_steps
            else self.number_of_updates
            * (
                self.step_tracker.interactions_after_warmup
                - self.step_tracker.get_tracker_value(
                    id=self._last_interaction_updated_on_tracker_id
                )
            )
        )

        # Tracking the last update step and episode for update condition and number of updates
        self.step_tracker.set_tracker_value(
            id=self._last_interaction_updated_on_tracker_id,
            value=self.step_tracker.interactions_after_warmup,
        )
        self.step_tracker.set_tracker_value(
            id=self._last_episode_updated_on_tracker_id,
            value=self.step_tracker.get_tracker_value(
                id=constants.TRACKER_ENVIRONMENT_EPISODES
            ),
        )

        for _ in range(number_of_updates):

            mini_batch = self.data_sampler()
            observations = mini_batch[constants.DATA_OBSERVATIONS]
            actions = mini_batch[constants.DATA_ACTIONS]
            rewards = mini_batch[constants.DATA_REWARDS]
            next_observations = mini_batch[constants.DATA_NEXT_OBSERVATIONS]
            terminateds = mini_batch[constants.DATA_TERMINATEDS]

            loss, new_q_value_function, new_optimizer = (
                JAXDQNValueUpdate._jitable_update(
                    q_value_function=self.mutable_q_value_function.get(),
                    target_net=self.mutable_target_net.get(),
                    optimizer=self.mutable_optimizer.get(),
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    terminateds=terminateds,
                    discount=self.discount,
                    criteria=self.criteria,
                    target_calculation=self.target_calculation,
                )
            )

            # update parameters and optimizer state
            self.mutable_q_value_function.set(new_q_value_function)
            self.mutable_optimizer.set(new_optimizer)

            losses.append(loss.item())

        log_data = {self.log_tag: np.mean(losses).item()}
        return log_data

    @property
    def update_condition(self) -> bool:
        if self.update_frequency > 0:
            # Update if training frequency is met
            return (
                self.step_tracker.is_warmup_done
                and (
                    self.step_tracker.interactions_after_warmup % self.update_frequency
                    == 0
                )  # But only if the effective number of updates is > 0
                and (
                    not self.multiply_number_of_updates_by_environment_steps
                    or (
                        self.step_tracker.interactions_after_warmup
                        > self.step_tracker.get_tracker_value(
                            self._last_interaction_updated_on_tracker_id
                        )
                    )
                )
            )
        # If update_frequency is <= 0, we update when an episode ends
        return (
            self.step_tracker.is_warmup_done
            and (
                self.step_tracker.get_tracker_value(
                    id=constants.TRACKER_ENVIRONMENT_EPISODES
                )
                > self.step_tracker.get_tracker_value(
                    id=self._last_episode_updated_on_tracker_id
                )
            )
            and self.step_tracker.interactions_after_warmup
            > self.step_tracker.get_tracker_value(
                id=self._last_interaction_updated_on_tracker_id
            )
        )
