from typing import Callable, Dict, Any, Tuple
from functools import partial

import flax
import optax
import numpy as np
from jax import numpy as jnp
import jax

from athlete.update.common import FrequencyUpdate
from athlete import constants
from athlete.jax_objects import (
    MutableJaxModule,
    MutableOptaxOptimizer,
    ModuleState,
    OptimizerState,
    FunctionWrapper,
)

from athlete.function import jax_mse_loss


class JAXDQNValueUpdate(FrequencyUpdate):
    LOG_TAG_LOSS = "loss"

    def __init__(
        self,
        mutable_q_value_function: MutableJaxModule,
        mutable_target_net: MutableJaxModule,
        mutable_optimizer: MutableOptaxOptimizer,
        data_sampler: Callable[[None], Dict[str, np.ndarray]],
        cross_validation: bool = False,
        minto: bool = False,  # TODO surface this option to the init, or remove it
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        discount: float = 0.99,
        criteria: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = jax_mse_loss,
        log_tag: str = LOG_TAG_LOSS,
    ) -> None:

        super().__init__(
            log_tag=log_tag,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
        )

        self.mutable_q_value_function = mutable_q_value_function
        self.mutable_target_net = mutable_target_net
        self.mutable_optimizer = mutable_optimizer
        self.data_sampler = data_sampler
        self.cross_validation = cross_validation
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
        self.minto = minto

    @partial(jax.jit, static_argnames=["criteria", "target_calculation", "minto"])
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
        minto: bool,
    ) -> Tuple[jnp.ndarray, optax.OptState, flax.core.FrozenDict]:

        raw_next_q_values = target_net(next_observations)

        if minto:
            online_raw_next_q_values = q_value_function(next_observations)
            raw_next_q_values = jnp.minimum(raw_next_q_values, online_raw_next_q_values)

        # calculate target
        target = target_calculation(
            rewards=rewards,
            next_observations=next_observations,
            terminateds=terminateds,
            raw_next_q_values=raw_next_q_values,
            discount=discount,
            q_value_function=q_value_function,
        )

        # calculate loss
        (loss, raw_q_values), gradients = jax.value_and_grad(
            JAXDQNValueUpdate._calculate_loss, has_aux=True
        )(
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
            variables=new_q_value_function_parameters
        )
        new_optimizer = optimizer.replace(opt_state=new_optimizer_state)

        # for logging
        mean_q_values = raw_q_values.mean()

        return loss, new_q_value_function, new_optimizer, mean_q_values

    @partial(jax.jit, static_argnames=["criteria"])
    def _calculate_loss(
        q_value_function_parameters: flax.core.FrozenDict,
        q_value_function: ModuleState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        target: jnp.ndarray,
        criteria: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        # calculate predictions
        raw_q_values = q_value_function.apply_fn(
            q_value_function_parameters, observations
        )
        q_values = jnp.take_along_axis(raw_q_values, actions, axis=1)

        # calculate loss
        loss = criteria(q_values, target)
        return loss, raw_q_values

    @jax.jit
    def _calculate_target(
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        terminateds: jnp.ndarray,
        raw_next_q_values: jnp.ndarray,
        discount: float,
        q_value_function: ModuleState,  # Not needed only here to have same signature as cross validation
    ) -> jnp.ndarray:
        next_q_values = jnp.max(raw_next_q_values, axis=1, keepdims=True)
        not_terminateds = jnp.logical_not(terminateds)
        target = rewards + not_terminateds * discount * next_q_values
        return target

    @jax.jit
    def _calculate_cross_validation_target(
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        terminateds: jnp.ndarray,
        raw_next_q_values: jnp.ndarray,
        discount: float,
        q_value_function: ModuleState,
    ) -> jnp.ndarray:
        next_actions = jnp.argmax(
            q_value_function(next_observations), axis=1, keepdims=True
        )
        next_q_values = jnp.take_along_axis(raw_next_q_values, next_actions, axis=1)
        not_terminateds = jnp.logical_not(terminateds)
        target = rewards + not_terminateds * discount * next_q_values
        return target

    def _update(self) -> Dict[str, Any]:
        mini_batch = self.data_sampler()
        observations = mini_batch[constants.DATA_OBSERVATIONS]
        actions = mini_batch[constants.DATA_ACTIONS]
        rewards = mini_batch[constants.DATA_REWARDS]
        next_observations = mini_batch[constants.DATA_NEXT_OBSERVATIONS]
        terminateds = mini_batch[constants.DATA_TERMINATEDS]

        loss, new_q_value_function, new_optimizer, mean_q_values = (
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
                minto=self.minto,
            )
        )

        # update parameters and optimizer state
        self.mutable_q_value_function.set(new_q_value_function)
        self.mutable_optimizer.set(new_optimizer)

        self.last_mean_q_values = mean_q_values

        return loss.item()

    def post_update_routine(self):
        # TODO this should also use a prefix
        logging_info = {
            "q_values_mean": self.last_mean_q_values.item(),
        }
        return logging_info
