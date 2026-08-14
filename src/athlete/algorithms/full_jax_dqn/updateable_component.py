from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import flax
import optax

from athlete import constants
from athlete.jax_objects import FunctionWrapper
from athlete.algorithms.full_jax_dqn.buffer import (
    EpisodeAwareFlatBuffer,
    EpisodeAwareFlatBufferState,
)


def flat_replay_buffer_transition_update(
    replay_buffer_func: EpisodeAwareFlatBuffer,
    replay_buffer_state: EpisodeAwareFlatBufferState,
    action: jax.Array,
    observation: jax.Array,
    reward: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    new_episode_started: jax.Array,
) -> EpisodeAwareFlatBufferState:
    experience = {
        constants.DATA_OBSERVATIONS: observation,
        constants.DATA_ACTIONS: action,
        constants.DATA_REWARDS: reward,
        constants.DATA_TERMINATEDS: terminated,
    }

    replay_buffer_state = replay_buffer_func.add(
        state=replay_buffer_state,
        entry=experience,
        new_episode_started=new_episode_started,
    )
    return replay_buffer_state


def dqn_value_update(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    target_q_value_function_variables: flax.core.FrozenDict,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    discount: float,
    criteria: FunctionWrapper,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    terminateds: jnp.ndarray,
    double_q: bool,
    minto: bool,
) -> Tuple[  # new variables, new optimizer state, loss and mean q values for logging
    flax.core.FrozenDict,
    optax.OptState,
    jax.Array,
    jax.Array,
]:

    raw_target_next_q_values = q_value_function.apply(
        target_q_value_function_variables, next_observations
    )

    if minto | double_q:
        raw_online_next_q_values = q_value_function.apply(
            q_value_function_variables, next_observations
        )

    if minto:
        raw_target_next_q_values = jnp.minimum(
            raw_online_next_q_values, raw_target_next_q_values
        )

    if double_q:
        online_next_actions = jnp.argmax(
            raw_online_next_q_values, axis=-1, keepdims=True
        )
        target_next_q_values = jnp.take_along_axis(
            raw_target_next_q_values, online_next_actions, axis=-1
        )
    else:
        target_next_q_values = jnp.max(raw_target_next_q_values, axis=-1, keepdims=True)

    # calculate target
    not_terminateds = jnp.logical_not(terminateds)
    target = rewards + not_terminateds * discount * target_next_q_values

    # calculate loss
    (loss, raw_q_values), gradients = jax.value_and_grad(
        calculate_dqn_loss, has_aux=True
    )(
        q_value_function_variables,
        q_value_function=q_value_function,
        observations=observations,
        actions=actions,
        target=target,
        criteria=criteria,
    )

    # apply gradients
    updates, optimizer_state = optimizer.update(
        gradients, optimizer_state, q_value_function_variables
    )

    q_value_function_variables = optax.apply_updates(
        q_value_function_variables, updates
    )

    # for logging
    mean_q_values = raw_q_values.mean()

    return (
        q_value_function_variables,
        optimizer_state,
        loss,
        mean_q_values,
    )


def calculate_dqn_loss(
    q_value_function_variables: flax.core.FrozenDict,
    q_value_function: flax.linen.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    target: jnp.ndarray,
    criteria: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    # calculate predictions
    raw_q_values = q_value_function.apply(q_value_function_variables, observations)
    q_values = jnp.take_along_axis(raw_q_values, actions, axis=-1)

    # calculate loss
    loss = criteria(q_values, target)
    return loss, raw_q_values


def get_transitions_from_flat_buffer(
    replay_buffer_func: EpisodeAwareFlatBuffer,
    replay_buffer_state: EpisodeAwareFlatBufferState,
    random_key: jax.Array,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    batch = replay_buffer_func.sample(replay_buffer_state, random_key)
    observations = batch.experience.first[constants.DATA_OBSERVATIONS]
    actions = batch.experience.second[constants.DATA_ACTIONS]
    rewards = batch.experience.second[constants.DATA_REWARDS]
    next_observations = batch.experience.second[constants.DATA_OBSERVATIONS]
    terminateds = batch.experience.second[constants.DATA_TERMINATEDS]
    return observations, actions, rewards, next_observations, terminateds


def perform_n_q_value_function_updates(
    replay_buffer_func: EpisodeAwareFlatBuffer,
    replay_buffer_state: EpisodeAwareFlatBufferState,
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    target_q_value_function_variables: flax.core.FrozenDict,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    discount: float,
    criteria: FunctionWrapper,
    double_q: bool,
    minto: bool,
    random_key: jax.Array,
    n_updates: int,
) -> Tuple[
    Tuple[flax.core.FrozenDict, optax.OptState, jax.Array], Tuple[jax.Array, jax.Array]
]:

    def sample_flat_buffer_and_update_q_value_function(carry, _) -> Tuple[
        Tuple[flax.core.FrozenDict, optax.OptState, jax.Array],
        Tuple[jax.Array, jax.Array],
    ]:
        q_value_function_variables, optimizer_state, random_key = carry
        random_key, subkey = jax.random.split(random_key)
        observations, actions, rewards, next_observations, terminateds = (
            get_transitions_from_flat_buffer(
                replay_buffer_func=replay_buffer_func,
                replay_buffer_state=replay_buffer_state,
                random_key=subkey,
            )
        )

        (
            q_value_function_variables,
            optimizer_state,
            loss,
            mean_q_values,
        ) = dqn_value_update(
            q_value_function=q_value_function,
            q_value_function_variables=q_value_function_variables,
            target_q_value_function_variables=target_q_value_function_variables,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            discount=discount,
            criteria=criteria,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminateds=terminateds,
            double_q=double_q,
            minto=minto,
        )

        new_carry = (q_value_function_variables, optimizer_state, random_key)
        logging_info = (loss, mean_q_values)
        return new_carry, logging_info

    (q_value_function_variables, optimizer_state, random_key), (
        losses,
        mean_q_values,
    ) = jax.lax.scan(
        sample_flat_buffer_and_update_q_value_function,
        init=(q_value_function_variables, optimizer_state, random_key),
        xs=None,
        length=n_updates,
    )

    return (
        q_value_function_variables,
        optimizer_state,
        random_key,
        (  # For logging
            jnp.array(True),
            losses.mean(),
            mean_q_values.mean(),
        ),
    )


def target_network_update(
    target_network_variables: flax.core.FrozenDict,
    q_value_function_variables: flax.core.FrozenDict,
    tau: float = 1.0,
) -> flax.core.FrozenDict:
    target_network_variables = optax.incremental_update(
        new_tensors=q_value_function_variables,
        old_tensors=target_network_variables,
        step_size=tau,
    )
    return target_network_variables
