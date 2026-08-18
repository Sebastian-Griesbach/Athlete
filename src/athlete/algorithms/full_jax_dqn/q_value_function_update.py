from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import flax
import optax

from athlete.algorithms.full_jax_dqn.replay_buffer_update import (
    EpisodeAwareFlatBuffer,
    EpisodeAwareFlatBufferState,
)
from athlete.algorithms.full_jax_dqn.interface import LogValue
from athlete import constants

LOSS_LOG_TAG = "loss"
MEAN_Q_VALUES_LOG_TAG = "mean_q_values"


def perform_n_q_value_function_updates(
    replay_buffer_func: EpisodeAwareFlatBuffer,
    replay_buffer_state: EpisodeAwareFlatBufferState,
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    target_q_value_function_variables: flax.core.FrozenDict,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    discount: float,
    loss_function: Callable[[jax.Array, jax.Array], jax.Array],
    double_q: bool,
    minto: bool,
    log_loss: bool,
    log_mean_q_values: bool,
    log_prefix: str,
    random_key: jax.Array,
    n_updates: int,
    post_replay_buffer_observation_preprocessing: Callable[[jax.Array], jax.Array],
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
                post_replay_buffer_observation_preprocessing=post_replay_buffer_observation_preprocessing,
            )
        )

        (
            q_value_function_variables,
            optimizer_state,
            log_info,
        ) = dqn_value_update(
            q_value_function=q_value_function,
            q_value_function_variables=q_value_function_variables,
            target_q_value_function_variables=target_q_value_function_variables,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            discount=discount,
            loss_function=loss_function,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminateds=terminateds,
            double_q=double_q,
            minto=minto,
            log_loss=log_loss,
            log_mean_q_values=log_mean_q_values,
            log_prefix=log_prefix,
        )

        new_carry = (q_value_function_variables, optimizer_state, random_key)
        return new_carry, log_info

    (q_value_function_variables, optimizer_state, random_key), (log_infos) = (
        jax.lax.scan(
            sample_flat_buffer_and_update_q_value_function,
            init=(q_value_function_variables, optimizer_state, random_key),
            xs=None,
            length=n_updates,
        )
    )

    aggregated_log_info = jax.tree.map(
        lambda values: LogValue(value=values.mean(axis=0), valid=jnp.array(True)),
        log_infos,
    )

    return (
        q_value_function_variables,
        optimizer_state,
        random_key,
        aggregated_log_info,
    )


def get_transitions_from_flat_buffer(
    replay_buffer_func: EpisodeAwareFlatBuffer,
    replay_buffer_state: EpisodeAwareFlatBufferState,
    random_key: jax.Array,
    post_replay_buffer_observation_preprocessing: Callable[[jax.Array], jax.Array],
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    batch = replay_buffer_func.sample(replay_buffer_state, random_key)
    observations = post_replay_buffer_observation_preprocessing(
        batch.experience.first[constants.DATA_OBSERVATIONS]
    )
    actions = batch.experience.second[constants.DATA_ACTIONS]
    rewards = batch.experience.second[constants.DATA_REWARDS]
    next_observations = post_replay_buffer_observation_preprocessing(
        batch.experience.second[constants.DATA_OBSERVATIONS]
    )
    terminateds = batch.experience.second[constants.DATA_TERMINATEDS]
    return observations, actions, rewards, next_observations, terminateds


def dqn_value_update(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    target_q_value_function_variables: flax.core.FrozenDict,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    discount: float,
    loss_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    terminateds: jnp.ndarray,
    double_q: bool,
    minto: bool,
    log_loss: bool,
    log_mean_q_values: bool,
    log_prefix: str,
) -> Tuple[  # new variables, new optimizer state, loss and mean q values for logging
    flax.core.FrozenDict,
    optax.OptState,
    Dict[str, LogValue],
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
        loss_function=loss_function,
    )

    # apply gradients
    updates, optimizer_state = optimizer.update(
        gradients, optimizer_state, q_value_function_variables
    )

    q_value_function_variables = optax.apply_updates(
        q_value_function_variables, updates
    )

    # for logging
    logging_info = {}
    if log_loss:
        logging_info[f"{log_prefix}{LOSS_LOG_TAG}"] = loss
    if log_mean_q_values:
        logging_info[f"{log_prefix}{MEAN_Q_VALUES_LOG_TAG}"] = raw_q_values.mean()

    return (
        q_value_function_variables,
        optimizer_state,
        logging_info,
    )


def calculate_dqn_loss(
    q_value_function_variables: flax.core.FrozenDict,
    q_value_function: flax.linen.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    target: jnp.ndarray,
    loss_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    # calculate predictions
    raw_q_values = q_value_function.apply(q_value_function_variables, observations)
    q_values = jnp.take_along_axis(raw_q_values, actions, axis=-1)

    # calculate loss
    loss = loss_function(q_values, target)
    return loss, raw_q_values


def make_dqn_value_update_dummy_log_info(
    log_loss: bool, log_mean_q_values: bool, log_prefix: str
) -> Dict[str, LogValue]:
    dummy_logging_info = {}

    if log_loss:
        dummy_logging_info[f"{log_prefix}{LOSS_LOG_TAG}"] = LogValue(
            value=jnp.nan, valid=jnp.array(False)
        )
    if log_mean_q_values:
        dummy_logging_info[f"{log_prefix}{MEAN_Q_VALUES_LOG_TAG}"] = LogValue(
            value=jnp.nan, valid=jnp.array(False)
        )

    return dummy_logging_info
