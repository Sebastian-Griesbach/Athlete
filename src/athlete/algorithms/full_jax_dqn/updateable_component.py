from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import flax
import flashbax as fbx
import optax

from athlete import constants
from athlete.jax_objects import FunctionWrapper


@jax.jit
def replay_buffer_update(
    replay_buffer_func: fbx.FlatBuffer,
    replay_buffer_state: fbx.FlatBufferState,
    transition_data: dict,
) -> fbx.FlatBufferState:
    new_replay_buffer_state = replay_buffer_func.add(
        replay_buffer_state, transition_data
    )
    return new_replay_buffer_state


@partial(
    jax.jit,
    static_argnames=[
        "criteria",
        "target_calculation",
        "minto",
        "double_q",
        "logging_prefix",
    ],
)
def dqn_value_update(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    target_q_value_function_variables: flax.core.FrozenDict,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    replay_buffer_func: fbx.FlatBuffer,
    replay_buffer_state: fbx.FlatBufferState,
    discount: float,
    criteria: FunctionWrapper,
    double_q: bool,
    minto: bool,
    random_key: jax.Array,
    logging_prefix: str = "",
) -> Tuple[  # new variables, new optimizer state, new replay buffer state, random_key, dictionary for logging
    flax.core.FrozenDict,
    optax.OptState,
    fbx.FlatBufferState,
    jax.Array,
    Dict[str, jax.Array],
]:

    random_key, subkey = jax.random.split(random_key)
    # Construct transitions from flat replay buffer
    batch = replay_buffer_func.sample(replay_buffer_state, subkey)
    observations = batch.first[constants.DATA_OBSERVATIONS]
    actions = batch.first[constants.DATA_ACTIONS]
    rewards = batch.first[constants.DATA_REWARDS]
    next_observations = batch.second[constants.DATA_OBSERVATIONS]
    terminateds = batch.first[constants.DATA_TERMINATEDS]

    raw_target_next_q_values = q_value_function.apply(
        target_q_value_function_variables, next_observations
    )

    if minto:
        raw_online_next_q_values = q_value_function.apply(
            q_value_function_variables, next_observations
        )
        raw_target_next_q_values = jnp.minimum(
            raw_online_next_q_values, raw_target_next_q_values
        )

    if double_q:
        raw_online_next_q_values = q_value_function.apply(
            q_value_function_variables, next_observations
        )
        online_next_actions = jnp.argmax(raw_online_next_q_values, axis=-1)
        target_next_q_values = jnp.take_along_axis(
            raw_target_next_q_values, online_next_actions[..., None], axis=-1
        )
    else:
        target_next_q_values = jnp.max(raw_target_next_q_values, axis=-1)

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
    updates, optimizer_state = optimizer.tx.update(
        gradients, optimizer_state, q_value_function_variables
    )

    q_value_function_variables = optax.apply_updates(
        q_value_function_variables, updates
    )

    # for logging
    mean_q_values = raw_q_values.mean()
    logging_dict = {
        f"{logging_prefix}{constants.LOGGING_DATA_VALID}": jnp.array(True),
        f"{logging_prefix}loss": loss,
        f"{logging_prefix}mean_q_values": mean_q_values,
    }

    return (
        q_value_function_variables,
        optimizer_state,
        replay_buffer_state,
        random_key,
        logging_dict,
    )


@partial(jax.jit, static_argnames=["criteria"])
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
    q_values = jnp.take_along_axis(raw_q_values, actions, axis=1)

    # calculate loss
    loss = criteria(q_values, target)
    return loss, raw_q_values
