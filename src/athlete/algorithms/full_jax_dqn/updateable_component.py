from functools import partial

import jax
import jax.numpy as jnp
import flax
import flashbax as fbx
import optax


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


@partial(jax.jit, static_argnames=["criteria", "target_calculation", "minto"])
def _jitable_update(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    target_q_value_function_variables: flax.core.FrozenDict,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    replay_buffer_func: fbx.FlatBuffer,
    replay_buffer_state: fbx.FlatBufferState,
    discount: float,
    criteria: FunctionWrapper,
    target_calculation: FunctionWrapper,
    minto: bool,
    random_key: jax.Array,
) -> Tuple[  # new variables, new optimizer state, new replay buffer state, loss, random key
    flax.core.FrozenDict, optax.OptState, fbx.FlatBufferState, jax.Array, jax.Array
]:

    raw_next_q_values = target_q_value_function_variables(next_observations)

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
    raw_q_values = q_value_function.apply_fn(q_value_function_parameters, observations)
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
