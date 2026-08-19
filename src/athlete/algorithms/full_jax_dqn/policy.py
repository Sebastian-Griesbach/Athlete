from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import flax

from athlete.algorithms.full_jax_dqn.jax_interface import InfoValue

GREEDY_ACTION_LOG_TAG = "greedy_action"


def get_random_action(
    num_actions: int, select_n_actions: int, random_key: jax.Array
) -> int:
    return jax.random.randint(
        random_key, shape=(select_n_actions,), minval=0, maxval=num_actions
    )


def get_greedy_action(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    observation: jax.Array,
    post_replay_buffer_observation_preprocessing: Callable[[jax.Array], jax.Array],
) -> int:

    observation = jnp.expand_dims(observation, axis=0)
    observation = post_replay_buffer_observation_preprocessing(observation)
    q_values = q_value_function.apply(q_value_function_variables, observation)

    return jnp.argmax(q_values, axis=-1)


def get_dqn_train_action(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    epsilon_schedule: Callable[[int], float],
    warm_up_steps: int,
    step_count: int,
    observation: jax.Array,
    random_key: jax.Array,
    num_actions: int,
    post_replay_buffer_observation_preprocessing: Callable[[jax.Array], jax.Array],
    log_greedy_action: bool,
    log_prefix: str,
) -> Tuple[
    jax.Array, jax.Array, Dict[str, InfoValue]
]:  # action(s), random_key, if the selected action was greedy, action_data_valid
    log_info = {}

    random_key, sub_key = jax.random.split(random_key)

    perform_random_action = (step_count < warm_up_steps) | (
        jax.random.uniform(sub_key) < epsilon_schedule(step_count)
    )

    random_key, sub_key = jax.random.split(random_key)

    action = jax.lax.cond(
        perform_random_action,
        lambda: get_random_action(
            num_actions=num_actions,
            select_n_actions=1,
            random_key=sub_key,
        ),
        lambda: get_greedy_action(
            q_value_function=q_value_function,
            q_value_function_variables=q_value_function_variables,
            observation=observation,
            post_replay_buffer_observation_preprocessing=post_replay_buffer_observation_preprocessing,
        ),
    )

    if log_greedy_action:
        log_info[f"{log_prefix}{GREEDY_ACTION_LOG_TAG}"] = InfoValue(
            value=jnp.logical_not(perform_random_action),
            valid=jnp.array(True),
        )

    return action, random_key, log_info


def make_dqn_action_dummy_info(
    log_greedy_action: bool, log_prefix: str
) -> Dict[str, InfoValue]:
    dummy_logging_info = {}

    if log_greedy_action:
        dummy_logging_info[f"{log_prefix}{GREEDY_ACTION_LOG_TAG}"] = InfoValue(
            value=jnp.array(False), valid=jnp.array(False)
        )

    return dummy_logging_info
