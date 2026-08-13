from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import flax


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
    post_replay_buffer_preprocessing: Callable[[jax.Array], jax.Array],
) -> int:
    observation = post_replay_buffer_preprocessing(observation)
    q_values = q_value_function.apply(q_value_function_variables, observation)
    return jnp.argmax(q_values, axis=1)


def get_dqn_train_action(
    q_value_function: flax.linen.Module,
    q_value_function_variables: flax.core.FrozenDict,
    epsilon_schedule: Callable[[int], float],
    warm_up_steps: int,
    step_count: int,
    observation: jax.Array,
    random_key: jax.Array,
    num_actions: int,
    post_replay_buffer_preprocessing: Callable[[jax.Array], jax.Array],
) -> Tuple[
    jax.Array, jax.Array, jax.Array
]:  # action(s), random_key, if the selected action was greedy

    random_key, sub_key = jax.random.split(random_key)

    perform_random_action = (step_count < warm_up_steps) | (
        jax.random.uniform(sub_key) < epsilon_schedule(step_count)
    )

    random_key, sub_key = jax.random.split(random_key)

    action = jax.lax.cond(
        perform_random_action,
        lambda: get_random_action(
            num_actions=num_actions, select_n_actions=1, random_key=sub_key
        ),
        lambda: get_greedy_action(
            q_value_function=q_value_function,
            q_value_function_variables=q_value_function_variables,
            observation=observation,
            post_replay_buffer_preprocessing=post_replay_buffer_preprocessing,
        ),
    )

    return action, random_key, ~perform_random_action
