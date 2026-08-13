from functools import partial
from typing import Callable, Dict

import flax
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

from athlete.algorithms.full_jax_dqn.updateable_component import (
    flat_replay_buffer_transition_update,
    perform_n_q_value_function_updates,
    FlatDataCollectorState,
    target_network_update,
)
from athlete.algorithms.full_jax_dqn.policy import (
    get_dqn_train_action,
    get_greedy_action,
)

from athlete import constants


class DQNAgentState(flax.struct.PyTreeNode):
    replay_buffer_state: fbx.FlatBufferState = flax.struct.field(pytree_node=True)
    data_collector_state: FlatDataCollectorState = flax.struct.field(pytree_node=True)
    last_action: jax.Array = flax.struct.field(pytree_node=True)
    random_key: jax.Array = flax.struct.field(pytree_node=True)
    q_value_function_variables: Dict[str, jax.Array] = flax.struct.field(
        pytree_node=True
    )
    target_q_value_function_variables: Dict[str, jax.Array] = flax.struct.field(
        pytree_node=True
    )
    optimizer_state: optax.OptState = flax.struct.field(pytree_node=True)
    step_count: jax.Array = flax.struct.field(pytree_node=True)


class DQNAgentSpecification(flax.struct.PyTreeNode):
    replay_buffer_func: fbx.FlatBuffer = flax.struct.field(pytree_node=False)
    q_value_function: flax.linen.Module = flax.struct.field(pytree_node=False)
    discount: float = flax.struct.field(pytree_node=False)
    criteria: Callable = flax.struct.field(pytree_node=False)
    minto: bool = flax.struct.field(pytree_node=False)
    double_q: bool = flax.struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    value_function_update_frequency: int = flax.struct.field(pytree_node=False)
    value_function_number_of_updates: int = flax.struct.field(pytree_node=False)
    warm_up_steps: int = flax.struct.field(pytree_node=False)
    epsilon_schedule: Callable = flax.struct.field(pytree_node=False)
    target_network_update_frequency: int = flax.struct.field(pytree_node=False)
    target_network_update_tau: float = flax.struct.field(pytree_node=False)
    num_actions: int = flax.struct.field(pytree_node=False)
    post_replay_buffer_preprocessing: Callable = flax.struct.field(pytree_node=False)


@partial(
    jax.jit,
    static_argnames=("agent_specification",),
    donate_argnames=("agent_state",),
)
def dqn_train_step(
    agent_specification: DQNAgentSpecification,
    agent_state: DQNAgentState,
    observation: jax.Array,
    reward: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,  # Not needed but should be part of the interface
) -> DQNAgentState:

    # all state objects that might get updated
    replay_buffer_state = agent_state.replay_buffer_state
    data_collector_state = agent_state.data_collector_state
    q_value_function_variables = agent_state.q_value_function_variables
    optimizer_state = agent_state.optimizer_state
    random_key = agent_state.random_key
    target_q_value_function_variables = agent_state.target_q_value_function_variables
    last_action = agent_state.last_action
    step_count = agent_state.step_count

    # Replay buffer update
    replay_buffer_state, data_collector_state = flat_replay_buffer_transition_update(
        replay_buffer_func=agent_specification.replay_buffer_func,
        replay_buffer_state=replay_buffer_state,
        data_collector_state=data_collector_state,
        action=last_action,
        observation=observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )

    # Value function update
    (
        q_value_function_variables,
        optimizer_state,
        random_key,
        (
            is_logging_data_valid,
            losses,
            mean_q_values,
        ),  # TODO add marker for if logging data is valid or not
    ) = jax.lax.cond(
        (step_count >= agent_specification.warm_up_steps)
        & agent_specification.replay_buffer_func.can_sample(replay_buffer_state)
        & (step_count % agent_specification.value_function_update_frequency == 0),
        lambda: perform_n_q_value_function_updates(
            replay_buffer_func=agent_specification.replay_buffer_func,
            replay_buffer_state=replay_buffer_state,
            q_value_function=agent_specification.q_value_function,
            q_value_function_variables=q_value_function_variables,
            target_q_value_function_variables=target_q_value_function_variables,
            optimizer=agent_specification.optimizer,
            optimizer_state=optimizer_state,
            discount=agent_specification.discount,
            criteria=agent_specification.criteria,
            double_q=agent_specification.double_q,
            minto=agent_specification.minto,
            random_key=random_key,
            n_updates=agent_specification.value_function_number_of_updates,
        ),
        lambda: (
            q_value_function_variables,
            optimizer_state,
            random_key,
            (jnp.array(False), jnp.nan, jnp.nan),
        ),
    )

    # Update target network
    target_q_value_function_variables = jax.lax.cond(
        (step_count >= agent_specification.warm_up_steps)
        & (step_count % agent_specification.target_network_update_frequency == 0),
        lambda: target_network_update(
            target_network_variables=target_q_value_function_variables,
            q_value_function_variables=q_value_function_variables,
            tau=agent_specification.target_network_update_tau,
        ),
        lambda: target_q_value_function_variables,
    )

    # Policy
    action, random_key, is_greedy = get_dqn_train_action(
        q_value_function=agent_specification.q_value_function,
        q_value_function_variables=q_value_function_variables,
        epsilon_schedule=agent_specification.epsilon_schedule,
        warm_up_steps=agent_specification.warm_up_steps,
        step_count=step_count,
        observation=observation,
        random_key=random_key,
        num_actions=agent_specification.num_actions,
        post_replay_buffer_preprocessing=agent_specification.post_replay_buffer_preprocessing,
    )

    # New agent state

    agent_state = agent_state.replace(
        replay_buffer_state=replay_buffer_state,
        data_collector_state=data_collector_state,
        q_value_function_variables=q_value_function_variables,
        optimizer_state=optimizer_state,
        random_key=random_key,
        target_q_value_function_variables=target_q_value_function_variables,
        last_action=action,
        step_count=step_count + 1,
    )

    # Logging data
    logging_dict = {
        "loss": losses,
        "mean_q_values": mean_q_values,
        "action_is_greedy": is_greedy,
    }

    return agent_state, logging_dict
