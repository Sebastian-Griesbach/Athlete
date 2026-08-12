from functools import partial
from typing import Callable, Dict

import flax
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

from athlete.algorithms.full_jax_dqn.updateable_component import (
    dqn_value_update,
    replay_buffer_update,
)
from athlete.algorithms.full_jax_dqn.data_collector import (
    FlatDataCollectorState,
    flat_collect,
    flat_collect_reset,
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
    step_count: int = flax.struct.field(pytree_node=True)


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


# TODO Figure out how Data collection and Policy works

# Use Donation for agent state for efficient memory usage
# use lax.scan for multiple gradient updates for faster compilation time
# consider if we need a DQN Agent class
# we need a collection of dynamic objects (state)
# and a collection of static objects (e.g. discount, learning rate, etc.)
# update function should probably be stand alone
# make update  conditions also stand alone that can be paired with update functions

# TODO no Data Collector needed here in this form due to how flat flashbax buffer works
# add observation _t with action, reward, and terminated t+1, flash back samples two
# sequential entries such next observation comes from second, if terminated is true in first, ignore next observation, which would be invalid (initial state)


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

    # Data collection
    # TODO think how and if to abstract this
    data_collector_state, experience = flat_collect(
        collector_state=agent_state.data_collector_state,
        action=agent_state.last_action,
        observation=observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )
    agent_state = agent_state.replace(data_collector_state=data_collector_state)

    # Replay buffer update
    replay_buffer_state = jax.lax.cond(
        True,
        lambda: replay_buffer_update(
            agent_state.replay_buffer_func,
            agent_state.replay_buffer_state,
            experience,
        ),
        lambda: agent_state.replay_buffer_state,
    )

    agent_state = agent_state.replace(replay_buffer_state=replay_buffer_state)

    # Value function update
    # TODO use scan to perform multiple updates
    (
        q_value_function_variables,
        optimizer_state,
        replay_buffer_state,
        random_key,
        logging_dict,
    ) = (
        jax.lax.cond(
            agent_state.step_count >= agent_specification.warm_up_steps
            and agent_state.step_count
            % agent_specification.value_function_update_frequency
            == 0,
            dqn_value_update(
                q_value_function=agent_specification.q_value_function,
                q_value_function_variables=agent_state.q_value_function_variables,
                target_q_value_function_variables=agent_state.target_q_value_function_variables,
                optimizer=agent_specification.optimizer,
                optimizer_state=agent_state.optimizer_state,
                replay_buffer_func=agent_state.replay_buffer_func,
                replay_buffer_state=agent_state.replay_buffer_state,
                discount=agent_specification.discount,
                criteria=agent_specification.criteria,
                double_q=agent_specification.double_q,
                minto=agent_specification.minto,
                random_key=agent_state.random_key,
            ),
        ),
        lambda: (
            agent_state.q_value_function_variables,
            agent_state.optimizer_state,
            agent_state.replay_buffer_state,
            agent_state.random_key,
            {
                constants.LOGGING_DATA_VALID: jnp.array(True),
                "loss": jnp.nan,
                "mean_q_values": jnp.nan,
            },
        ),
    )

    new_agent_state = agent_state.replace(
        q_value_function_variables=q_value_function_variables,
        optimizer_state=optimizer_state,
        replay_buffer_state=replay_buffer_state,
        random_key=random_key,
    )
    return new_agent_state
