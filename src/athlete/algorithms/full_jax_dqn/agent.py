from functools import partial
from typing import Dict

import flax
import flashbax as fbx
import jax

from athlete.algorithms.full_jax_dqn.updateable_component import replay_buffer_update
from athlete.algorithms.full_jax_dqn.data_collector import (
    TransitionDataCollector,
    TransitionDataCollectorState,
)


class DQNAgentState(flax.struct.PyTreeNode):
    replay_buffer_state: fbx.FlatBufferState = flax.struct.field(pytree_node=True)
    data_collector_state: TransitionDataCollectorState = flax.struct.field(
        pytree_node=True
    )
    last_action: jax.Array = flax.struct.field(pytree_node=True)
    random_key: jax.Array = flax.struct.field(pytree_node=True)
    q_value_function_variables: Dict[str, jax.Array] = flax.struct.field(
        pytree_node=True
    )
    target_q_value_function_variables: Dict[str, jax.Array] = flax.struct.field(
        pytree_node=True
    )


class DQNAgentSpecification(flax.struct.PyTreeNode):
    replay_buffer_func: fbx.FlatBuffer = flax.struct.field(pytree_node=False)
    q_value_function: flax.linen.Module = flax.struct.field(pytree_node=False)


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
    collector_state, new_experience, has_new_data = TransitionDataCollector.collect(
        collector_state=agent_state.data_collector_state,
        action=agent_state.last_action,
        observation=observation,
        reward=reward,
        terminated=terminated,
    )

    agent_state = agent_state.replace(data_collector_state=collector_state)

    # Replay buffer update
    replay_buffer_state = jax.lax.cond(
        has_new_data,
        lambda: replay_buffer_update(
            agent_state.replay_buffer_func,
            agent_state.replay_buffer_state,
            new_experience,
        ),
        lambda: agent_state.replay_buffer_state,
    )

    agent_state = agent_state.replace(replay_buffer_state=replay_buffer_state)

    # Value function update

    return new_agent_state
