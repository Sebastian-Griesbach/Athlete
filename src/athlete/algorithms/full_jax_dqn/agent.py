from typing import Dict

import flax
import flashbax as fbx

from athlete.algorithms.full_jax_dqn.updateable_component import replay_buffer_update


class DQNAgentState(flax.struct.PyTreeNode):
    replay_buffer_func: fbx.FlatBuffer = flax.struct.field(pytree_node=False)
    replay_buffer_state: fbx.FlatBufferState = flax.struct.field(pytree_node=True)


class DQNAgent:
    def __init__(self):
        pass

    # TODO Figure out how Data collection and Policy works

    # Use Donation for agent state for efficient memory usage
    # use lax.scan for multiple gradient updates for faster compilation time
    # consider if we need a DQN Agent class
    # we need a collection of dynamic objects (state)
    # and a collection of static objects (e.g. discount, learning rate, etc.)
    # update function should probably be stand alone
    # make update  conditions also stand alone that can be paired with update functions

    def update(
        self, agent_state: DQNAgentState, new_experience: Dict = None
    ) -> DQNAgentState:

        if new_experience is not None:
            new_replay_buffer_state = replay_buffer_update(
                agent_state.replay_buffer_func,
                agent_state.replay_buffer_state,
                new_experience,
            )

        # TODO figure out how Update conditions works

        new_agent_state = agent_state.replace(
            replay_buffer_state=new_replay_buffer_state
        )

        return new_agent_state
