# TODO rename files according to ne structure

import jax
import flax
import flashbax as fbx


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
