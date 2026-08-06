import flax
import flashbax as fbx


class DQNAgentState(flax.struct.PyTreeNode):
    replay_buffer_func: fbx.FlatBuffer = flax.struct.field(pytree_node=False)
    replay_buffer_state: fbx.FlatBufferState = flax.struct.field(pytree_node=True)
