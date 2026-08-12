from typing import Dict, Tuple

import jax
import flax

from athlete import constants


class FlatDataCollectorState(flax.struct.PyTreeNode):
    last_observation: jax.Array = flax.struct.field(pytree_node=True)


def flat_collect(
    collector_state: FlatDataCollectorState,
    action: jax.Array,
    observation: jax.Array,
    reward: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
) -> Tuple[FlatDataCollectorState, Dict[str, jax.Array]]:
    experience = {
        constants.DATA_OBSERVATIONS: collector_state.last_observation,
        constants.DATA_ACTIONS: action,
        constants.DATA_NEXT_OBSERVATIONS: observation,
        constants.DATA_REWARDS: reward,
        constants.DATA_TERMINATED: terminated,
    }

    new_collector_state = FlatDataCollectorState(last_observation=observation)
    return new_collector_state, experience


def flat_collect_reset(
    collector_state: FlatDataCollectorState,
    observation: jax.Array,
) -> FlatDataCollectorState:
    new_collector_state = FlatDataCollectorState(last_observation=observation)
    return new_collector_state
