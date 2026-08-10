from typing import Dict, Tuple

import jax
import flax
import jax.numpy as jnp

from athlete import constants


class TransitionDataCollectorState(flax.struct.PyTreeNode):
    last_observation: jax.Array = flax.struct.field(pytree_node=True)
    episode_ended: jax.Array = flax.struct.field(pytree_node=True)


class TransitionDataCollector(flax.struct.PyTreeNode):

    def collect_reset(
        self,
        collector_state: TransitionDataCollectorState,
        observation: jax.Array,
    ) -> TransitionDataCollectorState:
        new_collector_state = TransitionDataCollectorState(
            last_observation=observation, episode_ended=jax.numpy.array(False)
        )
        return new_collector_state

    def collect(
        self,
        collector_state: TransitionDataCollectorState,
        action: jax.Array,
        observation: jax.Array,
        reward: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> Tuple[TransitionDataCollectorState, Dict[str, jax.Array], jax.Array]:
        # add batch dimension
        reward = reward.reshape((1,))
        terminated = terminated.reshape((1,))

        valid = jnp.logical_not(collector_state.episode_ended)

        transition = {
            constants.DATA_OBSERVATIONS: collector_state.last_observation,
            constants.DATA_ACTIONS: action,
            constants.DATA_NEXT_OBSERVATIONS: observation,
            constants.DATA_REWARDS: reward,
            constants.DATA_TERMINATED: terminated,
        }

        new_collector_state = TransitionDataCollectorState(
            last_observation=observation,
            episode_ended=jnp.logical_or(terminated[0], truncated),
        )
        return new_collector_state, transition, valid
