from typing import Tuple

import jax
import flax
import jax.numpy as jnp

from athlete import constants


class TransitionDataCollectorState(flax.struct.PyTreeNode):
    last_observation: jax.Array = flax.struct.field(pytree_node=True)
    episode_ended: jax.Array = flax.struct.field(pytree_node=True)


class Transition(flax.struct.PyTreeNode):
    observation: jax.Array = flax.struct.field(pytree_node=True)
    action: jax.Array = flax.struct.field(pytree_node=True)
    next_observation: jax.Array = flax.struct.field(pytree_node=True)
    reward: jax.Array = flax.struct.field(pytree_node=True)
    terminated: jax.Array = flax.struct.field(pytree_node=True)


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
    ) -> Tuple[TransitionDataCollectorState, Transition, jax.Array]:
        # add batch dimension
        reward = reward.reshape((1,))
        terminated = terminated.reshape((1,))

        valid = jnp.logical_not(collector_state.episode_ended)

        transition = Transition(
            observation=collector_state.last_observation,
            action=action,
            next_observation=observation,
            reward=reward,
            terminated=terminated,
        )

        new_collector_state = TransitionDataCollectorState(
            last_observation=observation,
            episode_ended=jnp.logical_or(terminated[0], truncated),
        )
        return new_collector_state, transition, valid
