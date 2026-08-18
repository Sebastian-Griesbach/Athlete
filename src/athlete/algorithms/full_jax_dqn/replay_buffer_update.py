from typing import Callable, NamedTuple, Any, Callable, Generic, TypeVar
from functools import partial

import chex
import jax
import jax.numpy as jnp
import flax

from athlete import constants

Experience = TypeVar("Experience")
INDEX_DTYPE = jnp.int32


# Map dtype families to specific jax default dtype
def map_replay_buffer_dtype(dtype):
    dtype = jnp.dtype(dtype)

    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.int32

    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.float32

    if jnp.issubdtype(dtype, jnp.bool_):
        return jnp.bool_

    return dtype


class ExperiencePair(NamedTuple, Generic[Experience]):
    first: Experience
    second: Experience


class TransitionSample(NamedTuple, Generic[Experience]):
    experience: ExperiencePair[Experience]


# TODO Add Batch and sequence functionality from flashbax
# TODO Add functionality to also zeroth -first -second entry... to do efficient frame stacking (needs to consider episode boundaries for zero overwrites)


class EpisodeAwareFlatBufferState(flax.struct.PyTreeNode):
    experience: Any = flax.struct.field(pytree_node=True)
    valid_ids: jax.Array = flax.struct.field(pytree_node=True)
    valid_id_positions: jax.Array = flax.struct.field(pytree_node=True)
    is_valid_id: jax.Array = flax.struct.field(pytree_node=True)
    write_index: jax.Array = flax.struct.field(pytree_node=True)
    size: jax.Array = flax.struct.field(pytree_node=True)
    num_valid_ids: jax.Array = flax.struct.field(pytree_node=True)
    is_full: jax.Array = flax.struct.field(pytree_node=True)


@chex.dataclass(frozen=True)
class EpisodeAwareFlatBuffer(Generic[Experience]):
    init: Callable[[Experience], EpisodeAwareFlatBufferState]
    add: Callable[
        [EpisodeAwareFlatBufferState, Experience, jax.Array],
        EpisodeAwareFlatBufferState,
    ]
    sample: Callable[
        [EpisodeAwareFlatBufferState, chex.PRNGKey],
        TransitionSample[Experience],
    ]
    can_sample: Callable[[EpisodeAwareFlatBufferState], jax.Array]


def make_episode_aware_flat_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
) -> EpisodeAwareFlatBuffer:
    """Create a flat replay buffer that samples valid adjacent observation pairs.

    The buffer stores one entry per observation. `add` receives an extra boolean,
    `new_episode_started`, which must be true for reset observations. A transition
    is valid if its first entry has a successor from the same episode.
    """

    _validate_buffer_arguments(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
    )

    init_fn = partial(init, max_length=max_length)
    add_fn = partial(add, max_length=max_length)
    sample_fn = partial(
        sample,
        max_length=max_length,
        sample_batch_size=sample_batch_size,
    )
    can_sample_fn = partial(can_sample, min_length=min_length)

    return EpisodeAwareFlatBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
    )


make_flat_buffer = make_episode_aware_flat_buffer


def init(
    example_entry: Experience,
    max_length: int,
) -> EpisodeAwareFlatBufferState:
    experience = jax.tree.map(
        lambda leaf: jnp.zeros((max_length, *leaf.shape), dtype=leaf.dtype),
        example_entry,
    )

    return EpisodeAwareFlatBufferState(
        experience=experience,
        valid_ids=jnp.zeros((max_length,), dtype=INDEX_DTYPE),
        valid_id_positions=jnp.zeros((max_length,), dtype=INDEX_DTYPE),
        is_valid_id=jnp.zeros((max_length,), dtype=bool),
        write_index=jnp.array(0, dtype=INDEX_DTYPE),
        size=jnp.array(0, dtype=INDEX_DTYPE),
        num_valid_ids=jnp.array(0, dtype=INDEX_DTYPE),
        is_full=jnp.array(False),
    )


def add(
    state: EpisodeAwareFlatBufferState,
    entry: Experience,
    new_episode_started: jax.Array,
    max_length: int,
) -> EpisodeAwareFlatBufferState:
    new_episode_started = jnp.asarray(new_episode_started, dtype=bool).reshape(())
    write_index = state.write_index
    previous_index = (write_index - 1) % max_length
    has_previous_entry = state.size > 0

    state = _remove_valid_id(state=state, entry_id=write_index)

    experience = jax.tree.map(
        lambda experience_leaf, entry_leaf: experience_leaf.at[write_index].set(
            entry_leaf
        ),
        state.experience,
        entry,
    )
    state = state.replace(experience=experience)

    previous_entry_is_valid = has_previous_entry & jnp.logical_not(new_episode_started)
    state = _set_valid_id(
        state=state,
        entry_id=previous_index,
        should_be_valid=previous_entry_is_valid,
    )

    new_size = jnp.minimum(state.size + 1, max_length)
    new_write_index = (write_index + 1) % max_length

    return state.replace(
        write_index=new_write_index,
        size=new_size,
        is_full=state.is_full | (new_size == max_length),
    )


def sample(
    state: EpisodeAwareFlatBufferState,
    rng_key: chex.PRNGKey,
    max_length: int,
    sample_batch_size: int,
) -> TransitionSample[Experience]:
    valid_id_positions = jax.random.randint(
        rng_key,
        shape=(sample_batch_size,),
        minval=0,
        maxval=state.num_valid_ids,
        dtype=state.valid_ids.dtype,
    )
    first_ids = state.valid_ids[valid_id_positions]
    second_ids = (first_ids + 1) % max_length

    first = jax.tree.map(lambda leaf: leaf[first_ids], state.experience)
    second = jax.tree.map(lambda leaf: leaf[second_ids], state.experience)

    return TransitionSample(experience=ExperiencePair(first=first, second=second))


def can_sample(
    state: EpisodeAwareFlatBufferState,
    min_length: int,
) -> jax.Array:
    return state.num_valid_ids >= min_length


def _set_valid_id(
    state: EpisodeAwareFlatBufferState,
    entry_id: jax.Array,
    should_be_valid: jax.Array,
) -> EpisodeAwareFlatBufferState:
    return jax.lax.cond(
        should_be_valid,
        lambda state: _add_valid_id(state=state, entry_id=entry_id),
        lambda state: _remove_valid_id(state=state, entry_id=entry_id),
        state,
    )


def _add_valid_id(
    state: EpisodeAwareFlatBufferState,
    entry_id: jax.Array,
) -> EpisodeAwareFlatBufferState:
    entry_is_already_valid = state.is_valid_id[entry_id]

    def add_id(
        state: EpisodeAwareFlatBufferState,
    ) -> EpisodeAwareFlatBufferState:
        position = state.num_valid_ids
        valid_ids = state.valid_ids.at[position].set(entry_id)
        valid_id_positions = state.valid_id_positions.at[entry_id].set(position)
        is_valid_id = state.is_valid_id.at[entry_id].set(True)

        return state.replace(
            valid_ids=valid_ids,
            valid_id_positions=valid_id_positions,
            is_valid_id=is_valid_id,
            num_valid_ids=state.num_valid_ids + 1,
        )

    return jax.lax.cond(
        entry_is_already_valid,
        lambda state: state,
        add_id,
        state,
    )


def _remove_valid_id(
    state: EpisodeAwareFlatBufferState,
    entry_id: jax.Array,
) -> EpisodeAwareFlatBufferState:
    entry_is_valid = state.is_valid_id[entry_id]

    def remove_id(
        state: EpisodeAwareFlatBufferState,
    ) -> EpisodeAwareFlatBufferState:
        position = state.valid_id_positions[entry_id]
        last_position = state.num_valid_ids - 1
        last_entry_id = state.valid_ids[last_position]

        valid_ids = state.valid_ids.at[position].set(last_entry_id)
        valid_id_positions = state.valid_id_positions.at[last_entry_id].set(position)
        is_valid_id = state.is_valid_id.at[entry_id].set(False)

        return state.replace(
            valid_ids=valid_ids,
            valid_id_positions=valid_id_positions,
            is_valid_id=is_valid_id,
            num_valid_ids=state.num_valid_ids - 1,
        )

    return jax.lax.cond(
        entry_is_valid,
        remove_id,
        lambda state: state,
        state,
    )


def _validate_buffer_arguments(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
) -> None:
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if min_length < 1:
        raise ValueError("min_length must be at least 1.")
    if min_length > max_length - 1:
        raise ValueError("min_length cannot exceed max_length - 1.")
    if sample_batch_size < 1:
        raise ValueError("sample_batch_size must be at least 1.")
    if max_length >= jnp.iinfo(INDEX_DTYPE).max:
        raise ValueError("max_length is too large for int32 buffer indices.")


def flat_replay_buffer_transition_update(
    replay_buffer_func: EpisodeAwareFlatBuffer,
    replay_buffer_state: EpisodeAwareFlatBufferState,
    action: jax.Array,
    observation: jax.Array,
    reward: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    new_episode_started: jax.Array,
) -> EpisodeAwareFlatBufferState:
    experience = {
        constants.DATA_OBSERVATIONS: observation,
        constants.DATA_ACTIONS: action,
        constants.DATA_REWARDS: reward,
        constants.DATA_TERMINATEDS: terminated,
    }

    replay_buffer_state = replay_buffer_func.add(
        state=replay_buffer_state,
        entry=experience,
        new_episode_started=new_episode_started,
    )
    return replay_buffer_state
