from functools import partial
from typing import Callable, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from athlete.algorithms.full_jax_dqn.updateable_component import (
    flat_replay_buffer_transition_update,
    perform_n_q_value_function_updates,
    target_network_update,
)
from athlete.algorithms.full_jax_dqn.policy import (
    get_dqn_train_action,
    get_greedy_action,
)
from athlete.algorithms.full_jax_dqn.buffer import (
    EpisodeAwareFlatBuffer,
    EpisodeAwareFlatBufferState,
)
from athlete.algorithms.full_jax_dqn.jax_interface import JaxEvaluationAgent


class DQNAgentState(flax.struct.PyTreeNode):
    replay_buffer_state: EpisodeAwareFlatBufferState = flax.struct.field(
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
    optimizer_state: optax.OptState = flax.struct.field(pytree_node=True)
    step_count: jax.Array = flax.struct.field(pytree_node=True)


class DQNEvaluationAgentState(flax.struct.PyTreeNode):
    q_value_function_variables: Dict[str, jax.Array] = flax.struct.field(
        pytree_node=True
    )


class DQNAgentSpecification(flax.struct.PyTreeNode):
    replay_buffer_func: EpisodeAwareFlatBuffer = flax.struct.field(pytree_node=False)
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


class DQNEvaluationAgentSpecification(flax.struct.PyTreeNode):
    q_value_function: flax.linen.Module = flax.struct.field(pytree_node=False)
    post_replay_buffer_preprocessing: Callable = flax.struct.field(pytree_node=False)


# TODO add specification with make function as part of a class using partial similar to how buffer does it
# TODO make compatible for parallel environments
# TODO think about how to separate part of agent state that is required for evaluation
# TODO save and load functionality


def _reshape_to_single_value_array(array: jax.Array) -> jax.Array:
    return jnp.asarray(array).reshape((1,))


@partial(
    jax.jit,
    static_argnames=("agent_specification",),
    donate_argnames=("agent_state",),
)
def dqn_train_reset_step(
    agent_specification: DQNAgentSpecification,
    agent_state: DQNAgentState,
    observation: jax.Array,
) -> Tuple[DQNAgentState, jax.Array, Dict[str, jax.Array]]:
    observation = jnp.asarray(observation)

    # all state objects that might get updated
    replay_buffer_state = agent_state.replay_buffer_state
    random_key = agent_state.random_key
    # We only advance the step count for a step in the environment, which has happened if train_step is called

    # Replay buffer update
    replay_buffer_state = flat_replay_buffer_transition_update(
        replay_buffer_func=agent_specification.replay_buffer_func,
        replay_buffer_state=replay_buffer_state,
        observation=observation,
        reward=jnp.full((1,), jnp.nan),  # no reward for reset step
        action=jnp.full(
            (1,),
            agent_specification.num_actions,
            dtype=agent_state.last_action.dtype,
        ),  # invalid action for reset step
        terminated=jnp.array([False]),
        truncated=jnp.array([False]),
        new_episode_started=jnp.array(True),  # reset step starts a new episode
    )

    # Policy
    action, random_key, is_greedy, action_data_valid = get_dqn_train_action(
        q_value_function=agent_specification.q_value_function,
        q_value_function_variables=agent_state.q_value_function_variables,
        epsilon_schedule=agent_specification.epsilon_schedule,
        warm_up_steps=agent_specification.warm_up_steps,
        step_count=agent_state.step_count,
        observation=observation,
        random_key=random_key,
        num_actions=agent_specification.num_actions,
        post_replay_buffer_preprocessing=agent_specification.post_replay_buffer_preprocessing,
    )

    # New agent state
    agent_state = agent_state.replace(
        replay_buffer_state=replay_buffer_state,
        last_action=action,
        random_key=random_key,
    )

    # Logging data
    logging_dict = {
        "action_is_greedy": is_greedy,
        "action_data_valid": action_data_valid,
    }
    return agent_state, action, logging_dict


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
) -> Tuple[DQNAgentState, jax.Array, Dict[str, jax.Array]]:
    observation = jnp.asarray(observation)
    reward = _reshape_to_single_value_array(reward)
    terminated = _reshape_to_single_value_array(terminated).astype(bool)
    truncated = _reshape_to_single_value_array(truncated).astype(bool)

    # all state objects that might get updated
    replay_buffer_state = agent_state.replay_buffer_state
    q_value_function_variables = agent_state.q_value_function_variables
    optimizer_state = agent_state.optimizer_state
    random_key = agent_state.random_key
    target_q_value_function_variables = agent_state.target_q_value_function_variables
    last_action = _reshape_to_single_value_array(agent_state.last_action).astype(
        agent_state.last_action.dtype
    )
    step_count = agent_state.step_count

    # Replay buffer update
    replay_buffer_state = flat_replay_buffer_transition_update(
        replay_buffer_func=agent_specification.replay_buffer_func,
        replay_buffer_state=replay_buffer_state,
        action=last_action,
        observation=observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        new_episode_started=jnp.array(
            False
        ),  # we know this to be tru if step was called instead of reset_step
    )

    # Value function update
    (
        q_value_function_variables,
        optimizer_state,
        random_key,
        (
            update_data_valid,
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

    action, random_key, is_greedy, action_data_valid = jax.lax.cond(
        jnp.logical_not(
            jnp.asarray(terminated | truncated, dtype=bool).reshape(())
        ),  # Only return an action if the episode continues
        lambda: get_dqn_train_action(
            q_value_function=agent_specification.q_value_function,
            q_value_function_variables=q_value_function_variables,
            epsilon_schedule=agent_specification.epsilon_schedule,
            warm_up_steps=agent_specification.warm_up_steps,
            step_count=step_count,
            observation=observation,
            random_key=random_key,
            num_actions=agent_specification.num_actions,
            post_replay_buffer_preprocessing=agent_specification.post_replay_buffer_preprocessing,
        ),
        lambda: (
            jnp.full_like(last_action, agent_specification.num_actions),
            random_key,
            jnp.array(False),
            jnp.array(False),
        ),
    )

    # New agent state

    agent_state = agent_state.replace(
        replay_buffer_state=replay_buffer_state,
        q_value_function_variables=q_value_function_variables,
        optimizer_state=optimizer_state,
        random_key=random_key,
        target_q_value_function_variables=target_q_value_function_variables,
        last_action=action,
        step_count=step_count + 1,
    )

    # Logging data
    logging_dict = {
        "update_data_valid": update_data_valid,
        "loss": losses,
        "mean_q_values": mean_q_values,
        "action_is_greedy": is_greedy,
        "action_data_valid": action_data_valid,
    }

    return agent_state, action, logging_dict


@partial(
    jax.jit,
    static_argnames=("agent_specification",),
)
def dqn_make_evaluation_agent(
    agent_specification: DQNAgentSpecification,
    agent_state: DQNAgentState,
) -> Tuple[DQNEvaluationAgentState, Callable, Callable]:

    evaluation_agent_state = DQNEvaluationAgentState(
        q_value_function_variables=agent_state.q_value_function_variables.copy()
    )  # use copy to break donation
    evaluation_agent_specification = DQNEvaluationAgentSpecification(
        q_value_function=agent_specification.q_value_function,
        post_replay_buffer_preprocessing=agent_specification.post_replay_buffer_preprocessing,
    )  # needs no copy since specification is static

    evaluation_agent = JaxEvaluationAgent(
        step=partial(
            dqn_eval_step,
            evaluation_agent_specification,
        ),
        reset_step=partial(
            dqn_eval_reset_step,
            evaluation_agent_specification,
        ),
    )
    return evaluation_agent_state, evaluation_agent


@partial(
    jax.jit,
    static_argnames=("agent_specification",),
    donate_argnames=("agent_state",),
)
def dqn_eval_step(
    agent_specification: DQNAgentSpecification,
    agent_state: DQNAgentState,
    observation: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:  # action, logging_dict
    observation = jnp.asarray(observation)

    action = get_greedy_action(
        q_value_function=agent_specification.q_value_function,
        q_value_function_variables=agent_state.q_value_function_variables,
        observation=observation,
        post_replay_buffer_preprocessing=agent_specification.post_replay_buffer_preprocessing,
    )

    # For DQN agent state is unchanged during evaluation

    return agent_state, action, {}


dqn_eval_reset_step = (
    dqn_eval_step  # For DQN eval reset step is the same as regular eval step
)
