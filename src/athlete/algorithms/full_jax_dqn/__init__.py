from typing import Callable, Dict, Any, Optional, Tuple
from functools import partial
import random
import copy

from gymnasium.spaces import Space, Box, Discrete
import optax
import jax.numpy as jnp
import jax

from athlete.algorithms.full_jax_dqn.agent_function import (
    DQNAgentState,
    DQNAgentSpecification,
)
from athlete import constants
from athlete.module.jax.common import FlaxFCDiscreteQValueFunction
from athlete.function import create_transition_data_info

from athlete.algorithms.full_jax_dqn.agent_function import (
    DQNAgentState,
    DQNAgentSpecification,
    dqn_train_step,
    dqn_train_reset_step,
    make_dqn_evaluation_agent,
)
from athlete.algorithms.full_jax_dqn.jax_interface import (
    JaxAgent,
    JaxMakeSpecification,
)
from athlete.algorithms.full_jax_dqn.interface import (
    Agent,
)
from athlete.algorithms.full_jax_dqn.jax_agent_wrapper import (
    JaxAgentWrapper,
)
from athlete.algorithms.full_jax_dqn.replay_buffer_update import (
    make_episode_aware_flat_buffer,
    map_replay_buffer_dtype,
)
from athlete.algorithms.full_jax_dqn.function import (
    identity,
    mean_squared_error,
    freeze_static_config,
)
from athlete import constants

# TODO maybe have different file to define all the functions and only expose those here which are supposed to be part of the "surface"

# TODO Consider only taking observation and action shape and dtypes as arguments to remove gymnasium dependency

# TODO maybe also add the load function back here again (just import and assign) so all the methods to create an agent of this algorithm are in one place


def make_jax_agent(
    observation_space: Space,
    action_space: Space,
    replay_buffer_capacity: int = 100_000,
    replay_buffer_mini_batch_size: int = 128,
    replay_buffer_frame_stacking: int = 1,
    replay_buffer_frame_stack_axis: int = 0,
    value_network_class: Any = FlaxFCDiscreteQValueFunction,
    value_network_arguments: Dict[str, Any] = {
        "observation_shape": constants.VALUE_PLACEHOLDER,
        "num_actions": constants.VALUE_PLACEHOLDER,
        "hidden_dims": (256, 256),
    },
    optimizer_class: Any = optax.adam,
    optimizer_arguments: Dict[str, Any] = {"learning_rate": 6.3e-4},
    seed: Optional[int] = None,
    discount: float = 0.99,
    loss_function: Callable[[jax.Array, jax.Array], jax.Array] = mean_squared_error,
    minto: bool = False,
    double_q: bool = False,
    value_function_update_frequency: int = 4,
    value_function_number_of_updates: int = 4,
    warm_up_steps: int = 1000,
    target_network_update_frequency: int = 250,
    target_network_update_tau: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay_steps: int = 12_000,
    post_replay_buffer_observation_preprocessing: Callable[
        [jax.Array], jax.Array
    ] = identity,
    log_loss: bool = False,
    log_mean_q_values: bool = False,
    log_greedy_action: bool = False,
) -> Tuple[JaxAgent, DQNAgentState, JaxMakeSpecification]:
    make_arguments = copy.deepcopy(locals())

    if not isinstance(observation_space, Box):
        raise ValueError(
            f"This DQN implementation only supports {Box.__name__} observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Discrete):
        raise ValueError(
            f"This DQN implementation only supports {Discrete.__name__} action spaces, but got {type(action_space)}"
        )

    # Replay Buffer
    replay_buffer = make_episode_aware_flat_buffer(
        max_length=replay_buffer_capacity,
        min_length=replay_buffer_mini_batch_size,
        sample_batch_size=replay_buffer_mini_batch_size,
        frame_stacking=replay_buffer_frame_stacking,
        frame_stacking_field=constants.DATA_OBSERVATIONS,
        frame_stack_axis=replay_buffer_frame_stack_axis,
    )

    transition_data_info = create_transition_data_info(
        observation_space=observation_space,
        action_space=action_space,
        flat_transition=True,
    )
    dummy_transition = {
        field: jnp.zeros(info["shape"], dtype=map_replay_buffer_dtype(info["dtype"]))
        for field, info in transition_data_info.items()
    }

    replay_buffer_state = replay_buffer.init(dummy_transition)

    # Value function
    dummy_observation = post_replay_buffer_observation_preprocessing(
        jnp.zeros((1, *observation_space.shape), dtype=jnp.float32)
    )

    # create new dictionary to avoid in place mutation
    value_network_arguments = {
        **value_network_arguments,
        "observation_shape": dummy_observation.shape[1:],
        "num_actions": action_space.n,
    }

    # Freeze dictionary to enable static configuration for jax
    frozen_value_network_arguments = freeze_static_config(value_network_arguments)
    q_value_function = value_network_class(**frozen_value_network_arguments)

    if seed is None:
        random_key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))
    else:
        random_key = jax.random.PRNGKey(seed)

    random_key, sub_key = jax.random.split(random_key)

    q_value_function_variables = target_q_value_function_variables = (
        q_value_function.init(sub_key, dummy_observation)
    )

    # Target network
    target_q_value_function_variables = jax.tree.map(
        jnp.copy,
        q_value_function_variables,
    )

    # Optimizer
    frozen_optimizer_arguments = freeze_static_config(optimizer_arguments)
    optimizer_function = optimizer_class(**frozen_optimizer_arguments)
    initial_optimizer_state = optimizer_function.init(q_value_function_variables)

    # Epsilon-greedy schedule
    epsilon_schedule = optax.linear_schedule(
        init_value=epsilon_start,
        end_value=epsilon_end,
        transition_steps=epsilon_decay_steps,
    )

    agent_state = DQNAgentState(
        replay_buffer_state=replay_buffer_state,
        q_value_function_variables=q_value_function_variables,
        target_q_value_function_variables=target_q_value_function_variables,
        random_key=random_key,
        optimizer_state=initial_optimizer_state,
        last_action=jnp.array(
            (action_space.n + 1,), dtype=jnp.int32
        ),  # Invalid action as placeholder
        step_count=jnp.array(0, dtype=jnp.int32),
    )

    agent_specification = DQNAgentSpecification(
        replay_buffer=replay_buffer,
        q_value_function=q_value_function,
        discount=discount,
        loss_function=loss_function,
        minto=minto,
        double_q=double_q,
        optimizer=optimizer_function,
        value_function_update_frequency=value_function_update_frequency,
        value_function_number_of_updates=value_function_number_of_updates,
        warm_up_steps=warm_up_steps,
        epsilon_schedule=epsilon_schedule,
        target_network_update_frequency=target_network_update_frequency,
        target_network_update_tau=target_network_update_tau,
        num_actions=action_space.n,
        post_replay_buffer_observation_preprocessing=post_replay_buffer_observation_preprocessing,
        log_loss=log_loss,
        log_mean_q_values=log_mean_q_values,
        log_greedy_action=log_greedy_action,
    )

    agent = JaxAgent(
        step=partial(dqn_train_step, agent_specification=agent_specification),
        reset_step=partial(
            dqn_train_reset_step, agent_specification=agent_specification
        ),
        make_evaluation_agent=partial(
            make_dqn_evaluation_agent, agent_specification=agent_specification
        ),
    )

    make_specification = JaxMakeSpecification(
        make_function_path=f"{make_jax_agent.__module__}.{make_jax_agent.__qualname__}",
        make_arguments=make_arguments,
    )

    return agent, agent_state, make_specification


def make_agent(
    observation_space: Space,
    action_space: Space,
    replay_buffer_capacity: int = 100_000,
    replay_buffer_mini_batch_size: int = 128,
    replay_buffer_frame_stacking: int = 1,
    replay_buffer_frame_stack_axis: int = 0,
    value_network_class: Any = FlaxFCDiscreteQValueFunction,
    value_network_arguments: Dict[str, Any] = {
        "observation_shape": constants.VALUE_PLACEHOLDER,
        "num_actions": constants.VALUE_PLACEHOLDER,
        "hidden_dims": (256, 256),
    },
    optimizer_class: Any = optax.adam,
    optimizer_arguments: Dict[str, Any] = {"learning_rate": 6.3e-4},
    seed: Optional[int] = None,
    discount: float = 0.99,
    loss_function: Callable[[jax.Array, jax.Array], jax.Array] = mean_squared_error,
    minto: bool = False,
    double_q: bool = False,
    value_function_update_frequency: int = 4,
    value_function_number_of_updates: int = 4,
    warm_up_steps: int = 1000,
    target_network_update_frequency: int = 250,
    target_network_update_tau: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay_steps: int = 12_000,
    post_replay_buffer_observation_preprocessing: Callable[
        [jax.Array], jax.Array
    ] = identity,
    log_loss: bool = False,
    log_mean_q_values: bool = False,
    log_greedy_action: bool = False,
) -> Agent:

    jax_agent, agent_state, make_specification = make_jax_agent(
        observation_space=observation_space,
        action_space=action_space,
        replay_buffer_capacity=replay_buffer_capacity,
        replay_buffer_mini_batch_size=replay_buffer_mini_batch_size,
        replay_buffer_frame_stacking=replay_buffer_frame_stacking,
        replay_buffer_frame_stack_axis=replay_buffer_frame_stack_axis,
        value_network_class=value_network_class,
        value_network_arguments=value_network_arguments,
        optimizer_class=optimizer_class,
        optimizer_arguments=optimizer_arguments,
        seed=seed,
        discount=discount,
        loss_function=loss_function,
        minto=minto,
        double_q=double_q,
        value_function_update_frequency=value_function_update_frequency,
        value_function_number_of_updates=value_function_number_of_updates,
        warm_up_steps=warm_up_steps,
        target_network_update_frequency=target_network_update_frequency,
        target_network_update_tau=target_network_update_tau,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        post_replay_buffer_observation_preprocessing=post_replay_buffer_observation_preprocessing,
        log_loss=log_loss,
        log_mean_q_values=log_mean_q_values,
        log_greedy_action=log_greedy_action,
    )

    return JaxAgentWrapper(
        jax_agent=jax_agent,
        agent_state=agent_state,
        action_space=action_space,
        make_specification=make_specification,
    )
