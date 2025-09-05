from typing import Any, Dict, Tuple
from functools import partial

from gymnasium.spaces import Discrete
import jax
import jax.numpy as jnp
import numpy as np

from athlete.global_objects import StepTracker, RNGHandler
from athlete.policy.policy import Policy
from athlete.jax_objects import MutableJaxModule, ModuleState, FunctionWrapper


@partial(jax.jit, static_argnames=["post_replay_buffer_preprocessing"])
def _jitable_act(
    q_value_function: ModuleState,
    observation: np.ndarray,
    post_replay_buffer_preprocessing: FunctionWrapper,
) -> int:

    observation = jnp.expand_dims(observation, axis=0)
    observation = post_replay_buffer_preprocessing(observation)
    q_values = q_value_function(observation)
    action = jnp.argmax(q_values, axis=1)
    return action


@jax.jit
def _jitable_get_random_action(num_actions: int, random_key: jax.Array) -> int:
    return jax.random.randint(random_key, shape=(), minval=0, maxval=num_actions)


@jax.jit
def _jitable_identity(observation: jnp.ndarray) -> jnp.ndarray:
    return observation


class JAXDQNTrainingPolicy(Policy):

    def __init__(
        self,
        mutable_q_value_function: MutableJaxModule,
        action_space: Discrete,
        start_epsilon: float,
        end_epsilon: float,
        epsilon_decay_steps: int,
        post_replay_buffer_preprocessing: FunctionWrapper = None,
    ) -> None:
        self.mutable_q_value_function = mutable_q_value_function
        self.num_actions = action_space.n
        if post_replay_buffer_preprocessing is None:
            self.post_replay_buffer_preprocessing = FunctionWrapper(
                function=_jitable_identity
            )
        else:
            self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.step_tracker = StepTracker.get_instance()
        self.rng_handler = RNGHandler.get_instance()

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.epsilon_delta = (start_epsilon - end_epsilon) / epsilon_decay_steps

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        # Warmup period
        if not self.step_tracker.is_warmup_done:
            return _jitable_get_random_action(
                num_actions=self.num_actions, random_key=self.rng_handler.get_jax_key()
            ).item(), {
                "greedy": False,
                "epsilon": self.start_epsilon,
            }

        # Epsilon decay
        epsilon_threshold = max(
            self.end_epsilon,
            self.start_epsilon
            - self.epsilon_delta * self.step_tracker.interactions_after_warmup,
        )
        if jax.random.uniform(key=self.rng_handler.get_jax_key()) < epsilon_threshold:
            return _jitable_get_random_action(
                num_actions=self.num_actions, random_key=self.rng_handler.get_jax_key()
            ).item(), {
                "greedy": False,
                "epsilon": epsilon_threshold,
            }

        # Greedy action
        action = _jitable_act(
            q_value_function=self.mutable_q_value_function.get(),
            observation=observation,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

        return action.item(), {
            "greedy": True,
            "epsilon": epsilon_threshold,
        }


class JAXDQNEvaluationPolicy(Policy):

    def __init__(
        self,
        mutable_q_value_function: MutableJaxModule,
        post_replay_buffer_preprocessing: FunctionWrapper = None,
    ) -> None:
        self.mutable_q_value_function = mutable_q_value_function
        if post_replay_buffer_preprocessing is None:
            self.post_replay_buffer_preprocessing = FunctionWrapper(
                function=_jitable_identity
            )
        else:
            self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:

        action = _jitable_act(
            q_value_function=self.mutable_q_value_function.get(),
            observation=observation,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )
        return action.item(), {}
