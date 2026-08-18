from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import flax
import jax
import numpy as np
import gymnasium as gym

from athlete.algorithms.full_jax_dqn.jax_interface import JaxAgent, JaxEvaluationAgent

# TODO maybe write a function protocol for update functions to take update condition and return logging info


class LogValue(flax.struct.PyTreeNode):
    value: jax.Array
    valid: jax.Array


class Agent(ABC):
    @abstractmethod
    def step(
        self,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def make_evaluation_agent(self) -> "EvaluationAgent": ...

    @abstractmethod
    def save_agent(self, save_path: str) -> None: ...


class EvaluationAgent(ABC):
    @abstractmethod
    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def save_agent(self, save_path: str) -> None: ...


def convert_jax_action(action: jax.Array, action_space: gym.Space) -> Any:
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action.item())
    elif isinstance(action_space, gym.spaces.Box):
        return np.array(action)
    else:
        raise NotImplementedError(
            f"Action space type {type(action_space)} is not supported."
        )


class JaxAgentWrapper(Agent):
    def __init__(
        self,
        jax_agent: JaxAgent,
        agent_state: flax.struct.PyTreeNode,
        action_space: gym.Space,
    ):
        self.jax_agent = jax_agent
        self.agent_state = agent_state
        self.action_space = action_space

    def step(
        self,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_agent.step(
            agent_state=self.agent_state,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        converted_action = convert_jax_action(action, self.action_space)
        return converted_action, agent_info

    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_agent.reset_step(
            agent_state=self.agent_state, observation=observation
        )
        converted_action = convert_jax_action(action, self.action_space)
        return converted_action, agent_info

    def make_evaluation_agent(self) -> "EvaluationAgent":
        jax_eval_agent_state, jax_eval_agent = self.jax_agent.make_evaluation_agent(
            agent_state=self.agent_state,
        )
        return JaxEvaluationAgentWrapper(
            jax_eval_agent=jax_eval_agent,
            agent_state=jax_eval_agent_state,
            action_space=self.action_space,
        )

    # TODO Save functionality, should also save state and config, upon loading rebuild agent and set state
    def save_agent(self, save_path: str) -> None:
        pass

    # TODO load function should be on the same level as make function


class JaxEvaluationAgentWrapper(EvaluationAgent):
    def __init__(
        self,
        jax_eval_agent: JaxEvaluationAgent,
        agent_state: flax.struct.PyTreeNode,
        action_space: gym.Space,
    ):
        self.jax_eval_agent = jax_eval_agent
        self.agent_state = agent_state
        self.action_space = action_space

    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_eval_agent.step(
            agent_state=self.agent_state, observation=observation
        )
        converted_action = convert_jax_action(action, self.action_space)
        return converted_action, agent_info

    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_eval_agent.reset_step(
            agent_state=self.agent_state, observation=observation
        )
        converted_action = convert_jax_action(action, self.action_space)
        return converted_action, agent_info

    def save_agent(self, save_path: str) -> None:
        pass  # TODO
