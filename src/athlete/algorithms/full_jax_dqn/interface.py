from abc import ABC, abstractmethod, classmethod
from typing import Tuple, Any, Dict

import flax

from athlete.algorithms.full_jax_dqn.jax_interface import JaxAgent, JaxEvaluationAgent


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

    @abstractmethod
    @classmethod
    def load_agent(cls, load_path: str) -> "Agent": ...


class EvaluationAgent(ABC):
    @abstractmethod
    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def save_agent(self, save_path: str) -> None: ...

    @abstractmethod
    @classmethod
    def load_agent(cls, load_path: str) -> "EvaluationAgent": ...


class JaxAgentWrapper(Agent):
    def __init__(self, jax_agent: JaxAgent, agent_state: flax.struct.PyTreeNode):
        self.jax_agent = jax_agent
        self.agent_state = agent_state

    def step(
        self,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_agent.step(
            self.agent_state, observation, reward, terminated, truncated
        )
        return action, agent_info

    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_agent.reset_step(
            self.agent_state, observation
        )
        return action, agent_info

    def make_evaluation_agent(self) -> "EvaluationAgent":
        jax_eval_agent = self.jax_agent.make_evaluation_agent(
            self.agent_state, self.agent_state
        )
        return JaxEvaluationAgentWrapper(jax_eval_agent, self.agent_state)

    # TODO Save functionality, should also save state and config, upon loading rebuild agent and set state


class JaxEvaluationAgentWrapper(EvaluationAgent):
    def __init__(
        self, jax_eval_agent: JaxEvaluationAgent, agent_state: flax.struct.PyTreeNode
    ):
        self.jax_eval_agent = jax_eval_agent
        self.agent_state = agent_state

    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_eval_agent.step(
            self.agent_state, observation
        )
        return action, agent_info

    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_eval_agent.reset_step(
            self.agent_state, observation
        )
        return action, agent_info
