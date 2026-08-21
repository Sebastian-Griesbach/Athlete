from typing import Tuple, Any, Dict

import flax
import jax
import numpy as np
import gymnasium as gym

from athlete.algorithms.full_jax_dqn.interface import (
    Agent,
    EvaluationAgent,
)
from athlete.algorithms.full_jax_dqn.jax_interface import (
    JaxAgent,
    JaxEvaluationAgent,
    InfoValue,
    JaxMakeSpecification,
    JaxAgentCheckpointPayload,
)


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
        make_specification: JaxMakeSpecification,
    ):
        self.jax_agent = jax_agent
        self.agent_state = agent_state
        self.action_space = action_space
        self.make_specification = make_specification
        self._log_keys = None

    def _clean_agent_info(
        self,
        agent_info: Dict[str, InfoValue],
    ) -> Dict[str, Any]:
        if self._log_keys is None:
            self._log_keys = tuple(agent_info.keys())

        return {
            key: log_value.value.item()
            for key in self._log_keys
            if bool((log_value := agent_info[key]).valid)
        }

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
        action, agent_info = jax.device_get((action, agent_info))
        converted_action = convert_jax_action(action, self.action_space)
        agent_info = self._clean_agent_info(agent_info)
        return converted_action, agent_info

    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_agent.reset_step(
            agent_state=self.agent_state, observation=observation
        )
        action, agent_info = jax.device_get((action, agent_info))
        converted_action = convert_jax_action(action, self.action_space)
        agent_info = self._clean_agent_info(agent_info)
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

    def get_save_payload(self) -> JaxAgentCheckpointPayload:
        return JaxAgent.get_save_payload(
            agent_state=self.agent_state,
            make_specification=self.make_specification,
        )

    @classmethod
    def load_from_payload(
        cls,
        payload: JaxAgentCheckpointPayload,
    ) -> "JaxAgentWrapper":
        agent, agent_state, make_specification = JaxAgent.load_from_payload(payload)
        return cls(
            jax_agent=agent,
            agent_state=agent_state,
            action_space=make_specification.make_arguments["action_space"],
            make_specification=make_specification,
        )


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
        self._log_keys = None

    def _clean_agent_info(
        self,
        agent_info: Dict[str, InfoValue],
    ) -> Dict[str, Any]:
        if self._log_keys is None:
            self._log_keys = tuple(agent_info.keys())

        return {
            key: log_value.value.item()
            for key in self._log_keys
            if bool((log_value := agent_info[key]).valid)
        }

    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_eval_agent.step(
            agent_state=self.agent_state, observation=observation
        )
        action, agent_info = jax.device_get((action, agent_info))
        converted_action = convert_jax_action(action, self.action_space)
        agent_info = self._clean_agent_info(agent_info)
        return converted_action, agent_info

    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.agent_state, action, agent_info = self.jax_eval_agent.reset_step(
            agent_state=self.agent_state, observation=observation
        )
        action, agent_info = jax.device_get((action, agent_info))
        converted_action = convert_jax_action(action, self.action_space)
        agent_info = self._clean_agent_info(agent_info)
        return converted_action, agent_info

    def save(self, save_path: str) -> None:
        pass  # TODO
