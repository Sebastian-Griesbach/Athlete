from abc import ABC, abstractmethod
import importlib
import pickle
from typing import Tuple, Any, Dict
from dataclasses import dataclass

from athlete.algorithms.full_jax_dqn.function import resolve_dotted_reference

# TODO add proper interface structure for make functions

# TODO maybe write a function protocol for update functions to take update condition and return logging info


@dataclass
class AgentCheckpoint:
    load_from_payload_function_path: str
    checkpoint_payload: object


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

    def save_to_file(self, save_path: str) -> None:
        payload = self.get_save_payload()
        checkpoint = AgentCheckpoint(
            load_from_payload_function_path=f"{self.__class__.__module__}.{self.__class__.__qualname__}.{self.__class__.load_from_payload.__name__}",
            checkpoint_payload=payload,
        )
        with open(save_path, "wb") as file:
            pickle.dump(checkpoint, file)

    @abstractmethod
    def get_save_payload(
        self,
    ) -> object: ...

    @classmethod
    def load_from_file(
        cls,
        load_path: str,
    ) -> "Agent":
        with open(load_path, "rb") as file:
            checkpoint: AgentCheckpoint = pickle.load(file)

        # import load function
        load_function = resolve_dotted_reference(
            checkpoint.load_from_payload_function_path
        )

        return load_function(checkpoint.checkpoint_payload)

    @classmethod
    @abstractmethod
    def load_from_payload(cls, payload: object) -> "Agent": ...


class EvaluationAgent(ABC):
    @abstractmethod
    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def save(self, save_path: str, **kwargs) -> None: ...

    # TODO add abstract method for evaluation agents once implemented on lower levels
