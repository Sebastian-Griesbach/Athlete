from abc import ABC, abstractmethod
import importlib
import pickle
from typing import Tuple, Any, Dict
from dataclasses import dataclass

# TODO add proper interface structure for make functions

# TODO maybe write a function protocol for update functions to take update condition and return logging info


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
    def save(self, save_path: str, **kwargs) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, load_path: str, **kwargs) -> "Agent": ...


class EvaluationAgent(ABC):
    @abstractmethod
    def step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def reset_step(self, observation: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]: ...

    @abstractmethod
    def save(self, save_path: str, **kwargs) -> None: ...

    # TODO add abstract method for evaluation agents once implemented on lower levels
