from abc import ABC, abstractmethod, classmethod
from typing import Tuple, Any, Dict


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
