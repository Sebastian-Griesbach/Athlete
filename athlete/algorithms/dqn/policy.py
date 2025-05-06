from typing import Callable, Any, Optional, Dict, Tuple
import numpy as np
import torch
from gymnasium.spaces import Discrete

from athlete.function import numpy_to_tensor
from athlete.global_objects import StepTracker, RNGHandler
from athlete.policy.policy_builder import Policy, PolicyBuilder


class DQNTrainingPolicy(Policy):
    """Training policy implementation for Deep Q-Networks.

    This policy applies epsilon-greedy exploration during training. The epsilon
    value decays over time according to configured parameters.
    """

    SAVE_FILE_NAME = "dqn_training_policy"

    def __init__(
        self,
        q_value_function: torch.nn.Module,
        action_space: Discrete,
        start_epsilon: float,
        end_epsilon: float,
        epsilon_decay_steps: int,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]],
    ) -> None:
        """Initializes the DQN training policy.

        Args:
            q_value_function (torch.nn.Module): The Q-value function to use for action selection.
            action_space (Discrete): The action space of the environment.
            start_epsilon (float): The starting value of epsilon for the decaying epsilon-greedy strategy.
            end_epsilon (float): The minimum value of epsilon for the decaying epsilon-greedy strategy.
            epsilon_decay_steps (int): The number of steps over which epsilon decays linearly from start_epsilon to end_epsilon.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]]):
                A function to preprocess the observation before passing it to the Q-value function.
        """
        self.q_value_function = q_value_function
        self.num_actions = action_space.n
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.module_device = next(self.q_value_function.parameters()).device
        self.step_tracker = StepTracker.get_instance()
        self.random_number_generator = RNGHandler.get_random_number_generator()

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.epsilon_delta = (start_epsilon - end_epsilon) / epsilon_decay_steps

    def _get_random_action(self) -> int:
        """Samples a random action from the action space in a reproducible way.

        Returns:
            int: A random action sampled from the action space.
        """

        # We don't use action_space.sample() to ensure reproducibility
        return self.random_number_generator.integers(low=0, high=self.num_actions)

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Choose an action based on the observation using the epsilon-greedy strategy.
        Respects a warmup period where only random actions are taken. The number of warmup steps is defined in the step tracker.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: The chosen action and a dictionary with additional information: the current epsilon value and whether the action was chosen greedily.
        """
        # Warmup period
        if not self.step_tracker.warmup_is_done:
            return self._get_random_action(), {
                "greedy": False,
                "epsilon": self.start_epsilon,
            }

        # Epsilon decay
        epsilon_threshold = max(
            self.end_epsilon,
            self.start_epsilon
            - self.epsilon_delta * self.step_tracker.interactions_after_warmup,
        )
        if self.random_number_generator.random() < epsilon_threshold:
            return self._get_random_action(), {
                "greedy": False,
                "epsilon": epsilon_threshold,
            }

        # Greedy action
        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            q_values = self.q_value_function(observation)
        action = torch.argmax(q_values, dim=1)
        action = action.item()
        return action, {"greedy": True, "epsilon": epsilon_threshold}


class DQNEvaluationPolicy(Policy):
    """The DQN evaluation policy.
    This policy implements the greedy action selection strategy
    and moves the data to the device of the Q-value function.
    """

    def __init__(
        self,
        q_value_function: torch.nn.Module,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]],
    ) -> None:
        """Initializes the DQN evaluation policy.

        Args:
            q_value_function (torch.nn.Module): The Q-value function to use for action selection.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]]):
                A function to preprocess the observation before passing it to the Q-value function.
        """
        self.q_value_function = q_value_function
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing
        self.module_device = next(self.q_value_function.parameters()).device

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Choose an action based on the observation using the greedy strategy.
        This policy does not respect a warmup period and always chooses the greedy action.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[int, Dict[str, Any]]: The chosen action and an empty dictionary.
        """

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)
        with torch.no_grad():
            q_values = self.q_value_function(observation)
        action = torch.argmax(q_values, dim=1)
        action = action.item()
        return action, {}


class DQNPolicyBuilder(PolicyBuilder):
    """The DQN policy builder."""

    def __init__(
        self,
        q_value_function: torch.nn.Module,
        action_space: Discrete,
        start_epsilon: float,
        end_epsilon: float,
        epsilon_decay_steps: int,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]],
    ) -> None:
        """Initializes the DQN policy builder handing it all dependencies to build the training and evaluation policies.

        Args:
            q_value_function (torch.nn.Module): The Q-value function to use for action selection.
            action_space (Discrete): The action space of the environment.
            start_epsilon (float): The starting value of epsilon for the decaying epsilon-greedy strategy.
            end_epsilon (float): The minimum value of epsilon for the decaying epsilon-greedy strategy.
            epsilon_decay_steps (int): The number of steps over which epsilon decays linearly from start_epsilon to end_epsilon.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]]):
                A function to preprocess the observation before passing it to the Q-value function.
        """
        super().__init__()
        self.q_value_function = q_value_function
        self.action_space = action_space
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

    def build_training_policy(self) -> Policy:
        """Builds the training policy of the DQN algorithm.

        Returns:
            Policy: The training policy.
        """
        return DQNTrainingPolicy(
            q_value_function=self.q_value_function,
            action_space=self.action_space,
            start_epsilon=self.epsilon,
            end_epsilon=self.end_epsilon,
            epsilon_decay_steps=self.epsilon_decay_steps,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    def build_evaluation_policy(self) -> Policy:
        """Builds the evaluation policy of the DQN algorithm.
        Returns:
            Policy: The evaluation policy.
        """
        return DQNEvaluationPolicy(
            q_value_function=self.q_value_function,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    @property
    def requires_rebuild_on_policy_change(self) -> bool:
        """Whether the policy builder requires a rebuild when the policy changes.

        Returns:
            bool: False, as during policy updates only the weights of the Q-value function are updated which
                are referenced by the policy classes.
        """
        return False
