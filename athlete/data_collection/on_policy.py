from typing import Any, Dict
import os

from gymnasium.spaces import Space
import numpy as np

from athlete.data_collection.collector import DataCollector
from athlete.function import gymnasium_value_to_batched_numpy_array
from athlete.data_collection.provider import UpdateDataProvider
from athlete import constants
from athlete.saving.saveable_component import SaveContext


class OnPolicyDataCollector(DataCollector):
    """A data collector designed specifically for on-policy reinforcement learning algorithms.

    This collector accumulates a fixed number of steps from environment interactions and
    provides them in batches to on-policy algorithms such as PPO. As commonly done in PPO it does not
    distinguish between terminal and truncated observations. It collects observations, actions,
    rewards, and log probabilities of actions taken by the policy. The collected data is then
    provided to the update data provider for further processing.
    """

    SAVE_FILE_NAME = "on_policy_data_collector"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        number_of_collection_steps: int,
        update_data_provider: UpdateDataProvider,
        policy_info_log_prob_key: str,
    ) -> None:
        """Initializes the data collector.

        Args:
            observation_space (Space): Observation space of the environment
            action_space (Space): Action space of the environment
            number_of_collection_steps (int): Number of steps to collect before updating the data provider,
            So how many step are part of one data point.
            update_data_provider (UpdateDataProvider): The data provider to update with the collected data.
            This is used to inject the data into the update rule.
            policy_info_log_prob_key (str): The key to use to get the log prob of the action from the policy info.
        """
        super().__init__()

        self.policy_info_log_prob_key = policy_info_log_prob_key
        self.update_data_provider = update_data_provider

        # One observation more to account for the last observation
        self.observations = np.zeros(
            shape=(number_of_collection_steps + 1, *observation_space.shape),
            dtype=observation_space.dtype,
        )
        self.actions = np.zeros(
            shape=(number_of_collection_steps, *action_space.shape),
            dtype=action_space.dtype,
        )
        self.log_probs = np.zeros(
            shape=(number_of_collection_steps, *action_space.shape),
            dtype=np.float64,
        )
        self.rewards = np.zeros(
            shape=(number_of_collection_steps, 1),
            dtype=np.float64,
        )

        self.next_dones = np.zeros(
            shape=(number_of_collection_steps, 1),
            dtype=np.bool_,
        )

        self.last_observation = None
        self.last_done = None
        self.collected_steps = 0
        self.number_of_collection_steps = number_of_collection_steps

    def _reset_memory(self) -> None:
        self.observations.fill(0)
        self.actions.fill(0)
        self.log_probs.fill(0)
        self.rewards.fill(0)
        self.next_dones.fill(0)

        self.collected_steps = 0

    def collect_reset(
        self, observation: np.ndarray, environment_info: Dict[str, Any]
    ) -> bool:

        self.last_observation = observation

        return False

    def collect(
        self,
        action: Any,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        environment_info: Dict[str, Any],
        policy_info: Dict[str, Any],
    ) -> bool:
        # This misses or terminal or truncated observations except if the very last one is one
        self.observations[self.collected_steps] = (
            gymnasium_value_to_batched_numpy_array(self.last_observation)
        )
        self.actions[self.collected_steps] = gymnasium_value_to_batched_numpy_array(
            action
        )
        self.log_probs[self.collected_steps] = gymnasium_value_to_batched_numpy_array(
            policy_info[self.policy_info_log_prob_key]
        )
        self.rewards[self.collected_steps] = gymnasium_value_to_batched_numpy_array(
            reward
        )
        # This is incorrect, a correct implementation would distinguish between termination and truncation
        # but this seems to be an accepted part of vanilla PPO, which is what this implementation is supposed to be
        self.next_dones[self.collected_steps] = gymnasium_value_to_batched_numpy_array(
            terminated or truncated
        )

        self.last_observation = observation

        self.collected_steps += 1

        collection_done = self.collected_steps >= self.number_of_collection_steps

        if collection_done:
            # Add the last observation
            self.observations[self.collected_steps] = (
                gymnasium_value_to_batched_numpy_array(self.last_observation)
            )

            memory_dict = {
                constants.DATA_OBSERVATIONS: self.observations.copy(),
                constants.DATA_ACTIONS: self.actions.copy(),
                constants.DATA_LOG_PROBS: self.log_probs.copy(),
                constants.DATA_REWARDS: self.rewards.copy(),
                constants.DATA_NEXT_DONES: self.next_dones.copy(),
            }

            self.update_data_provider.set_data(update_data=memory_dict)

            self._reset_memory()
            return True

        return False

    def save_checkpoint(self, context: SaveContext):
        if not context.metadata.get(
            constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE, False
        ):
            return

        to_save = (
            self.observations,
            self.actions,
            self.log_probs,
            self.rewards,
            self.next_dones,
            self.last_observation,
            self.collected_steps,
        )

        save_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        context.file_handler.save_to_file(to_save=to_save, save_path=save_path)

    def load_checkpoint(self, context: SaveContext):
        if not context.metadata.get(
            constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE, False
        ):
            self._reset_memory()
            self.last_observation = None
            return

        (
            self.observations,
            self.actions,
            self.log_probs,
            self.rewards,
            self.next_dones,
            self.last_observation,
            self.collected_steps,
        ) = context.file_handler.load_from_file(
            load_path=os.path.join(
                context.save_path, context.prefix + self.SAVE_FILE_NAME
            )
        )
