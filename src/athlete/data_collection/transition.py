from typing import Any, Dict, Union, Tuple
import os
import warnings

import numpy as np
from gymnasium.spaces import Space

from athlete.data_collection.provider import UpdateDataProvider
from athlete import constants
from athlete.function import (
    gymnasium_value_to_batched_numpy_array,
)
from athlete.saving.saveable_component import SaveContext
from athlete.data_collection.collector import DataCollector


class GymnasiumTransitionDataCollector(DataCollector):
    """A data collector specialized for collecting transitions from gymnasium environments.

    This collector is designed for off-policy reinforcement learning algorithms that use
    experience replay. It captures state transitions (observation, action, reward, next observation, terminated, truncated)
    from environment interactions and provides them to update rules through an update data provider.
    """

    SAVE_FILE_NAME = "gymnasium_transition_data_collector"

    def __init__(
        self,
        update_data_provider: UpdateDataProvider,
    ) -> None:
        self.update_data_provider = update_data_provider

        self.last_observation = None

        self.episode_ended = True

    def collect_reset(
        self, observation: Union[int, np.ndarray], environment_info: Dict[str, Any]
    ) -> bool:
        """Collecting the data from the environment on a reset call of the environment.
        This does not mean that the collector is reset.

        Args:
            observation (Union[int, np.ndarray]): The observation from the environment.
            environment_info (Dict[str, Any]): The environment info from the environment.

        Returns:
            bool: False, on reset we can not have a full transition.
        """
        if not self.episode_ended:
            warnings.warn(
                "The agent did not receive a terminated or truncated signal for the last episode."
            )

        self.last_observation = gymnasium_value_to_batched_numpy_array(observation)
        self.episode_ended = False

        return False

    def _get_step_data(
        self,
        action: Union[int, np.ndarray],
        observation: Union[int, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        environment_info: Dict[str, Any],
        policy_info: Dict[str, Any],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray], float, bool]:
        """Overwrite this function if you want to change what data is used to create the transition.
        This is for example useful when not the actual action is used for the update but the unscaled version of it.


        Args:
            action (Union[int, np.ndarray]): Action taken in the environment.
            observation (Union[int, np.ndarray]): Observation received from the environment.
            reward (float): Reward received from the environment.
            terminated (bool): Whether the observation was terminal.
            truncated (bool): Whether the episode is truncated.
            environment_info (Dict[str, Any]): Info dictionary from the environment.
            policy_info (Dict[str, Any]): Policy info from the agent.

        Returns:
            Tuple[Union[int, np.ndarray], Union[int, np.ndarray], float, bool]:
                The action, observation, reward and terminated signal to be used for the transition.
        """
        return action, observation, reward, terminated

    def collect(
        self,
        action: Union[int, np.ndarray],
        observation: Union[int, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        environment_info: Dict[str, Any],
        policy_info: Dict[str, Any],
    ) -> bool:
        """Collecting the data from the environment during a regular step.

        Args:
            action (Union[int, np.ndarray]): Action taken in the environment.
            observation (Union[int, np.ndarray]): Observation received from the environment.
            reward (float): Reward received from the environment.
            terminated (bool): Whether the observation was terminal.
            truncated (bool): Whether the episode is truncated.
            environment_info (Dict[str, Any]): Info dictionary from the environment.
            policy_info (Dict[str, Any]): Policy info from the agent.

        Raises:
            ValueError: If the previous episode has ended and the collect_reset function was not called.

        Returns:
            bool: True, as after the reset call, every further step will create a new transition.
        """

        if self.episode_ended:
            raise ValueError(
                f"The previous episode has ended, please call {self.collect_reset.__name__} before calling {self.collect.__name__} again."
            )

        action, observation, reward, terminated = self._get_step_data(
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            environment_info=environment_info,
            policy_info=policy_info,
        )

        action = gymnasium_value_to_batched_numpy_array(action)
        reward = gymnasium_value_to_batched_numpy_array(reward)
        terminated = gymnasium_value_to_batched_numpy_array(terminated)
        observation = gymnasium_value_to_batched_numpy_array(observation)

        transition = {
            constants.DATA_OBSERVATIONS: self.last_observation,
            constants.DATA_ACTIONS: action,
            constants.DATA_REWARDS: reward,
            constants.DATA_NEXT_OBSERVATIONS: observation,
            constants.DATA_TERMINATEDS: terminated,
        }

        self.episode_ended = terminated or truncated
        metadata = {
            constants.METADATA_EPISODE_ENDED: np.array([self.episode_ended]),
        }
        self.update_data_provider.set_data(update_data=transition, metadata=metadata)

        self.last_observation = observation

        return True

    def save_checkpoint(self, context: SaveContext) -> None:
        """Save the current state of the data collector to a file if
        the save environment state argument is set to true.

        Args:
            context (SaveContext): The context of the save operation, containing the save path and metadata.
        """
        if not context.metadata.get(
            constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE, False
        ):
            return

        to_save = (
            self.last_observation,
            self.episode_ended,
        )

        save_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        context.file_handler.save_to_file(to_save=to_save, save_path=save_path)

    def load_checkpoint(self, context: SaveContext) -> None:
        """Load the state of the data collector from a file if
        the save environment state argument is set to true.

        Args:
            context (SaveContext): The context of the load operation, containing the save path and metadata.
        """
        if not context.metadata.get(
            constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE, False
        ):
            self.episode_ended = True
            return

        load_path = os.path.join(
            context.save_path, context.prefix + self.SAVE_FILE_NAME
        )

        loaded = context.file_handler.load_from_file(load_path=load_path)

        (
            self.last_observation,
            self.episode_ended,
        ) = loaded


class ActionReplacementGymnasiumTransitionDataCollector(
    GymnasiumTransitionDataCollector
):
    """This variant of the GymnasiumTransitionDataCollector is used to replace the action
    in the transition with a value from the policy info dictionary. This is useful for
    example if during the update we want to use the unscaled action instead of the scaled action.
    """

    def __init__(
        self,
        policy_info_replacement_key: str,
        update_data_provider: UpdateDataProvider,
    ):
        """Initialize the ActionReplacementGymnasiumTransitionDataCollector.

        Args:
            policy_info_replacement_key (str): The key in the policy info dictionary
                which should be used to replace the action in the transition.
            update_data_provider (UpdateDataProvider): The update data provider to be used for
                communication between the data collector and the update rule.
        """
        super().__init__(
            update_data_provider=update_data_provider,
        )

        self.policy_info_replacement_key = policy_info_replacement_key

    def _get_step_data(
        self,
        action: Union[int, np.ndarray],
        observation: Union[int, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        environment_info: Dict[str, Any],
        policy_info: Dict[str, Any],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray], float, bool]:
        """Replace the action in the transition with a value from the policy info dictionary.

        Args:
            action (Union[int, np.ndarray]): The action taken in the environment.
            observation (Union[int, np.ndarray]): The observation received from the environment.
            reward (float): The reward received from the environment.
            terminated (bool): Whether the observation was terminal.
            truncated (bool): Whether the episode is truncated.
            environment_info (Dict[str, Any]): The environment info from the environment.
            policy_info (Dict[str, Any]): The policy info from the agent.

        Returns:
            Tuple[Union[int, np.ndarray], Union[int, np.ndarray], float, bool]:
                The action, observation, reward and terminated signal to be used for the transition.
        """
        return (
            policy_info[self.policy_info_replacement_key],
            observation,
            reward,
            terminated,
        )
