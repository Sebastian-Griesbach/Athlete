from typing import Dict, Any, Union
import numpy as np

from athlete import constants
from athlete.update.update_rule import UpdatableComponent
from athlete.global_objects import StepTracker
from athlete.data_collection.provider import UpdateDataProvider


class QTableUpdate(UpdatableComponent):
    """The update able component performing the Q table update."""

    LOG_TAG_LOSS = "loss"

    def __init__(
        self,
        update_data_provider: UpdateDataProvider,
        q_table: np.ndarray,
        learning_rate: float,
        discount: float,
        loss_log_tag: str = LOG_TAG_LOSS,
    ) -> None:
        """Initializes the Q table update component.

        Args:
            update_data_provider (UpdateDataProvider): The data provider used to communicate with the data collector.
            q_table (np.ndarray): The Q table to be updated.
            learning_rate (float): The learning rate for the Q table update.
            discount (float): The discount factor for the Q table update.
            loss_log_tag (str, optional): The tag used for logging the loss. Defaults to 'loss'.
        """
        super().__init__()

        self.update_data_provider = update_data_provider
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.discount = discount
        self.loss_log_tag = loss_log_tag
        self.step_tracker = StepTracker.get_instance()
        self._last_updated_on_interaction_tracker_id = (
            self.step_tracker.register_tracker(
                id="q_table_last_interaction_updated_on",
            )
        )

    def update(self) -> Union[None, Dict[str, Any]]:
        """Performs the Q table update.

        Returns:
            Union[None, Dict[str, Any]]: Logging information about the update containing the loss.
        """
        transition = self.update_data_provider.get_data()[0]

        # remove batch dimension as Q-learning works with single transitions
        state = transition[constants.DATA_OBSERVATIONS][0]
        action = transition[constants.DATA_ACTIONS][0]
        next_state = transition[constants.DATA_NEXT_OBSERVATIONS][0]
        reward = transition[constants.DATA_REWARDS][0]
        terminated = transition[constants.DATA_TERMINATEDS][0]

        # Q-learning update

        q_value = self.q_table[state, action]
        next_q_value = np.max(self.q_table[next_state])
        target = reward + (1 - terminated) * self.discount * next_q_value
        delta = target - q_value
        self.q_table[state, action] += self.learning_rate * delta

        # Track update step for update condition
        self.step_tracker.set_tracker_value(
            id=self._last_updated_on_interaction_tracker_id,
            value=self.step_tracker.interactions_after_warmup,
        )

        return {self.loss_log_tag: np.abs(delta).item()}

    @property
    def update_condition(self) -> bool:
        """Whether this update should be performed.

        Returns:
            bool: True if the warmup is done and no update was performed in this interaction yet
        """
        return (
            self.step_tracker.is_warmup_done
            and self.step_tracker.interactions_after_warmup
            > self.step_tracker.get_tracker_value(
                id=self._last_updated_on_interaction_tracker_id,
            )
        )
