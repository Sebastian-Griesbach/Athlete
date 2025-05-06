from typing import Tuple

from gymnasium.spaces import Discrete
import numpy as np

from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.q_learning.updatable_components import QTableUpdate
from athlete.data_collection.provider import UpdateDataProvider
from athlete.saving.saveable_component import CompositeSaveableComponent


class QLearningUpdate(UpdateRule, CompositeSaveableComponent):
    """The Update rule for Q-Learning.
    This class manages all updatable components and the saving and loading of stateful objects.
    """

    def __init__(
        self,
        observation_space: Discrete,
        action_space: Discrete,
        update_data_provider: UpdateDataProvider,
        discount: float,
        learning_rate: float,
    ) -> None:
        """Initializes the Q-Learning update rule.

        Args:
            observation_space (Discrete): Observation space of the environment.
            action_space (Discrete): Action space of the environment.
            update_data_provider (UpdateDataProvider): The data provider used to communicate with the data collector.
            discount (float): The discount factor for the Q table update.
            learning_rate (float): The learning rate for the Q table update.
        """

        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        # General stateful components

        num_states = observation_space.n
        num_actions = action_space.n

        self.q_table = np.zeros(shape=(num_states, num_actions))

        self.register_saveable_component("q_table", self.q_table)

        # Q-table update

        self.q_table_update = QTableUpdate(
            update_data_provider=update_data_provider,
            q_table=self.q_table,
            learning_rate=learning_rate,
            discount=discount,
            loss_log_tag=QTableUpdate.LOG_TAG_LOSS,
            changes_policy=True,
        )

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent]:
        """Returns all updatable components of the update rule in the order they should be updated in.
        Here it is only the Q table update.

        Returns:
            Tuple[UpdatableComponent]: The updatable components of the update rule as a tuple:
                - Q table update
        """
        return (self.q_table_update,)
