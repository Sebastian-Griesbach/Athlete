from typing import Any, Dict, Optional, Tuple
import os
import datetime

from gymnasium.spaces import Space

from athlete.data_collection.collector import DataCollector
from athlete.update.update_rule import UpdateRule
from athlete.global_objects import StepTracker, RNGHandler
from athlete.policy.policy_builder import PolicyBuilder
from athlete.saving.saveable_component import CompositeSaveableComponent, SaveContext
from athlete.saving.file_handler import FileHandler, TorchFileHandler
from athlete import constants
from athlete.algorithms.registry import AlgorithmRegistry


class Agent(CompositeSaveableComponent):
    """The Agent class serves as the central component for reinforcement learning algorithms.

    This class contains all components needed to interact with the environment and perform updates.
    It is not an abstract interface but a concrete implementation that utilizes provided
    components (data collector, update rule, policy builder) to perform the desired RL algorithm.
    The Agent class handles both training and evaluation modes, manages checkpointing, and
    coordinates the flow of data between environment interactions and algorithm updates.
    """

    SAVE_FILE_NAME = "agent_save_stats"
    SAVE_OBSERVATION_SPACE = "observation_space"
    SAVE_ACTION_SPACE = "action_space"
    SAVE_ALGORITHM_ID = "algorithm_id"
    SAVE_ALGORITHM_ARGUMENTS = "algorithm_arguments"
    SAVE_METADATA = "save_metadata"

    def __init__(
        self,
        data_collector: DataCollector,
        update_rule: UpdateRule,
        policy_builder: PolicyBuilder,
        save_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the agent by providing the major components.

        Args:
            data_collector (DataCollector): Responsible for taking in the data given to the agent and formatting it for the update rule.
            update_rule (UpdateRule): Responsible for performing the learning update.
            policy_builder (PolicyBuilder): Responsible for creating the policies which give the actions returned by the agent.
            save_metadata (Optional[Dict[str, Any]], optional): Contains information that is relevant for saving the agent,
            for example the initialization arguments used in the make function.
        """
        CompositeSaveableComponent.__init__(self)
        # These are initialized in the make function
        self.step_tracker = StepTracker.get_instance()
        self.register_saveable_component("step_tracker", self.step_tracker)

        # The agent class itself does not need randomness, but we need to register the RNGHandler somewhere so it will be saved
        self.rng_handler = RNGHandler.get_instance()
        self.register_saveable_component("rng_handler", self.rng_handler)

        self.update_rule = update_rule
        self.register_saveable_component("update_rule", self.update_rule)
        self.data_collector = data_collector
        self.register_saveable_component("data_collector", self.data_collector)

        self.policy_builder = policy_builder

        self.save_metadata = save_metadata

        # Start in training mode
        self.train()
        self.training_policy = self.policy_builder.build_training_policy()
        self.evaluation_policy = self.policy_builder.build_evaluation_policy()

        self.last_action = None
        self.last_policy_info = {}

    def train(self) -> None:
        """Puts the agent into training mode. In this mode, the agent will perform updates and collect data."""
        self._training_mode = True
        # does not need to be rebuild here as this happened automatically after each update if needed

    def eval(self) -> None:
        """Puts the agent into evaluation mode. In this mode, the agent will not perform updates or collect data, It will simply
        return actions according to its current evaluation policy.
        """
        self._training_mode = False
        if self.policy_builder.requires_rebuild_on_policy_change:
            self.evaluation_policy = self.policy_builder.build_evaluation_policy()

    def step(
        self,
        observation: Any,
        reward: Any = None,
        terminated: Any = None,
        truncated: Any = None,
        env_info: Dict[str, Any] = {},
    ) -> Tuple[Any, Dict[str, Any]]:
        """Perform a step of the agent given information from the environment.
        In training mode the agent will perform updates according the specific algorithm.
        (Not every step results in an update depending on the update conditions of the components)
        In evaluation mode only passing an observation is valid ant the agent will simply return an action according to its current evaluation policy.


        Args:
            observation (Any): Current observation from the environment.
            reward (Any, optional): Last reward received from the environment. Optional in evaluation mode.
            terminated (Any, optional): True if the episode has terminated with the given observation. Optional in evaluation mode.
            truncated (Any, optional): True if the episode has been truncated with the given observation. Optional in evaluation mode.
            env_info (Dict[str, Any], optional): Optional environment information. Might be useful for some algorithms that use privileged information during training. Defaults to {}.

        Raises:
            ValueError: In training mode, observation, reward, terminated, and truncated must be provided.

        Returns:
            Tuple[Any, Dict[str, Any]]: A Tuple of the action to take in the environment and a dictionary containing information about the agent (update info and policy info).
            # The returned action is None if the episode has ended.
        """
        if self._training_mode:
            if reward is None or terminated is None or truncated is None:
                raise ValueError(
                    "In training mode, observation, reward, terminated, and truncated must be provided."
                )
            return self._train_step(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                environment_info=env_info,
            )
        return self._eval_step(observation=observation)

    def reset_step(
        self, observation: Any, info: Dict[str, Any] = {}
    ) -> Tuple[Any, Dict[str, Any]]:
        """Perform a step of the agent given information from the environment after a reset.
        Is affected in the same way as the step function by the training mode.
        This does not reset the agent, but informs it about a reset of the environment.

        Args:
            observation (Any): Initial observation from the environment after a reset.
            info (Dict[str, Any], optional): Initial information from the environment after a reset. Defaults to {}.

        Returns:
            Tuple[Any, Dict[str, Any]]: A Tuple of the action to take in the environment and a dictionary containing information about the agent (update info and policy info).
        """
        if self._training_mode:
            return self._reset_train_step(
                observation=observation, environment_info=info
            )
        return self._reset_eval_step(observation=observation)

    def _train_step(
        self,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        environment_info: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Performs the step of the agent in training mode."""

        # Add data to the data collector
        new_data_point_accumulated = self.data_collector.collect(
            action=self.last_action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            environment_info=environment_info,
            policy_info=self.last_policy_info,
        )
        # only increment environment interactions if the data of that action is received, such that we know it actually happened
        self.step_tracker.increment_environment_interaction()
        if new_data_point_accumulated:
            self.step_tracker.increment_datapoint()
        if terminated or truncated:
            # we need to increment this before the update as an update condition might depend on it.
            self.step_tracker.increment_episode()

        # Potentially perform an update, whether update is actually performed depends on the update conditions of the updatable components
        update_info = self._update()

        # Only return an action if the episode has not ended
        if not (terminated or truncated):
            self.last_action, self.last_policy_info = self.training_policy.act(
                observation=observation
            )
            # We distinguish between the update info and the policy info,
            # the policy info might contain relevant information for the update, e.g. what is the unscaled action of a policy,
            # Therefore the policy info is passed to the data receiver
            # the update info contains information about the update itself e.g. the loss of the value function
            agent_info = {**self.last_policy_info, **update_info}
            return self.last_action, agent_info
        return None, update_info

    def _reset_train_step(
        self, observation: Any, environment_info: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Performs the reset step of the agent in training mode."""
        new_data_point_accumulated = self.data_collector.collect_reset(
            observation=observation, environment_info=environment_info
        )

        update_info = {}
        if new_data_point_accumulated:
            update_info = self._update()

        self.last_action, self.last_policy_info = self.training_policy.reset_act(
            observation=observation
        )
        return self.last_action, update_info

    def _eval_step(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        """Performs the step of the agent in evaluation mode."""
        return self.evaluation_policy.act(observation=observation)

    def _reset_eval_step(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        """Performs the reset step of the agent in evaluation mode."""
        return self.evaluation_policy.reset_act(observation=observation)

    def _update(self) -> Dict[str, float]:
        """Performs the update of the agent. This is called after each step in training mode.
        This does not mean that an actual update happens after every step. When a update is performed depends on the
        update conditions of the components.

        Returns:
            Dict[str, float]: A dictionary containing information about the update (e.g. loss).
        """
        policy_changed, update_info = self.update_rule.update()
        # If policy has been changed, and policy changes require a rebuild, rebuild the training policy
        if policy_changed and self.policy_builder.requires_rebuild_on_policy_change:
            self.training_policy = self.policy_builder.build_training_policy()

        return update_info

    # Overwriting the save_checkpoint function from CompositeSaveableComponent to use it as entry point for saving
    def save_checkpoint(
        self,
        save_path: Optional[str] = None,
        file_handler: Optional[FileHandler] = None,
        save_environment_state: bool = False,
    ) -> str:
        """Save the complete current state of the agent to a checkpoint to continue training from.

        Args:
            save_path (Optional[str], optional): Path to the checkpoint where the agent should be saved. If None, a default path will be generated containing the algorithm ID and time of saving. Defaults to None.
            file_handler (Optional[FileHandler], optional): File handler to use for saving the checkpoint. If None, a default file handler will be used. Defaults to None.
            save_environment_state (bool, optional): Whether the the agent should save information that reflects the current state of the environment.
            If you want to continue training by continuing the currently running episode without a reset after loading the agent set this to true.
            Otherwise the agent assumes a new episode will start after loading. Defaults to False.

        Returns:
            str: Path to the checkpoint where the agent was saved. The checkpoint itself is a directory containing individual files from all saveable components.
        """

        # create save context

        if save_path is None:
            algorithm_id = self.save_metadata[self.SAVE_ALGORITHM_ID]
            human_readable_datetime = datetime.datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S"
            )
            save_path = os.path.join(
                os.getcwd(), "checkpoints", algorithm_id + "_" + human_readable_datetime
            )

        if file_handler is None:
            file_handler = TorchFileHandler()

        self.save_metadata[constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE] = (
            save_environment_state
        )

        save_context = SaveContext(
            file_handler=file_handler,
            save_path=save_path,
            prefix="",
            metadata=self.save_metadata,
        )

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        print(f"Saving checkpoint to {save_path}")

        # Save the save metadata, need for loading from checkpoint
        save_context_metadata_path = os.path.join(
            save_path,
            save_context.prefix + self.SAVE_METADATA,
        )
        file_handler.save_to_file(
            to_save=self.save_metadata,
            save_path=save_context_metadata_path,
        )

        # Start cascade of saving all child savable components
        CompositeSaveableComponent.save_checkpoint(self, context=save_context)

        return save_path

    # We do not overwrite load_checkpoint which takes a save_context as input,
    # This is used in the load_from_checkpoint function. If we use the load_checkpoint function outside of it,
    # The agent must already be initialized and thus we have access to the needed save_context

    def _save_local_state(self, context: SaveContext) -> None:
        if context.metadata[constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE]:
            save_stats = (self.last_action, self.last_policy_info)
            save_stats_save_path = os.path.join(
                context.save_path,
                context.prefix + self.SAVE_FILE_NAME,
            )
            context.file_handler.save_to_file(
                to_save=save_stats, save_path=save_stats_save_path
            )

    def _load_local_state(self, context: SaveContext):
        if context.metadata[constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE]:
            save_stats_save_path = os.path.join(
                context.save_path, context.prefix + self.SAVE_FILE_NAME
            )
            save_stats = context.file_handler.load_from_file(
                load_path=save_stats_save_path
            )
            self.last_action, self.last_policy_info = save_stats

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        file_handler: Optional[FileHandler] = TorchFileHandler(),
        load_environment_state: bool = False,
    ) -> "Agent":
        """Load an agent from a checkpoint to continue training from or perform evaluation.

        Args:
            checkpoint_path (str): Path to the checkpoint to load the agent from.
            file_handler (Optional[FileHandler], optional): File handler to use for loading the checkpoint. Defaults to TorchFileHandler().
            load_environment_state (bool, optional): Whether the agent should load information that reflected the current state of the environment when it was saved.
            This can be only used if the agent was saved with save_environment_state=True. Defaults to False.

        Raises:
            ValueError: If load_environment_state is True and the agent was not saved with save_environment_state=True.

        Returns:
            Agent: The loaded agent ready to be used.
        """

        # Load metadata for initialization
        save_context_metadata_path = os.path.join(checkpoint_path, Agent.SAVE_METADATA)

        save_context_metadata = file_handler.load_from_file(
            load_path=save_context_metadata_path
        )

        if (
            load_environment_state
            and not save_context_metadata[
                constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE
            ]
        ):
            raise ValueError(
                "Cannot load environment state, as the agent was not saved with save_environment_state=True"
            )

        save_context_metadata[constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE] = (
            load_environment_state
        )

        agent = Agent.make(
            observation_space=save_context_metadata[Agent.SAVE_OBSERVATION_SPACE],
            action_space=save_context_metadata[Agent.SAVE_ACTION_SPACE],
            algorithm_id=save_context_metadata[Agent.SAVE_ALGORITHM_ID],
            **save_context_metadata[Agent.SAVE_ALGORITHM_ARGUMENTS],
        )

        save_context = SaveContext(
            file_handler=file_handler,
            save_path=checkpoint_path,
            prefix="",
            metadata=save_context_metadata,
        )

        agent.load_checkpoint(context=save_context)

        return agent

    @classmethod
    def make(
        cls,
        observation_space: Space,
        action_space: Space,
        algorithm_id: str,
        seed: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> "Agent":
        """Creates an agent according to the given algorithm ID and the provided observation and action space.

        Args:
            observation_space (Space): Observation space of the environment.
            action_space (Space): Action space of the environment.
            algorithm_id (str): Algorithm ID of the algorithm to use. This must be registered in the AlgorithmRegistry.
            seed (Optional[int], optional): Seed to use for the random number generator. If None, a random seed will be used. Defaults to None.
            **kwargs (Dict[str, Any]): Additional arguments to pass to the algorithm. These are passed to the algorithm's component factory and overwrite the default configuration of the algorithm.

        Returns:
            Agent: The created agent ready to be used.
        """

        algorithm = AlgorithmRegistry.get_algorithm(algorithm_id=algorithm_id)

        updated_configuration = algorithm.default_configuration.copy()
        updated_configuration.update(kwargs)

        # RNG Handler
        rng_handler = RNGHandler(seed)
        RNGHandler.set_global_instance(rng_handler=rng_handler)

        step_tracker = StepTracker(
            warmup_steps=updated_configuration.get(
                constants.GENERAL_ARGUMENT_WARMUP_STEPS, 0
            ),
        )
        StepTracker.set_global_instance(instance=step_tracker)

        # Needed for saving
        initialization_arguments = {
            Agent.SAVE_OBSERVATION_SPACE: observation_space,
            Agent.SAVE_ACTION_SPACE: action_space,
            Agent.SAVE_ALGORITHM_ID: algorithm_id,
            Agent.SAVE_ALGORITHM_ARGUMENTS: updated_configuration,
        }

        # Algorithm
        data_collector, update_rule, policy_builder = algorithm.component_factory(
            observation_space=observation_space,
            action_space=action_space,
            configuration=updated_configuration,
        )

        # Agent
        agent = cls(
            update_rule=update_rule,
            policy_builder=policy_builder,
            data_collector=data_collector,
            save_metadata=initialization_arguments,
        )

        return agent
