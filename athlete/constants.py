"""This file contains some constants that are used in different parts of the codebase."""

ENVIRONMENT_DATA_ACTION = "action"
ENVIRONMENT_DATA_OBSERVATION = "observation"
ENVIRONMENT_DATA_REWARD = "reward"
ENVIRONMENT_DATA_TERMINATED = "terminated"
ENVIRONMENT_DATA_TRUNCATED = "truncated"
ENVIRONMENT_DATA_INFO = "info"

DATA_RETURNS = "returns"
DATA_REWARDS = "rewards"
DATA_OBSERVATIONS = "observations"
DATA_ACTIONS = "actions"
DATA_PARAMETERS = "parameters"
DATA_DONES = "dones"
DATA_NEXT_DONES = "next_dones"
DATA_NEXT_OBSERVATIONS = "next_observations"
DATA_TERMINATEDS = "terminateds"
DATA_TRUNCATEDS = "truncateds"
DATA_LOG_PROBS = "log_probs"
DATA_VALUES = "values"
DATA_ADVANTAGES = "advantages"

ROLLOUT_DATA_OBSERVATIONS_PER_EPISODE = "observations_per_episode"
ROLLOUT_DATA_ACTIONS_PER_EPISODE = "actions_per_episode"
ROLLOUT_DATA_REWARDS_PER_EPISODE = "rewards_per_episode"
ROLLOUT_DATA_TERMINATEDS_PER_EPISODE = "terminateds_per_episode"
ROLLOUT_DATA_TRUNCATEDS_PER_EPISODE = "truncateds_per_episode"
ROLLOUT_DATA_INFOS_PER_EPISODE = "infos_per_episode"
ROLLOUT_DATA_STEPS = "steps"
ROLLOUT_DATA_VALUE_FUNCTION_PARAMETERS_PER_EPISODE = (
    "value_function_parameters_per_episode"
)
ROLLOUT_DATA_PARAMETERS_PER_EPISODE = "parameters_per_episode"
ROLLOUT_DATA_ACTOR_PARAMETERS_PER_EPISODE = "actor_parameters_per_episode"
ROLLOUT_DATA_ENVIRONMENT_SEEDS = "environment_seeds"

METADATA_EPISODE_ENDED = "episode_ended"

SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE = "save_environment_state"

GENERAL_ARGUMENT_WARMUP_STEPS = "warmup_steps"

GENERAL_ARGUMENT_OBSERVATION_SPACE = "observation_space"
GENERAL_ARGUMENT_ACTION_SPACE = "action_space"

VALUE_PLACEHOLDER = "PLACEHOLDER"
