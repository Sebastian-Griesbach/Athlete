# Athlete

An API for interacting with reinforcement learning agents which closes the RL loop with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

Athlete provides a similar interface for reinforcement learning agents as Gymnasium does for environments. As such it enables the handling of RL agents while being agnostic towards the underlying RL algorithm. The following shows a minimal training loop using Athlete.

```python
import athlete
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
# Initialize the agent, all hyperparameters have default values
# which can be overridden with keyword arguments
agent = athlete.make(
    algorithm_id="dqn",
    action_space=env.action_space,
    observation_space=env.observation_space,
    seed=42 # optional
)

observation, env_info = env.reset(seed=42)
# Inform the agent about the new episode and generate first action
action, agent_info = agent.reset_step(observation, env_info)
for _ in range(20000):
    observation, reward, terminated, truncated, env_info = env.step(action)
    # Feed the agent new information to receive next action.
    # This will automatically perform updates as defined
    # (e.g. following update frequency, performing a warmup phase etc.).
    # agent_info contains internal information like the loss
    action, agent_info = agent.step(
            observation, reward, terminated, truncated, env_info
        )

    if terminated or truncated:
        observation, env_info = env.reset()
        action, agent_info = agent.reset_step(observation, env_info)

env.close()
```

# But why?

There are plenty of Reinforcement Learning libraries out there. Why did I bother creating Athlete?
Most RL libraries are built with a specific purpose in mind. They provide all the tools needed for that purpose.
For example, most libraries are built for academic research and therefore provide experiment setups, configurations and logging capabilities.
However, this specialization often makes them 1) hard to adapt to other purposes and 2) more complex than often needed.
Athlete has not been built for a specific application. It is just an API for reinforcement learning agents.
Regardless of whether you want to create an application that uses RL for research, stock trading, robotics, optimization or just for fun, Athlete allows you to create the code for your application without having to commit to any specific RL algorithm.

# Installation

You can install Athlete either directly from PyPI:

```bash
pip install rl-athlete
```

Or install it from source:

```bash
# Clone the repository
git clone https://github.com/Sebastian-Griesbach/Athlete.git
cd Athlete

# Install in development mode
pip install -e .
```

Development mode (`-e`) allows you to modify the code and have changes take effect without reinstalling.

# Implemented Algorithms

Athlete currently implements a small selection of popular RL algorithms:

- **DQN** (Deep Q-Network) - For discrete action spaces with options for double Q-learning - ID: `"dqn"`
- **SAC** (Soft Actor-Critic) - For continuous action spaces with automatic entropy tuning - ID: `"sac"`
- **DDPG** (Deep Deterministic Policy Gradient) - For continuous action spaces - ID: `"ddpg"`
- **TD3** (Twin Delayed DDPG) - For continuous action spaces - ID: `"td3"`
- **PPO** (Proximal Policy Optimization) - For continuous action spaces - ID: `"ppo"`
- **Q-Learning** - For discrete state and action spaces - ID: `"q_learning"`

# Features

Athlete provides several features to make working with reinforcement learning agents more flexible:

## Training and Evaluation Modes

Inspired by [PyTorch](https://github.com/pytorch/pytorch), Athlete agents have separate training and evaluation modes which you can switch between. The following is a simple example of a training loop with a specific number of environment interactions followed by an rendered evaluation with a specific number of episodes.

```python
import athlete
import gymnasium as gym

# Create environment and agent
train_env = gym.make("CartPole-v1")
agent = athlete.make(
    algorithm_id="dqn",
    action_space=train_env.action_space,
    observation_space=train_env.observation_space
)

# Training mode (default mode when agent is created)
agent.train()  # Explicitly set to training mode

# Training loop
observation, env_info = train_env.reset()
action, agent_info = agent.reset_step(observation, env_info)
# A progressbar added with tqdm might be helpful here
for _ in range(100000):
    observation, reward, terminated, truncated, env_info = train_env.step(action)
    action, agent_info = agent.step(
            observation, reward, terminated, truncated, env_info
        )

    if terminated or truncated:
        observation, env_info = train_env.reset()
        action, agent_info = agent.reset_step(observation, env_info)

# Switch to evaluation mode to disable exploration and updates
eval_env = gym.make("CartPole-v1", render_mode="human")
agent.eval()
for _ in range(10):
    observation, env_info = eval_env.reset()
    action, agent_info = agent.reset_step(observation)
    done = False
    while not done:
        # In eval mode, only the observation is required (but you can also pass everything)
        action, agent_info = agent.step(observation)
        observation, reward, terminated, truncated, env_info = eval_env.step(action)
        done = terminated or truncated
```

In training mode, the agent collects data and performs updates according to the algorithm's update schedule. In evaluation mode, the agent uses a potentially different policy (e.g., without exploration noise) and doesn't perform any updates or collect data.

## Full State Saving and Loading

Athlete allows you to save and load the complete state of your agent so that training can be paused and
continued later without affecting the result.

```python
import athlete
import gymnasium as gym

# Create and train agent
env = gym.make("CartPole-v1")
agent = athlete.make(
    algorithm_id="dqn",
    action_space=env.action_space,
    observation_space=env.observation_space
)

# Train for a while...

# Save the complete agent state
# Optionally you can also set the path where the checkpoint should be saved
checkpoint_path = agent.save_checkpoint()

del agent

# Create agent from checkpoint
loaded_agent = athlete.from_checkpoint(checkpoint_path)
# Need to reset the environment before continuing

# Continue training or run evaluation
```

By default, an agent assumes that upon loading it will not continue the same episode.
You can pass `save_environment_state=True` to the `save_checkpoint` function:

```python
# The agent will remember information related to the current episode
checkpoint_path = agent.save_checkpoint(save_environment_state=True)

# Save the environment state itself (not part of Athlete)

# On loading, you can decide if you want to continue the episode
loaded_agent = athlete.from_checkpoint(
    checkpoint_path=checkpoint_path,
    load_environment_state=True
)
```

## Reproducibility

Athlete supports seeding for reproducibility, even across saves.

```python
import athlete
import gymnasium as gym

# Set seed for reproducibility
seed = 42

env = gym.make("CartPole-v1")
# Create agent with a seed
agent = athlete.make(
    algorithm_id="dqn",
    observation_space=env.observation_space,
    action_space=env.action_space,
    seed=seed
)

# for this to work properly you also need to seed the environment
# which is outside of the responsibility of Athlete.
# e.g. you can do the following on the first environment reset.
observation, info = env.reset(seed=seed)
```

## Algorithm Registry

### athlete.list_algorithms()

Lists all registered algorithms in the library:

```python
import athlete

# Get a list of all available algorithm IDs
available_algorithms = athlete.list_algorithms()
print(available_algorithms)  # ['q_learning', 'dqn', 'ddpg', 'td3', 'sac', 'ppo']
```

### athlete.get_default_configuration(algorithm_id)

Retrieves the default configuration dictionary for a specific algorithm:

```python
import athlete
from pprint import pprint

# Get the default configuration for SAC
config = athlete.get_default_configuration("sac")

# Print the dictionary
pprint(config)
```

These are the values that you can override during the `make()` call.

### athlete.make(algorithm_id, observation_space, action_space, \*\*kwargs)

Creates an agent with the specified algorithm. The `**kwargs` parameter allows you to override any of the default configuration values:

```python
import athlete
import gymnasium as gym

env = gym.make("BipedalWalker-v3")

agent = athlete.make(
    algorithm_id="sac",
    observation_space=env.observation_space,
    action_space=env.action_space,
    # Override any default values:
    discount=0.98,
    replay_buffer_capacity=300000,
    critic_update_frequency=64,
    critic_number_of_updates=64,
    target_critic_tau=0.02,
    warmup_steps=10000,
)
```

# How to Register Your Algorithm

The Agent itself is a concrete class that uses three components to implement any RL algorithm. By defining these three components you define your RL algorithm. The following provides a conceptual overview of the structure. For further details refer to the Docstring in the code.

## The Three Components

### 1. Data Collector

The _Data Collector_ receives the raw data that you pass to the agent in the `agent.step` function (plus the last performed action which is automatically added by the agent).
This data itself is usually not usable in that form. Most RL algorithms require transitions which contain two observations, the previous and the next one. On-policy algorithms often differ in how they format their data. The _Data Collector_ is responsible for formatting the data such that it can be used for the actual update.

This does **not** contain things like an experience replay buffer. The formatted data can then be accessed via a dependency by the updatable components; there is no direct passing of information via function arguments.

For an example, look at `GymnasiumTransitionDataCollector` in `data_collection/transition.py` which creates regular transitions.

### 2. Update Rule

As the name suggests, the _Update Rule_ is responsible for performing updates.
The _Update Rule_ orchestrates a collection of _Updatable Components_ which are responsible for updating specific elements of the algorithm.
These components are things like value functions, actors, target networks and also replay buffers.

The _Update Rule_ itself is mostly responsible for initializing the updatable components and providing all required dependencies. Importantly, an _Update Rule_ defines a tuple of updatable components which are updated in the order they appear in the tuple.

The _Updatable Components_ need to define two important things:

- The `update()` function which performs the actual update of that component and returns a dictionary containing potentially relevant logging information (this is added to the agent info).
- The property `update_condition` which returns a boolean indicating whether this component should be updated at the current point in time. The update condition can for example depend on a frequency according to the environment steps.

The `update()` function does not take any arguments. All required dependencies should be provided during initialization, e.g., access to the replay buffer if needed etc.

As an example, the update rule of DQN consists of three updatable components:

1. Replay Buffer Update - adds data from the _Data Collector_ to the replay buffer.
2. Value Function Update - updates the value function according to DQN using data from the replay buffer.
3. Target Network Update - performs a soft or hard update of the target network.

These components can be mixed and matched to create other algorithms. The replay buffer and target network updates, for example, are also used in DDPG, TD3, and SAC.

### 3. Policy Builder

The _Policy Builder_ constructs policies which are responsible for selecting an action for the agent.
It has separate functions for creating training and evaluation policies which are used during training or evaluation mode of the agent. It offers an option to rebuild policies if needed, e.g., when a network structure changes or an embedding has to be recalculated.

The Policies themselves must implement the `act(observation)` function, which takes an observation and returns an action as well as a policy info dictionary. The policy info can be used for logging as it is returned as part of the agent info, or can also be picked up by the _Data Collector_ e.g., for recording action log probabilities.

## Algorithm Registration

After defining the three components of your algorithm, write a function following the _Component Factory_ protocol (defined in `algorithms/registry.py`). This function should take action and observation Space objects as well as a dictionary for the configuration as input and should return a _Data Collector_, _Update Rule_, and _Policy Builder_ in that order as a tuple.

Additionally, define a default configuration which contains the default settings of your algorithm.

Finally register your algorithm with:

```python
athlete.register(
    id="my_new_algorithm",
    component_factory=make_my_algorithm_components,
    default_configuration=MY_DEFAULT_CONFIGURATION
)
```

Now you can use `athlete.make` to create an agent of your algorithm and override the default configuration as described above.

# Roadmap

Athlete is in early stages of development. Currently I am planning to add the following features in the future:

- **Vector Agents** - Agents that are compatible with Vector environments, either to train multiple agents in parallel or to train a single agent with multiple environments.
- **Jax Implementation** - Likely connected to the point above, to improve performance by reimplementing algorithms using Jax.
- **Policy-only saving option** - Currently the `save_checkpoint()` function saves every part of the training process, which is great if you might want to continue training at some point, but impractical if you're simply interested in saving the resulting policy.
- **Refactoring to reduce evolved complexity** - I've already spent a lot of time removing and simplifying code that has naturally grown over time. This is a continuous effort to improve readability and extensibility of the code.

Further features might be added as they are needed. There will be breaking changes, if you use Athlete as a dependency, make sure to use a strict version requirement.

# Feedback/Contribution

I'm happy about any feedback for athlete if you tried it. What issues did you have, what do you think could be simplified?
Simply open a Github issue and let's talk :)
